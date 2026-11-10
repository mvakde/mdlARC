"""Repair-gated scaffold relays: a single input-only geometric rule.

The constructor authors a clean conductive graph before making damage. The
solver reconstructs that graph from visible evidence. Both use the same signal
propagation primitive; verification is self-authored, not independent approval.
"""

from collections import deque
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 838017589, 505858004, 793411263, 698615939, 649619367,
              864477471, 278340815, 262750652, 510490250, 777654690,
              335383844, 961679389, 402757448, 604402785, 807754665,
              229652868, 344179358, 649605925, 66537359, 511261298,
              117301058, 204365192, 345701630, 548729472, 987430656,
              171806354, 180468608, 170420450, 193212289, 888609290,
              885366361, 239132859, 188124184, 221929441, 253064603,
              279911700, 71764951, 513381610, 631471186, 343635035,
              149057285)


def _step(p, d, n=1):
    dr, dc = DIRS[d]
    return p[0] + n * dr, p[1] + n * dc


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(cells, diagonal=False):
    remaining = set(cells)
    result = []
    steps = (tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c) if diagonal else DIRS)
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        comp, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                nxt = r + dr, c + dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        result.append(comp)
    return result


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _diagonal(comp):
    points = sorted(comp)
    return (len(points) in (2, 3)
            and all(points[i+1][0] - points[i][0] == 1
                    for i in range(len(points)-1))
            and len({points[i+1][1] - points[i][1]
                     for i in range(len(points)-1)}) == 1
            and abs(points[1][1] - points[0][1]) == 1)


def _reconstruct(raw, ports, blocked=None):
    """Detect all cues simultaneously in the original input, never iteratively.

    Noise is a length-2/3 diagonal component of carrier cells having no
    orthogonal support. A two-cell component can certify a bracketed gap.
    The retained anchor and continuation are tested against the original raw
    scaffold, so a newly inferred bridge cannot invent another repair cue.
    """
    occupied = raw | (set() if blocked is None else set(blocked))
    weak = {p for p in raw - ports
            if not any(_step(p, d) in raw for d in range(4))}
    noise_components = [c for c in _components(weak, True) if _diagonal(c)]
    noise = set().union(*noise_components) if noise_components else set()
    core = raw - noise
    pairs = {frozenset(c) for c in noise_components if len(c) == 2}
    repairs, unsupported = [], []
    for anchor in sorted(core):
        for d in (1, 2):
            far = _step(anchor, d, 4)
            gap = {_step(anchor, d, k) for k in (1, 2, 3)}
            if far not in core or gap & occupied:
                continue
            if _step(anchor, (d+2) % 4) not in core and _step(far, d) not in core:
                continue
            matches = []
            dr, dc = DIRS[d]
            for endpoint, toward in ((anchor, 1), (far, -1)):
                for side in (-1, 1):
                    spur = frozenset((endpoint[0] + k*(toward*dr + side*dc),
                                      endpoint[1] + k*(toward*dc + side*dr))
                                     for k in (1, 2))
                    if spur in pairs:
                        matches.append(spur)
            if matches:
                if len(matches) != 1:
                    raise ValueError('ambiguous repair cue')
                repairs.append((gap, matches[0], d))
            else:
                unsupported.append(gap)
    added = set().union(*(x[0] for x in repairs)) if repairs else set()
    if len({x[1] for x in repairs}) != len(repairs):
        raise ValueError('one diagonal cannot certify multiple gaps')
    if sum(len(x[0]) for x in repairs) != len(added):
        raise ValueError('repair gaps must not overlap')
    return core | added, {'repairs': repairs, 'noise': noise,
                          'unsupported': unsupported, 'unrepaired': core}


def _terminal_directions(cells, ports):
    result = {}
    for p in sorted(ports):
        neighbors = [d for d in range(4) if _step(p, d) in cells]
        if len(neighbors) != 1:
            raise ValueError('a marked port must have exactly one conductive neighbor')
        result[p] = (neighbors[0] + 2) % 4
    return result


def _energize(shape, cells, ports, tail, cap, initial, port_directions=None):
    """Union of directed first-contact rays and activated conductive components.

    A component emits each terminal once. This makes cycles and simultaneous
    arrivals order-independent. Only the original cleaned scaffold is opaque;
    already drawn rays cannot change a subsequent collision.
    """
    comps = _components(cells)
    owner = {p: i for i, comp in enumerate(comps) for p in comp}
    outgoing = (_terminal_directions(cells, ports) if port_directions is None
                else port_directions)
    trace, reached, emissions = set(tail) | {cap}, set(), []
    queue = deque([(cap, initial, 1)])
    depths, contacts = {}, []
    rays = []
    while queue:
        origin, direction, depth = queue.popleft()
        p, segment = _step(origin, direction), set()
        while _inside(p, shape) and p not in owner:
            segment.add(p)
            p = _step(p, direction)
        trace |= segment
        rays.append(segment)
        if p not in owner:
            continue
        index = owner[p]
        contacts.append((owner.get(origin), index))
        if index in reached:
            continue
        reached.add(index)
        depths[index] = depth
        trace |= comps[index]
        for port in sorted(comps[index] & ports):
            direction_out = outgoing[port]
            queue.append((port, direction_out, depth+1))
            emissions.append((port, direction_out))
    active_cells = set().union(*(comps[i] for i in reached)) if reached else set()
    return trace, {'components': comps, 'reached': reached,
                   'active_cells': active_cells, 'emissions': emissions,
                   'rays': rays, 'port_directions': outgoing,
                   'depths': depths, 'contacts': contacts}


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError('expected an integer ARC grid of size 1..30')
    return grid


def _decode(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 4 or counts.max() <= inp.size // 2:
        raise ValueError('four distinct roles and a majority background are required')
    bg = int(colors[np.argmax(counts)])
    foreground = {tuple(map(int, p)) for p in np.argwhere(inp != bg)}
    boundary = {p for p in foreground if p[0] in (0, inp.shape[0]-1)
                or p[1] in (0, inp.shape[1]-1)}
    if len(boundary) != 1:
        raise ValueError('only the launcher may touch the boundary')
    start = next(iter(boundary))
    if start[0] in (0, inp.shape[0]-1) and start[1] in (0, inp.shape[1]-1):
        raise ValueError('launcher cannot start at a corner')
    initial = (2 if start[0] == 0 else 0 if start[0] == inp.shape[0]-1
               else 1 if start[1] == 0 else 3)
    signal = int(inp[start])
    tail, p = [], start
    while _inside(p, inp.shape) and int(inp[p]) == signal:
        tail.append(p)
        p = _step(p, initial)
    if len(tail) < 2 or not _inside(p, inp.shape) or int(inp[p]) == bg:
        raise ValueError('launcher needs a signal tail and contrasting cap')
    cap, marker = p, int(inp[p])
    wire_colors = set(map(int, colors)) - {bg, signal, marker}
    if len(wire_colors) != 1:
        raise ValueError('indistinguishable palette roles')
    wire = next(iter(wire_colors))
    if {tuple(map(int, p)) for p in np.argwhere(inp == signal)} != set(tail):
        raise ValueError('signal color occurs outside its launcher')
    launcher = set(tail) | {cap}
    launcher_component = next(c for c in _components(foreground) if start in c)
    if launcher_component != launcher:
        raise ValueError('launcher must be separated from the scaffold')
    raw = foreground - launcher
    ports = {p for p in raw if int(inp[p]) == marker}
    if not ports or not any(int(inp[p]) == wire for p in raw):
        raise ValueError('missing scaffold or terminal ports')
    cells, repair_facts = _reconstruct(raw, ports, launcher)
    _terminal_directions(cells, ports)
    if not all(any(_step(p, d) in repair_facts['unrepaired'] for d in range(4))
               for p in repair_facts['unrepaired']):
        raise ValueError('scaffold pieces need orthogonal support')
    return inp, bg, signal, marker, wire, set(tail), cap, initial, cells, ports, repair_facts


def _predict(input_grid, repair=True, close_all=False):
    inp, bg, signal, marker, wire, tail, cap, initial, cells, ports, repairs = _decode(input_grid)
    directions = _terminal_directions(cells, ports)
    if not repair:
        cells = repairs['unrepaired']
    if close_all:
        cells = cells | set().union(*repairs['unsupported']) if repairs['unsupported'] else cells
    trace, facts = _energize(inp.shape, cells, ports, tail, cap, initial, directions)
    out = np.full(inp.shape, bg, dtype=np.int8)
    for p in trace:
        out[p] = signal
    facts.update(repairs)
    facts.update(shape=inp.shape, palette=(bg, signal, marker, wire), initial=initial,
                 cells=cells, ports=ports, tail=tail, cap=cap, trace=trace)
    return out, facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell using only the supplied input."""
    return _predict(input_grid)[0]


def _line(a, b):
    if a[0] == b[0]:
        return {(a[0], c) for c in range(min(a[1], b[1]), max(a[1], b[1])+1)}
    if a[1] == b[1]:
        return {(r, a[1]) for r in range(min(a[0], b[0]), max(a[0], b[0])+1)}
    raise ValueError('only orthogonal strokes are allowed')


def _main_scaffold(rng, shape, spec):
    """An entry bus with independently shaped terminal paths and optional stubs.

    This is an input construction convenience, not a solver layout assumption.
    Damage can split one apparent assembly into several conductive components.
    """
    h, w = shape
    n = spec['arms']
    spacing = int(rng.integers(4, 7))
    needed = (n-1)*spacing
    if needed + 6 >= h:
        spacing = 4
        needed = (n-1)*spacing
    lo = int(rng.integers(3, h-needed-2))
    rows = [lo + i*spacing for i in range(n)]
    x = int(rng.integers(4, 8))
    bus_lo = max(1, rows[0] - int(rng.integers(0, 3)))
    bus_hi = min(h-2, rows[-1] + int(rng.integers(0, 3)))
    intact = _line((bus_lo, x), (bus_hi, x))
    ports, branches = set(), []
    statuses = list(spec['statuses'])
    rng.shuffle(statuses)
    serial_index = statuses.index('repair') if spec['series'] else -1
    for i, row in enumerate(rows):
        length = int(rng.integers(7, min(13, w-x-5)))
        if spec['series'] and spec['loop']:
            length = int(rng.integers(13, min(17, w-x-2)))
        if i == serial_index:
            length = int(rng.integers(12, min(17, w-x-2)))
        end = (row, x+length)
        path = _line((row, x), end)
        # Different branches independently become straight, L, or hooked paths.
        bend = int(rng.choice([0, 0, -1 if i == 0 else 1])) if i in (0, n-1) else 0
        if bend:
            room = row-1 if bend == -1 else h-2-row
            if room >= 2:
                endpoint = (row + bend*int(rng.integers(2, min(5, room)+1)), end[1])
                path |= _line(end, endpoint)
                end = endpoint
        intact |= path
        ports.add(end)
        if i == serial_index:
            starts = [x+2, x+8]
        else:
            starts = [int(rng.integers(x+2, x+length-3))]
        branches.append((row, starts, statuses[i], end))
    # A visible alternate conductive route can energize the far side of an
    # intentional break without filling that break. It also varies topology.
    if spec['loop']:
        a = next((b for b in branches if b[2] == 'repair'), None)
        b = next((b for b in branches if b[2] in ('broken', 'near_miss')), None)
        if a is None or b is None:
            return None
        left = max(max(a[1]), max(b[1])) + 4
        right = min(a[3][1], b[3][1]) - 1
        if left > right:
            return None
        col = int(rng.integers(left, right+1))
        intact |= _line((a[0], col), (b[0], col))
    # Variable short dead-end branches supply additional connected geometry.
    stub_rows = []
    for row in range(bus_lo, bus_hi+1):
        if (rng.random() < .18 and row not in rows
                and all(abs(row-other) >= 3 for other in stub_rows)):
            intact |= _line((row, x), (row, x-int(rng.integers(1, 3))))
            stub_rows.append(row)
    raw, clean, deviations = set(intact), set(intact), set()
    valid_ports = []
    for row, starts, status, end in branches:
        if status == 'intact':
            continue
        for start in starts:
            gap = {(row, start+k) for k in range(3)}
            raw -= gap
            if status != 'repair':
                clean -= gap
            if status in ('repair', 'near_miss'):
                endpoint_right = bool(rng.integers(2))
                anchor = (row, start+3 if endpoint_right else start-1)
                toward = -1 if endpoint_right else 1
                options = []
                for side in (-1, 1):
                    shift = 0 if status == 'repair' else side
                    spur = {(row+shift+side*k, anchor[1]+toward*k) for k in (1, 2)}
                    if (all(0 < r < h-1 and 0 < c < w-1 for r, c in spur)
                            and not spur & intact and not spur & _halo(deviations)
                            and not any(_step(p, d) in intact-gap for p in spur for d in range(4))):
                        options.append(spur)
                if not options:
                    return None
                spur = options[int(rng.integers(len(options)))]
                deviations |= spur
        if status == 'repair':
            valid_ports.append(end)
    raw |= deviations
    try:
        _terminal_directions(clean, ports)
    except ValueError:
        return None
    # A random entry row can hit a stub, a bus, or the root of a terminal arm.
    entry = int(rng.integers(bus_lo, bus_hi+1))
    tail_length = int(rng.integers(2, x-1))
    tail = {(entry, c) for c in range(tail_length)}
    cap = (entry, tail_length)
    if (tail | {cap}) & _halo(raw):
        return None
    return raw, clean, ports, tail, cap, valid_ports


def _small_relay(rng, incoming, fork):
    """Sample an independently oriented L, T, cross, or stepped open tree."""
    h, w = int(rng.integers(3, 7)), int(rng.integers(3, 8))
    entry_row = int(rng.integers(h))
    pivot = int(rng.integers(1, w-1))
    mask = _line((entry_row, 0), (entry_row, w-1))
    lo = int(rng.integers(entry_row+1))
    hi = int(rng.integers(entry_row, h))
    if lo == hi:
        lo, hi = 0, h-1
    mask |= _line((lo, pivot), (hi, pivot))
    # A secondary arm changes both the footprint and the port alignment.
    if rng.random() < .45:
        level = lo if lo != entry_row else hi
        endpoint = (level, int(rng.choice([0, w-1])))
        mask |= _line((level, pivot), endpoint)
    entry = (entry_row, 0)
    endpoints = [p for p in sorted(mask) if p != entry
                 and sum(_step(p, d) in mask for d in range(4)) == 1]
    if len(endpoints) < (2 if fork else 1):
        return None
    count = int(rng.integers(2 if fork else 1, min(3, len(endpoints))+1))
    chosen = set(endpoints[int(i)] for i in rng.choice(len(endpoints), count, replace=False))
    # Rotate canonical rightward entry to its requested direction.
    for _ in range((1-incoming) % 4):
        rot = lambda p: (w-1-p[1], p[0])
        mask, chosen, entry = {rot(p) for p in mask}, {rot(p) for p in chosen}, rot(entry)
        h, w = w, h
    return mask, chosen, entry


def _place_receiver(rng, shape, clean, raw, ports, pending, fork):
    occupied = _halo(raw)
    options = []
    directions = _terminal_directions(clean, ports)
    for origin in sorted(pending):
        direction = directions[origin]
        for distance in range(2, max(shape)):
            hit = _step(origin, direction, distance)
            if not (0 < hit[0] < shape[0]-1 and 0 < hit[1] < shape[1]-1):
                break
            if hit in clean:
                break
            options.append((origin, direction, hit))
    if not options:
        return None
    for index in rng.permutation(len(options)):
        origin, direction, hit = options[int(index)]
        for _ in range(8):
            sampled = _small_relay(rng, direction, fork)
            if sampled is None:
                continue
            mask, new_ports, entry = sampled
            shift = lambda p: (p[0]+hit[0]-entry[0], p[1]+hit[1]-entry[1])
            mask, new_ports = {shift(p) for p in mask}, {shift(p) for p in new_ports}
            if (mask & occupied or not all(0 < r < shape[0]-1 and 0 < c < shape[1]-1
                                          for r, c in mask)):
                continue
            # Verify that this beam's first contact really is the new component.
            q = _step(origin, direction)
            while _inside(q, shape) and q not in (clean | mask):
                q = _step(q, direction)
            if q not in mask:
                continue
            return mask, new_ports, origin
    return None


def _scatter_decoy(rng, shape, raw, clean, ports):
    for _ in range(50):
        sampled = _small_relay(rng, int(rng.integers(4)), False)
        if sampled is None:
            continue
        mask, new_ports, _ = sampled
        r0, c0 = min(r for r, c in mask), min(c for r, c in mask)
        sh = max(r for r, c in mask)-r0+1
        sw = max(c for r, c in mask)-c0+1
        if sh+2 >= shape[0] or sw+2 >= shape[1]:
            continue
        row, col = int(rng.integers(1, shape[0]-sh)), int(rng.integers(1, shape[1]-sw))
        shift = lambda p: (p[0]-r0+row, p[1]-c0+col)
        mask, new_ports = {shift(p) for p in mask}, {shift(p) for p in new_ports}
        if not mask & _halo(raw):
            return mask, new_ports
    return None


def _make_pair(rng, spec, palette, rotation, reflect):
    # Hold requested semantic complexity fixed while retrying its geometry.
    for _ in range(160):
        shape = (int(rng.integers(18, 31)), int(rng.integers(23, 31)))
        if spec['series']:
            shape = (shape[0], 30)
        main = _main_scaffold(rng, shape, spec)
        if main is None:
            continue
        raw, clean, ports, tail, cap, pending = main
        pending = set(pending)
        success = True
        for i in range(spec['receivers']):
            placed = _place_receiver(rng, shape, clean, raw | tail | {cap}, ports,
                                     pending, fork=i == 0 and spec['fork'])
            if placed is None:
                success = False
                break
            mask, new_ports, origin = placed
            raw |= mask
            clean |= mask
            ports |= new_ports
            pending.discard(origin)
            pending = new_ports if spec['deep'] else pending | new_ports
        if not success:
            continue
        for _ in range(spec['decoys']):
            placed = _scatter_decoy(rng, shape, raw | tail | {cap}, clean, ports)
            if placed is not None:
                mask, new_ports = placed
                raw |= mask
                clean |= mask
                ports |= new_ports
        for _ in range(int(rng.integers(1, 5))):
            for _ in range(20):
                r, c = int(rng.integers(1, shape[0]-1)), int(rng.integers(1, shape[1]-1))
                dc = int(rng.choice([-1, 1]))
                length = int(rng.choice([2, 3]))
                noise = {(r+k, c+k*dc) for k in range(length)}
                if (all(0 < rr < shape[0]-1 and 0 < cc < shape[1]-1 for rr, cc in noise)
                        and not noise & _halo(raw | tail | {cap})):
                    raw |= noise
                    break
        try:
            recovered, _ = _reconstruct(raw, ports, tail | {cap})
        except ValueError:
            continue
        if recovered != clean:
            continue
        if any({p, (p[0]+1, p[1]), (p[0], p[1]+1), (p[0]+1, p[1]+1)} <= clean
               for p in clean):
            continue
        try:
            trace, route = _energize(shape, clean, ports, tail, cap, 1)
        except ValueError:
            continue
        if spec['receivers'] and len(route['reached']) < spec['receivers']+1:
            continue
        if spec['deep'] and max(route['depths'].values(), default=0) < 3:
            continue
        if spec['shallow'] and max(route['depths'].values(), default=0) > 2:
            continue
        if spec['contrast'] and not (clean - route['active_cells']):
            continue
        bg, signal, marker, wire = palette
        inp, out = np.full(shape, bg, np.int8), np.full(shape, bg, np.int8)
        for p in raw:
            inp[p] = marker if p in ports else wire
        for p in tail:
            inp[p] = signal
        inp[cap] = marker
        for p in trace:
            out[p] = signal
        inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
        if reflect:
            inp, out = inp[:, ::-1], out[:, ::-1]
        inp, out = inp.copy(), out.copy()
        try:
            prediction, facts = _predict(inp)
            if not np.array_equal(prediction, out):
                continue
            if spec['contrast']:
                if (np.array_equal(_predict(inp, repair=False)[0], out)
                        or np.array_equal(_predict(inp, close_all=True)[0], out)):
                    continue
            return {'input': inp, 'output': out}
        except ValueError:
            continue
    raise RuntimeError('could not pack the requested scaffold circuit in 160 attempts')


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer from {lo} through {hi}')
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    varying_palette = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    varied_orientation = bool(rng.integers(2))
    shift = str(rng.choice(['series', 'depth', 'branches']))
    loop_mode = rng.random() < .35
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for index in range(count):
            demo = split == 'train' and index < 2
            is_test = split == 'test'
            arms = (3 if is_test and shift == 'branches' else
                    2 if demo else int(rng.integers(1, 3 if shift == 'branches' else 4)))
            if is_test and shift == 'series':
                arms = int(rng.integers(2, 4))
            statuses = ['repair']
            if arms > 1:
                statuses.append('near_miss' if (demo and index == 1) else 'broken')
            while len(statuses) < arms:
                statuses.append(str(rng.choice(['repair', 'intact', 'broken', 'near_miss'])))
            if is_test and shift == 'branches':
                statuses = ['repair', 'broken', 'intact']
            receivers = (2 if is_test and shift == 'depth' else
                         1 if demo else int(rng.integers(0, 3)))
            if not is_test and shift == 'depth':
                receivers = min(1, receivers)
            loop = loop_mode and arms > 1 and ((demo and index == 1) or (not demo and rng.random() < .6))
            spec = {'arms': arms, 'statuses': statuses,
                    'series': is_test and shift == 'series',
                    'receivers': receivers, 'decoys': int(rng.integers(0, 3)),
                    'fork': demo or (is_test and shift == 'branches'), 'contrast': demo,
                    'deep': is_test and shift == 'depth',
                    'shallow': not is_test and shift == 'depth', 'loop': loop}
            if loop and demo:
                spec['decoys'] = max(1, spec['decoys'])
            palette = (tuple(map(int, rng.choice(10, 4, replace=False)))
                       if varying_palette and (is_test or index > 0) else base_palette)
            rotation = int(rng.integers(4)) if varied_orientation else base_rotation
            reflect = bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, spec, palette, rotation, reflect)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('could not construct distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(n)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    n, t = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= t <= 4 and n > t
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            prediction, f = _predict(inp)
            assert np.array_equal(prediction, out), 'incorrect full output'
            bg, signal, marker, wire = f['palette']
            assert set(map(int, np.unique(out))) == {bg, signal}
            assert not np.array_equal(inp, out)
            assert len(_components(f['trace'])) == 1, 'signal must stay connected to its launcher'
            assert not (f['noise'] & f['cells'])
            assert all(len(gap) == 3 and len(spur) == 2 for gap, spur, _ in f['repairs'])
            assert all(out[p] == signal for p in f['active_cells'])
            assert all(out[p] == bg for p in f['cells']-f['active_cells'])
            # No carrier cell outside a recognized deviation may be unsupported.
            assert all(any(_step(p, d) in f['cells'] for d in range(4)) for p in f['cells'])
            # A thin scaffold cannot acquire a filled 2x2 block during repair.
            assert not any({p, (p[0]+1, p[1]), (p[0], p[1]+1),
                            (p[0]+1, p[1]+1)} <= f['cells'] for p in f['cells'])
            if split == 'train':
                train.append((pair, f))
    # Demonstrations must show repair causally changing signal, rejection of
    # indiscriminate gap closure, and off-route deletion, not just cue colors.
    witnesses, gated_ports = [], []
    for pair, f in train:
        out = np.asarray(pair['output'])
        without_repair, no_b = _predict(pair['input'], repair=False)
        witnesses.append(not np.array_equal(without_repair, out)
                         and not np.array_equal(_predict(pair['input'], close_all=True)[0], out)
                         and bool(f['cells']-f['active_cells'])
                         and len(f['reached']) >= 2)
        gated_ports.append(bool({p for p, _ in f['emissions']} - {p for p, _ in no_b['emissions']}))
    assert sum(witnesses) >= 2, 'missing causal demonstrations'
    assert any(gated_ports), 'repairs must unlock an actual emitted port in training'
    assert any(len(comp & f['ports']) >= 2 for _, f in train
               for i, comp in enumerate(f['components']) if i in f['reached'])


def self_test():
    """Bounded 42-seed sweep, causal ablations, symmetry, counts, bad labels."""
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test')
                             for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                out, f = _predict(p['input'])
                assert np.array_equal(out, p['output'])
                records.append((split, f))
        # At least one example per seed checks exact rotation/reflection symmetry.
        p = puzzle['train'][0]
        for turns in range(4):
            a, b = np.rot90(p['input'], turns), np.rot90(p['output'], turns)
            assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
    facts = [f for _, f in records]
    assert len(signatures) == 42
    assert len({f['shape'] for f in facts}) >= 40
    assert {f['initial'] for f in facts} == set(range(4))
    assert {d for f in facts for d in f['port_directions'].values()} == set(range(4))
    assert {axis for f in facts for _, _, axis in f['repairs']} == {1, 2}
    assert len({len(f['repairs']) for f in facts}) >= 3
    assert len({len(f['reached']) for f in facts}) >= 3
    assert len({len(f['ports']) for f in facts}) >= 5
    assert len({f['palette'] for f in facts}) >= 30
    assert all({f['palette'][i] for f in facts} == set(range(10)) for i in range(4))
    assert any(len(f['reached']) >= 3 for s, f in records if s == 'test')
    assert any(len(f['repairs']) >= 2 for s, f in records if s == 'test')
    assert any(len(comp & f['ports']) >= 3 for f in facts
               for i, comp in enumerate(f['components']) if i in f['reached'])
    # An intentional gap can have energized brackets because another route
    # reaches its far side; energized endpoints are not permission to close it.
    bypasses = 0
    for f in facts:
        for gap in f['unsupported']:
            rs, cs = {p[0] for p in gap}, {p[1] for p in gap}
            brackets = ({(next(iter(rs)), min(cs)-1), (next(iter(rs)), max(cs)+1)}
                        if len(rs) == 1 else
                        {(min(rs)-1, next(iter(cs))), (max(rs)+1, next(iter(cs)))})
            bypasses += brackets <= f['active_cells'] and not gap & f['trace']
    assert bypasses > 0
    # Mutation: omit one complete energized repair from a previously valid label.
    p = generate_puzzle(0, 2, 1)
    broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]]
              for s in ('train', 'test')}
    pair = broken['train'][0]
    _, f = _predict(pair['input'])
    gap = next(g for g, _, _ in f['repairs'] if g <= f['active_cells'])
    for cell in gap:
        pair['output'][cell] = f['palette'][0]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing repair accepted')
    # Focused hand-authored reconstruction fixture; no sampled coordinates.
    raw = _line((4, 1), (4, 12)) | _line((4, 1), (8, 1))
    raw -= {(4, 5), (4, 6), (4, 7)}
    raw |= {(3, 5), (2, 6)}
    repaired, detail = _reconstruct(raw, {(4, 12)})
    assert len(detail['repairs']) == 1
    assert repaired == _line((4, 1), (4, 12)) | _line((4, 1), (8, 1))
    # Move the diagonal off its retained anchor: the same bracket stays broken.
    near = raw - {(3, 5), (2, 6)} | {(2, 5), (1, 6)}
    repaired_near, detail = _reconstruct(near, {(4, 12)})
    assert not detail['repairs'] and (4, 6) not in repaired_near
    triple = raw - {(3, 5), (2, 6)} | {(3, 5), (2, 6), (1, 7)}
    assert not _reconstruct(triple, {(4, 12)})[1]['repairs']
    # Exact hand-authored signal fixture: repair unlocks an upward terminal,
    # which reaches a horizontal relay; that terminal redirects to the right.
    inp = np.zeros((15, 20), np.int8)
    wire_cells = (_line((4, 5), (10, 5)) | _line((1, 5), (1, 10))
                  | _line((12, 12), (12, 15)))
    wire_cells -= {(6, 5), (7, 5), (8, 5)}
    wire_cells |= {(8, 6), (7, 7)}
    for p in wire_cells:
        inp[p] = 7
    for p in ((4, 5), (1, 10), (12, 15), (10, 2)):
        inp[p] = 4
    inp[10, :2] = 2
    expected = np.zeros_like(inp)
    expected[10, :6] = 2
    expected[2:11, 5] = 2
    expected[1, 5:] = 2
    assert np.array_equal(solve(inp), expected)
    assert not np.array_equal(_predict(inp, repair=False)[0], expected)
    # Two L relays send a ray back to the first relay. Emitting once yields a
    # closed rectangle, not an infinite recursion or an erased repeated relay.
    cycle = np.zeros((15, 15), np.int8)
    cycle_cells = (_line((4, 4), (4, 8)) | _line((4, 8), (8, 8))
                   | _line((11, 4), (11, 8)) | _line((7, 4), (11, 4)))
    for p in cycle_cells:
        cycle[p] = 7
    cycle[4, :2] = 2
    for p in ((4, 2), (8, 8), (7, 4)):
        cycle[p] = 4
    cycle_expected = np.zeros_like(cycle)
    cycle_expected[4, :9] = 2
    cycle_expected[11, 4:9] = 2
    cycle_expected[4:12, 4] = cycle_expected[4:12, 8] = 2
    assert np.array_equal(solve(cycle), cycle_expected)
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            validate_generated_puzzle(generate_puzzle(31337, n, t))
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(73, 5, 3), generate_puzzle(73, 5, 3)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ('input', 'output'))
    return {'seeds': 42, 'pairs': len(records), 'shapes': len({f['shape'] for f in facts}),
            'palettes': len({f['palette'] for f in facts}),
            'repair_counts': sorted({len(f['repairs']) for f in facts}),
            'active_component_counts': sorted({len(f['reached']) for f in facts}),
            'port_counts': sorted({len(f['ports']) for f in facts}),
            'bypassed_unfilled_gaps': bypasses,
            'independent_solver': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
