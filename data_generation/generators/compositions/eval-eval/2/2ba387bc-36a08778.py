"""ARC family: hollow receivers consume pigment queues carried by branching rays.

Only NumPy and the standard library are used. All semantic information is visible
in the input. See task_spec.md for the object-level collision and queue rules.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _canonical(a, direction):
    return np.rot90(a, direction)


def _actual(a, direction):
    return np.ascontiguousarray(np.rot90(a, -direction))


def _tile(size, color, bg, hollow):
    a = np.full((size, size), color, dtype=np.int8)
    if hollow:
        a[1:-1, 1:-1] = bg
    return a


def _objects(a, bg, guide):
    """Segment spatial instances, rather than grouping cells by color."""
    h, w = a.shape
    pending = set(zip(*np.nonzero((a != bg) & (a != guide))))
    objects = []
    while pending:
        first = min(pending)
        pending.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
                    cells.append(p)
        r = int(min(x for x, _ in cells))
        c = int(min(y for _, y in cells))
        rr = int(max(x for x, _ in cells)) + 1
        cc = int(max(y for _, y in cells)) + 1
        size = rr-r
        _require(size == cc-c and size >= 3, "objects must be square tiles of side >=3")
        block = a[r:rr, c:cc]
        color = int(block[0, 0])
        _require(color not in (bg, guide), "tile role color")
        hollow = not np.all(block == color)
        _require(np.array_equal(block, _tile(size, color, bg, hollow)), "malformed tile")
        neighborhood = a[max(0, r-1):min(h, rr+1), max(0, c-1):min(w, cc+1)].copy()
        neighborhood[r-max(0, r-1):rr-max(0, r-1), c-max(0, c-1):cc-max(0, c-1)] = bg
        _require(np.all((neighborhood == bg) | (neighborhood == guide)),
                 "tiles need a complete one-cell gap from other tiles")
        _require(r >= 2, "tiles cannot overlap entry layers")
        objects.append(dict(top=r, left=c, size=size, color=color, hollow=bool(hollow)))
    objects.sort(key=lambda o: (o['top'], o['left']))
    _require(objects and any(o['hollow'] for o in objects) and any(not o['hollow'] for o in objects),
             "both tile geometries required")
    return objects


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(sum(counts == max(counts)) == 1, "background must be uniquely dominant")
    bg = int(colors[np.argmax(counts)])
    interpretations = []
    for d in range(4):
        a = _canonical(grid, d)
        h, w = a.shape
        if h < 5 or w < 5:
            continue
        for guide in map(int, colors):
            if guide == bg:
                continue
            entries = list(map(int, np.flatnonzero(a[0] == guide)))
            if not entries or any(y-x < 2 for x, y in zip(entries, entries[1:])):
                continue
            mask = np.zeros(a.shape, dtype=bool)
            lengths = []
            for c in entries:
                rows = np.flatnonzero(a[:, c] == guide)
                if not 2 <= len(rows) < h or not np.array_equal(rows, np.arange(len(rows))):
                    break
                mask[:len(rows), c] = True
                lengths.append(len(rows))
            else:
                if not np.array_equal(mask, a == guide):
                    continue
                try:
                    objs = _objects(a, bg, guide)
                except AssertionError:
                    continue
                interpretations.append(dict(grid=a.copy(), direction=d, bg=bg, guide=guide,
                                            entries=entries, lengths=lengths, objects=objs))
    _require(len(interpretations) == 1, "roles/direction must be uniquely identifiable")
    return interpretations[0]


def _merge(mapping, c, tokens, streak=0):
    """State is an object-ID set and a diagnostic source-chain length."""
    old_tokens, old_streak = mapping.get(c, (frozenset(), 0))
    mapping[c] = (old_tokens | frozenset(tokens), max(old_streak, streak))


def _simulate(analysis, variant=None):
    """Forward simulation on immutable footprints; token identity is spatial.

    The optional variants are wrong rules used solely as demonstration checks.
    Rendering never affects the state or the immutable input geometry.
    """
    a = analysis['grid']
    bg, guide = analysis['bg'], analysis['guide']
    h, w = a.shape
    objs = analysis['objects']
    out = a.copy()
    occupancy = np.full(a.shape, -1, dtype=int)
    tops = defaultdict(list)
    for oid, o in enumerate(objs):
        r, c, s = o['top'], o['left'], o['size']
        occupancy[r:r+s, c:c+s] = oid
        tops[r].append(oid)
    active = {c: (frozenset(), 0) for c in analysis['entries']}
    scheduled = defaultdict(dict)
    events = []
    simultaneous = merges = clipped = max_queue = max_serial = 0
    for r in range(h):
        for c, (q, k) in scheduled.pop(r, {}).items():
            if c in active:
                merges += 1
            _merge(active, c, q, k)
        hits = {}
        following = {}
        for c, state in sorted(active.items()):
            oid = int(occupancy[r, c])
            if oid >= 0:
                _require(objs[oid]['top'] == r, "unexpected side-entry into an object")
                hits.setdefault(oid, []).append((c, state))
            else:
                _merge(following, c, *state)
                if a[r, c] == bg:
                    out[r, c] = guide
        simultaneous += len(hits) > 1
        for oid, incoming in sorted(hits.items()):
            o = objs[oid]
            left, size = o['left'], o['size']
            right = left+size-1
            q = frozenset().union(*(state[0] for _, state in incoming))
            incoming_tokens = sorted(q)
            streak = max(state[1] for _, state in incoming)
            max_queue = max(max_queue, len(q))
            hollow = o['hollow'] and variant != 'all_solid'
            consumed = None
            if hollow:
                # Merging copies of the SAME source yields one token. Equal
                # colors from DIFFERENT source instances remain separate.
                if variant == 'deduplicate_colors':
                    unique = {}
                    for token in sorted(q):
                        unique.setdefault(objs[token]['color'], token)
                    q = frozenset(unique.values())
                if q:
                    consumed = max(q) if variant == 'newest_first' else min(q)
                    out[r+1:r+size-1, left+1:right] = (o['color'] if variant == 'own_rim'
                                                    else objs[consumed]['color'])
                    q = q - {consumed}
                # A receiver is a gate through its whole footprint. Incident
                # columns resume beyond the far face; they never outline it.
                for c, _ in incoming:
                    _merge(scheduled[r+size], c, q, 0)
            else:
                q = q | {oid}
                streak += 1
                max_serial = max(max_serial, streak)
                max_queue = max(max_queue, len(q))
                if variant == 'straight':
                    for c, _ in incoming:
                        _merge(scheduled[r+size], c, q, streak)
                else:
                    for c in range(max(0, left-1), min(w, right+2)):
                        if r > 0 and a[r-1, c] == bg and occupancy[r-1, c] == -1:
                            out[r-1, c] = guide
                    flanks = [c for c in (left-1, right+1) if 0 <= c < w]
                    clipped += len(flanks) < 2
                    for c in flanks:
                        if c in following:
                            merges += 1
                        _merge(following, c, q, streak)
                        _require(occupancy[r, c] == -1, "blocked flank")
                        if a[r, c] == bg:
                            out[r, c] = guide
            events.append(dict(object=oid, depth=r, hollow=hollow, incoming=[c for c, _ in incoming],
                               incoming_queues=[sorted(state[0]) for _, state in incoming],
                               received_tokens=incoming_tokens, tokens=sorted(q), consumed=consumed))
        active = following
    reached = {e['object'] for e in events}
    stats = dict(events=events, reached=reached, unreached=set(range(len(objs))) - reached,
                 filled=sum(e['consumed'] is not None for e in events),
                 empty=sum(e['hollow'] and e['consumed'] is None for e in events),
                 simultaneous=int(simultaneous), merges=merges, clipped=clipped,
                 max_queue=max_queue, max_serial=max_serial)
    if variant == 'global_pairing':
        # Keep the correct trace, but substitute parent A's global rank match
        # without causal delivery. An initially empty receiver refutes this.
        frames = [o for o in objs if o['hollow']]
        sources = [o for o in objs if not o['hollow']]
        for rank, o in enumerate(frames):
            r, c, s = o['top'], o['left'], o['size']
            color = sources[rank]['color'] if rank < len(sources) else bg
            out[r+1:r+s-1, c+1:c+s-1] = color
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    analysis = _parse(input_grid)
    output, _ = _simulate(analysis)
    return _actual(output, analysis['direction'])


def _put(a, objs, r, c, s, color, hollow, bg):
    a[r:r+s, c:c+s] = _tile(s, color, bg, hollow)
    objs.append(dict(top=r, left=c, size=s, color=color, hollow=hollow))


def _contains(rng, c, size, width):
    return int(rng.integers(max(0, c-size+1), min(c, width-size)+1))


def _depths(rng, sizes):
    """Unequal forward spacing under the 30-cell limit."""
    required = 2 + sum(sizes) + len(sizes)-1 + 1
    _require(required <= 30, "core does not fit")
    extra = int(rng.integers(0, 31-required))
    slack = rng.multinomial(extra, np.ones(len(sizes)+1)/(len(sizes)+1))
    tops, r = [], 2+int(slack[0])
    for i, size in enumerate(sizes):
        tops.append(r)
        r += size + (1+int(slack[i+1]) if i+1 < len(sizes) else 0)
    return tops, r+1+int(slack[-1])


def _stream(rng, bg, guide, colors, source_count, receiver_count, witness=False, test=False):
    """A freely drifting causal spine; each solid chooses an actual flank."""
    width = int(rng.integers(15, 31))
    prefix = bool(witness or (not test and rng.random() < .35))
    tail = bool(witness or (rng.random() < .3 and source_count+receiver_count+prefix < 7))
    core = [False]*source_count + [True]*receiver_count
    if not witness and not test and source_count == 2 and rng.random() < .5:
        core = [False] + list(map(bool, rng.permutation([False]+[True]*receiver_count)))
        if not core[-1] and len(core)+prefix < 7:
            tail = True
    kinds = ([True] if prefix else []) + core + ([True] if tail else [])
    count = len(kinds)
    # Variable scales in the core, including a constant receiver footprint to
    # make depletion explicit without unconsumed parallel copies re-entering.
    receiver_size = int(rng.integers(3, 5)) if count <= 6 else 3
    sizes = [receiver_size if k else int(rng.integers(3, 5)) for k in kinds]
    while 2+sum(sizes)+count > 30:
        sizes = [3]*count
        receiver_size = 3
    tops, height = _depths(rng, sizes)
    a = np.full((height, width), bg, dtype=np.int8)
    target = int(rng.integers(width))
    entry = target
    stub = min(tops[0], int(rng.integers(2, 4)))
    a[:stub, entry] = guide
    objs = []
    src_colors = [int(rng.choice(colors)) for _ in range(source_count)]
    if source_count >= 2:
        src_colors[1] = int(rng.choice([c for c in colors if c != src_colors[0]]))
    receiver_left = None
    src_index = recv_index = 0
    carried = []
    expected = []
    for i, (hollow, s, r) in enumerate(zip(kinds, sizes, tops)):
        if hollow and receiver_left is not None:
            left = receiver_left
        else:
            left = _contains(rng, target, s, width)
        color = src_colors[src_index] if not hollow else int(rng.choice(colors))
        if hollow and carried and recv_index == 0:
            color = src_colors[0]  # same literal color in both shape roles
        if hollow and carried and recv_index == 1 and source_count >= 2:
            color = src_colors[0]  # second payload differs from this rim
        _put(a, objs, r, left, s, color, hollow, bg)
        if hollow:
            receiver_left = left
            fill = carried.pop(0) if carried else None
            if fill is not None:
                recv_index += 1
            expected.append((r, left, s, fill))
        else:
            carried.append(src_colors[src_index])
            src_index += 1
            receiver_left = None
            flanks = [c for c in (left-1, left+s) if 0 <= c < width]
            target = int(rng.choice(flanks))
    if not witness and rng.random() < .45:
        entries = [entry]
        for c in map(int, rng.permutation(width)):
            if all(abs(c-x) >= 2 for x in entries) and np.all(a[:2, c] == bg):
                a[:2, c] = guide
                entries.append(c)
                if len(entries) == int(rng.integers(2, 4)):
                    break
                if len(entries) >= 3:
                    break
    return a, objs, expected


def _collector(rng, bg, guide, colors):
    """Three concurrent sources teach ordered union, token identity and depletion."""
    source_sizes = list(map(int, rng.integers(3, 5, size=3)))
    gaps = list(map(int, rng.integers(1, 4, size=2)))
    origin = int(rng.integers(0, 5))
    lefts = [origin, origin+source_sizes[0]+gaps[0]]
    lefts.append(lefts[1]+source_sizes[1]+gaps[1])
    inner = sorted({lefts[0]+source_sizes[0], lefts[1]-1,
                    lefts[1]+source_sizes[1], lefts[2]-1})
    collect_left = inner[0]
    collect_size = inner[-1]-inner[0]+1
    width_min = lefts[-1]+source_sizes[-1]
    width = int(rng.integers(max(15, width_min), 31))
    sizes = [max(source_sizes), collect_size, 3, 3, 3]
    tops, height = _depths(rng, sizes)
    a = np.full((height, width), bg, dtype=np.int8)
    objs, expected = [], []
    ca, cb = map(int, rng.choice(colors, 2, replace=False))
    for left, s, color in zip(lefts, source_sizes, (ca, cb, cb)):
        entry = int(rng.integers(left, left+s))
        stub = min(tops[0], int(rng.integers(2, 4)))
        a[:stub, entry] = guide
        _put(a, objs, tops[0], left, s, color, False, bg)
    _put(a, objs, tops[1], collect_left, collect_size, ca, True, bg)
    expected.append((tops[1], collect_left, collect_size, ca))
    target = int(rng.choice(inner))
    receiver_left = _contains(rng, target, 3, width)
    for i, (r, fill) in enumerate(zip(tops[2:], (cb, cb, None))):
        rim = ca if i == 0 else int(rng.choice(colors))
        _put(a, objs, r, receiver_left, 3, rim, True, bg)
        expected.append((r, receiver_left, 3, fill))
    return a, objs, expected


def _fits(a, objs, r, c, s, bg):
    h, w = a.shape
    if r < 2 or r+s > h or c < 0 or c+s > w:
        return False
    if not np.all(a[max(0, r-1):min(h, r+s+1), max(0, c-1):min(w, c+s+1)] == bg):
        return False
    return all(r+s < o['top'] or o['top']+o['size'] < r or
               c+s < o['left'] or o['left']+o['size'] < c for o in objs)


def _expected_interiors(out, expected, bg):
    return all(np.all(out[r+1:r+s-1, c+1:c+s-1] == (bg if fill is None else fill))
               for r, c, s, fill in expected)


def _decorate(rng, a, objs, expected, bg, guide, colors, training):
    """Add free-position objects, keeping the teaching spine's visible outcomes.

    Added objects may be reached on other branches or remain inert. This is
    rejection on a semantic witness, not a shelf/site layout restriction.
    """
    desired = int(rng.integers(1, 5))
    added = 0
    for _ in range(90):
        if added >= desired:
            break
        s = int(rng.integers(3, min(7, *a.shape)+1))
        r = int(rng.integers(2, a.shape[0]-s+1))
        c = int(rng.integers(0, a.shape[1]-s+1))
        if not _fits(a, objs, r, c, s, bg):
            continue
        o = dict(top=r, left=c, size=s, color=int(rng.choice(colors)), hollow=bool(rng.integers(2)))
        trial = a.copy()
        trial[r:r+s, c:c+s] = _tile(s, o['color'], bg, o['hollow'])
        try:
            analysis = _parse(trial)
            out, stats = _simulate(analysis)
        except AssertionError:
            continue
        if not _expected_interiors(out, expected, bg):
            continue
        if training and stats['max_serial'] > 2:
            continue
        a, objs = trial, objs+[o]
        added += 1
    return a


def _make_pair(rng, style, bg, guide, colors, direction, seen):
    for attempt in range(80):
        try:
            if style == 'collector':
                a, objs, expected = _collector(rng, bg, guide, colors)
            elif style == 'fifo':
                a, objs, expected = _stream(rng, bg, guide, colors, 2, 2, witness=True)
            elif style == 'test':
                a, objs, expected = _stream(rng, bg, guide, colors, 3, int(rng.integers(1, 4)), test=True)
            else:
                a, objs, expected = _stream(rng, bg, guide, colors,
                                           int(rng.integers(1, 3)), int(rng.integers(1, 4)))
            if attempt < 60:
                a = _decorate(rng, a, objs, expected, bg, guide, colors, style != 'test')
            analysis = _parse(a)
            out, stats = _simulate(analysis)
        except AssertionError:
            continue
        if not _expected_interiors(out, expected, bg):
            continue
        if style != 'test' and stats['max_serial'] > 2:
            continue
        if style == 'test' and stats['max_serial'] < 3:
            continue
        if style == 'fifo' and not (stats['empty'] >= 2 and stats['filled'] >= 2):
            continue
        if style == 'collector' and not (stats['max_queue'] >= 3 and stats['filled'] >= 3 and stats['empty']):
            continue
        if style in ('fifo', 'collector') and attempt < 60 and not stats['unreached']:
            continue
        # Shape-first, FIFO and branching are visibly distinguished by these
        # actual outputs, not merely by counts of nominal events.
        variants = ('all_solid', 'newest_first', 'straight', 'own_rim', 'global_pairing') if style == 'fifo' else ()
        if any(np.array_equal(out, _simulate(analysis, v)[0]) for v in variants):
            continue
        if style == 'collector' and np.array_equal(out, _simulate(analysis, 'deduplicate_colors')[0]):
            continue
        inp = _actual(a, direction)
        if _key(inp) in seen:
            continue
        return {'input': inp, 'output': _actual(out, direction)}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    bg, guide = palette[:2]
    colors = palette[2:2+int(rng.integers(2, 7))]
    mixed_palette = bool(rng.random() < .4)
    direction = int(rng.integers(4))
    mixed_direction = bool(rng.random() < .7)
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+nq):
        training = index < nt
        style = ('fifo' if index == 0 else 'collector' if index == 1 else 'general') if training else 'test'
        local_bg, local_guide, local_colors = bg, guide, colors
        if mixed_palette and index > 1:
            local = list(map(int, rng.permutation(10)))
            local_bg, local_guide = local[:2]
            local_colors = local[2:2+int(rng.integers(2, 7))]
        d = (direction+index) % 4 if mixed_direction else direction
        pair = _make_pair(rng, style, local_bg, local_guide, local_colors, d, seen)
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training, testing = set(), [], []
    witnesses = {v: False for v in ('all_solid', 'newest_first', 'straight', 'deduplicate_colors',
                                  'own_rim', 'global_pairing')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "output extent/change")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _parse(inp)
            expected, stats = _simulate(a)
            actual = _canonical(out, a['direction'])
            _require(np.array_equal(actual, expected), "incorrect queue propagation")
            _require(np.array_equal(out[inp != a['bg']], inp[inp != a['bg']]), "original pigment/rims/stubs changed")
            # All changed non-guide cells are inside reached hollow cavities
            # and have the consumed SOURCE's color, not the receiver's color.
            allowed = np.zeros(inp.shape, dtype=bool)
            for e in stats['events']:
                o = a['objects'][e['object']]
                if e['consumed'] is not None:
                    r, c, s = o['top'], o['left'], o['size']
                    allowed_can = np.zeros(expected.shape, dtype=bool)
                    allowed_can[r+1:r+s-1, c+1:c+s-1] = True
                    allowed |= _actual(allowed_can, a['direction']).astype(bool)
            _require(np.all((out == inp) | (out == a['guide']) | allowed), "unexpected repainting")
            for oid in stats['unreached']:
                o = a['objects'][oid]
                r, c, s = o['top'], o['left'], o['size']
                _require(np.array_equal(actual[r:r+s, c:c+s], a['grid'][r:r+s, c:c+s]), "inert object changed")
            if split == 'train':
                training.append((a, stats))
                for v in witnesses:
                    witnesses[v] |= not np.array_equal(_simulate(a, v)[0], actual)
            else:
                testing.append(stats)
    _require(all(witnesses.values()), "missing observable concept/order/identity demonstration")
    _require(any(s['empty'] >= 2 for _, s in training), "teach empty receivers before and after supply")
    _require(any(s['simultaneous'] and s['max_queue'] >= 3 for _, s in training), "teach concurrent inventory union")
    _require(any(any(len([t for q in e['incoming_queues'] for t in q]) > len(e['received_tokens'])
                     and e['hollow'] for e in s['events']) for _, s in training),
             "teach merging copies of the same spatial source")
    _require(any(s['unreached'] for _, s in training), "teach causal distractors")
    _require(all(s['max_serial'] <= 2 for _, s in training), "training serial-depth contract")
    _require(all(s['max_serial'] >= 3 for s in testing), "test must extend causal source depth")


def _signature(p):
    return tuple(tuple((_key(_grid(x['input'])), _key(_grid(x['output']))) for x in p[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded semantic checks and structural variation over 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, layouts, palettes, directions, scales, totals, entries, fill_counts = (set() for _ in range(8))
    reached_extras = clipped = multi_hits = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _parse(pair['input'])
                out, stats = _simulate(a)
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve")
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list input")
                shapes.add(pair['input'].shape)
                layouts.add(tuple((o['top'], o['left'], o['size'], o['hollow']) for o in a['objects']))
                palettes.add((a['bg'], a['guide']))
                directions.add(a['direction'])
                scales.update(o['size'] for o in a['objects'])
                totals.add(len(a['objects']))
                entries.add(len(a['entries']))
                fill_counts.add(stats['filled'])
                clipped += bool(stats['clipped'])
                multi_hits += any(len(e['incoming']) > 1 for e in stats['events'])
                reached_extras += len(stats['events']) > 7
                # Rotation and palette permutation are equivariances, which
                # test local role recognition without any new generation.
                if split == 'test':
                    _require(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output'])), "rotation")
                    mapping = np.arange(10, dtype=np.int8)[::-1]
                    _require(np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']]), "palette")
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity")
    _require(len(shapes) >= 35 and len(layouts) >= 180, "structural diversity")
    _require(len(palettes) >= 25 and directions == {0, 1, 2, 3}, "role/orientation diversity")
    _require({3, 4, 5, 6, 7} <= scales and len(scales) >= 6, "tile scale diversity")
    _require(len(totals) >= 6 and {1, 2, 3} <= entries and len(fill_counts) >= 3, "count diversity")
    _require(clipped > 5 and multi_hits > 40, "branch topology diversity")
    _require(_signature(generate_puzzle(99)) == _signature(generate_puzzle(99)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(0, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('invalid counts accepted')
    damaged = {s: [{k: _grid(v) for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in p),
                shapes=len(shapes), layouts=len(layouts), palettes=len(palettes),
                directions=sorted(directions), sizes=sorted(scales), tile_counts=sorted(totals),
                entry_counts=sorted(entries), filled_counts=sorted(fill_counts),
                clipped_pairs=clipped, multiple_hit_pairs=multi_hits, reached_extra_pairs=reached_extras)
