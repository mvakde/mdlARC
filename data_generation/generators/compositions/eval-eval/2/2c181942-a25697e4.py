"""Color relay hub: simultaneous slot fitting, parity, and clockwise addresses.

Only the standard library and NumPy are used. No files are read or written by
generation, solving, validation, or self_test. Returned answers are painted
from constructed rigid keys and independently checked by input-only search.
"""
from __future__ import annotations

import hashlib
import numpy as np

DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError('require integer counts 2..10 train, 1..4 test, train > test')
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid size'
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), 'invalid cells'
    return a.astype(np.int8, copy=True)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(frozenset(part))
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(cells, op):
    out = []
    for r, c in cells:
        if op >= 4:
            c = -c
        for _ in range(op % 4):
            r, c = -c, r
        out.append((r, c))
    return frozenset(out)


def _transform(regions, op):
    transformed = tuple(_raw(s, op) for s in regions)
    r, c, _, _ = _bbox(set().union(*transformed))
    return tuple(_shift(s, -r, -c) for s in transformed)


def _signature(cells):
    r, c, _, _ = _bbox(cells)
    return tuple(sorted(_shift(cells, -r, -c)))


def _canonical(cells):
    return min(_signature(_raw(cells, op)) for op in range(8))


def _array_key(a):
    return a.shape, a.tobytes()


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size // 2, 'background must occupy a strict majority'
    parts = _components(map(tuple, np.argwhere(a != bg)))
    cores, keys = [], []
    for cells in parts:
        colors = sorted({int(a[p]) for p in cells})
        if len(colors) >= 4:
            cores.append((cells, colors))
        else:
            assert len(colors) == 2, 'loose objects are two-colored keys'
            regions = tuple(frozenset(p for p in cells if a[p] == c) for c in colors)
            assert all(len(_components(s)) == 1 for s in regions), 'disconnected key region'
            keys.append(dict(cells=cells, colors=tuple(colors), regions=regions))
    assert len(cores) == 1, 'one multicolored hub required'
    core, colors = cores[0]
    r0, c0, r1, c1 = _bbox(core)
    rectangle = frozenset((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    assert all(a[p] == bg or p in core for p in rectangle), 'key inside hub rectangle'
    frame = max(colors, key=lambda c: sum(a[p] == c for p in core))
    labels = [c for c in colors if c != frame]
    assert len(labels) in (3, 4), 'three or four active ports'
    label_cells = {c: frozenset(p for p in core if a[p] == c) for c in labels}
    assert all(len(s) == 2 and len(_components(s)) == 1 for s in label_cells.values()), 'labels are dominoes'
    slots = _components(rectangle - core)
    assert len(slots) == len(labels), 'one slot per label'
    ports, used_labels = [], set()
    for slot in slots:
        sides = [i for i, touch in enumerate((any(r == r0 for r, c in slot),
                  any(c == c1 for r, c in slot), any(r == r1 for r, c in slot),
                  any(c == c0 for r, c in slot))) if touch]
        assert len(sides) == 1, 'slot must open on exactly one hub side'
        side = sides[0]
        dr, dc = DIR4[side]
        # The deepest pair touches its address domino in the inward direction.
        adjacent = [(color, cells) for color, cells in label_cells.items()
                    if _shift(cells, dr, dc) <= slot]
        assert len(adjacent) == 1, 'slot needs its own inward address'
        color, cells = adjacent[0]
        edge = (max(r for r, c in slot) if side == 0 else min(c for r, c in slot)
                if side == 1 else min(r for r, c in slot) if side == 2
                else max(c for r, c in slot))
        inner = frozenset((r, c) for r, c in slot if (r if side % 2 == 0 else c) == edge)
        assert inner == _shift(cells, dr, dc), 'inward slot end must be the attachment pair'
        assert color not in used_labels
        used_labels.add(color)
        ports.append(dict(slot=slot, side=side, color=color, label=cells))
    ports.sort(key=lambda p: p['side'])
    assert len({p['side'] for p in ports}) == len(ports), 'at most one port per side'
    for i, port in enumerate(ports):
        port['next_color'] = ports[(i + 1) % len(ports)]['color']
    assert 3 <= len(keys) <= 6
    return dict(grid=a, bg=bg, core=core, rectangle=rectangle, frame=frame,
                ports=ports, keys=keys, bbox=(r0, c0, r1, c1))


def _candidates(info, use_addresses=True, swap_reflections=True, clockwise=True):
    a, rectangle = info['grid'], info['rectangle']
    result = []
    for pi, port in enumerate(info['ports']):
        slot = port['slot']
        sr, sc, _, _ = _bbox(slot)
        options = {}
        successor = (port['next_color'] if clockwise else
                     info['ports'][(pi - 1) % len(info['ports'])]['color'])
        for ki, key in enumerate(info['keys']):
            for op in range(8):
                regions = _transform(key['regions'], op)
                colors = key['colors'][::-1] if op >= 4 and swap_reflections else key['colors']
                for plug_index in (0, 1):
                    if use_addresses and (colors[plug_index] != port['color'] or
                                          colors[1 - plug_index] != successor):
                        continue
                    pr, pc, _, _ = _bbox(regions[plug_index])
                    placed_regions = tuple(_shift(s, sr - pr, sc - pc) for s in regions)
                    if placed_regions[plug_index] != slot:
                        continue
                    handle = placed_regions[1 - plug_index]
                    if handle & rectangle:
                        continue
                    dr, dc = DIR4[port['side']]
                    r0, c0, r1, c1 = info['bbox']
                    # The second region lies entirely beyond this port's side.
                    if not all((r < r0 if dr == -1 else r > r1 if dr == 1 else
                                c < c0 if dc == -1 else c > c1) for r, c in handle):
                        continue
                    full = placed_regions[0] | placed_regions[1]
                    if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in full):
                        continue
                    paint = tuple(sorted((r, c, colors[i]) for i, s in enumerate(placed_regions) for r, c in s))
                    sig = ki, paint
                    options.setdefault(sig, dict(key=ki, op=op, plug_index=plug_index,
                        regions=placed_regions, colors=colors, full=full, paint=paint, port=pi))
        result.append(list(options.values()))
    return result


def _answers(info, use_addresses=True, swap_reflections=True, clockwise=True, limit=2):
    options = _candidates(info, use_addresses, swap_reflections, clockwise)
    order = sorted(range(len(options)), key=lambda i: len(options[i]))
    solutions, seen = [], set()

    def search(depth, used, occupied, moves):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            signature = tuple(sorted(p for m in moves for p in m['paint']))
            if signature not in seen:
                seen.add(signature)
                out = np.full_like(info['grid'], info['bg'])
                for p in info['core']:
                    out[p] = info['grid'][p]
                for r, c, color in signature:
                    out[r, c] = color
                solutions.append((out, sorted(moves, key=lambda m: m['port'])))
            return
        for move in options[order[depth]]:
            if move['key'] in used or move['full'] & occupied:
                continue
            search(depth + 1, used | {move['key']}, occupied | move['full'], moves + [move])

    search(0, set(), set(info['core']), [])
    return solutions


def _parity_geometry_witness(info, moves):
    """A color endpoint selects between different mirror-related footprints."""
    options = _candidates(info, use_addresses=False)
    return any(candidate['key'] == move['key']
               and candidate['plug_index'] == move['plug_index']
               and (candidate['op'] >= 4) != (move['op'] >= 4)
               and candidate['full'] != move['full']
               for move in moves for candidate in options[move['port']])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Search input geometry and visible color-cycle constraints only."""
    answers = _answers(_parse(input_grid))
    assert len(answers) == 1, 'hub has no unique colored assembly'
    return answers[0][0]


def _shape(rng, small=False):
    width = int(rng.choice((4, 6)))
    depth = int(rng.integers(2, 4 if small else 5))
    center = width // 2
    plug = {(r, c) for r in range(depth) for c in (center - 1, center)}
    style = int(rng.integers(3))
    body = int(rng.integers(1, depth))
    for r in range(body):
        if style == 0:
            lo, hi = 0, width
        elif style == 1:
            lo, hi = max(0, center - 2 - (body - r)), min(width, center + 1 + (body - r))
        else:
            lo, hi = int(rng.integers(center)), int(rng.integers(center + 1, width + 1))
        plug.update((r, c) for c in range(lo, hi))
    # Both transverse extremes are real cells, while the attachment stays narrow.
    plug.update((0, c) for c in range(width))
    relationship = int(rng.integers(3))
    size = (int(rng.integers(2, max(3, len(plug)))) if relationship == 0 else
            len(plug) if relationship == 1 else len(plug) + int(rng.integers(1, 5)))
    size = min(18, size)
    start = (-1, int(rng.integers(width)))
    handle = {start}
    tip = start
    for _ in range(size - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIR4
                           if -4 <= r + dr <= -1 and -1 <= c + dc <= width} - handle)
        weights = np.array([(4 if abs(r - tip[0]) + abs(c - tip[1]) == 1 else 1)
                            if style == 1 else (1 + sum((r + dr, c + dc) in handle for dr, dc in DIR4))
                            if style == 0 else 1 for r, c in frontier], dtype=float)
        tip = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        handle.add(tip)
    return frozenset(plug), frozenset(handle), depth, width


def _port_regions(shape, side, h, w, offset=0, mirrored=False):
    plug, handle, depth, width = shape
    label = frozenset((depth, c) for c in (width // 2 - 1, width // 2))
    regions = (plug, handle, label)
    if mirrored:
        regions = tuple(frozenset((r, width - 1 - c) for r, c in s) for s in regions)
    # Rotate a top port clockwise around the hub; offset is along its edge.
    if side == 0:
        return tuple(_shift(s, 0, w // 2 - width // 2 + offset) for s in regions)
    if side == 1:
        return tuple(frozenset((h // 2 - width // 2 + c + offset, w - 1 - r) for r, c in s) for s in regions)
    if side == 2:
        return tuple(frozenset((h - 1 - r, w // 2 + width // 2 - 1 - c + offset) for r, c in s) for s in regions)
    return tuple(frozenset((h // 2 + width // 2 - 1 - c + offset, r) for r, c in s) for s in regions)


def _pack(rng, canvas, core_cells, core_origin, loose):
    h, w = canvas
    blocked = np.zeros(canvas, dtype=bool)

    def block(cells):
        for r, c in cells:
            blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True

    # Reserve the complete hub rectangle, including empty sockets.
    block(_shift(core_cells, *core_origin))
    placements = [None] * len(loose)
    order = sorted(range(len(loose)), key=lambda i: -len(loose[i][0] | loose[i][1]) + rng.random())
    for i in order:
        full = loose[i][0] | loose[i][1]
        rr, cc = np.array(sorted(full), dtype=int).T
        ph, pw = int(rr.max()) + 1, int(cc.max()) + 1
        if ph > h or pw > w:
            return None
        tops, lefts = np.indices((h - ph + 1, w - pw + 1))
        rows, cols = tops.ravel()[:, None] + rr, lefts.ravel()[:, None] + cc
        good = np.flatnonzero(~blocked[rows, cols].any(axis=1))
        if not len(good):
            return None
        j = int(rng.choice(good))
        pos = int(tops.ravel()[j]), int(lefts.ravel()[j])
        placements[i] = tuple(_shift(s, *pos) for s in loose[i])
        block(placements[i][0] | placements[i][1])
    return placements


def _build_pair(rng, label_cycle, n, witnesses, decoys, fallback=False, strong_witness=False):
    sides = sorted(map(int, rng.choice(4, n, replace=False)))
    labels = list(label_cycle)
    k = int(rng.integers(n))
    labels = labels[k:] + labels[:k]
    bg, frame = map(int, rng.choice([c for c in range(10) if c not in labels], 2, replace=False))
    h, w = (10, 10) if fallback else (int(rng.integers(11, 16)), int(rng.integers(11, 16)))
    rectangle = frozenset((r, c) for r in range(h) for c in range(w))
    if fallback:
        shared = (frozenset(((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2))),
                  frozenset(((-1, 0), (-2, 0), (-2, 1))), 2, 4)
    else:
        shared = _shape(rng)
        # Unequal region sizes give unambiguous parity witnesses in training.
        if witnesses and len(shared[0]) == len(shared[1]):
            shared = (shared[0], frozenset(((-1, 0), (-2, 0), (-2, 1))), shared[2], shared[3])
    shapes = [shared, shared] + [shared if rng.random() < .35 else _shape(rng) for _ in range(n - 2)]
    destinations, holes, addresses = [], set(), set()
    for i, side in enumerate(sides):
        port = _port_regions(shapes[i], side, h, w, 0 if fallback else int(rng.integers(-1, 2)),
                             bool(rng.integers(2)))
        plug, handle, label = port
        assert (plug | label) <= rectangle
        assert not (plug | label) & (holes | addresses), 'ports collide'
        holes.update(plug)
        addresses.update(label)
        destinations.append(port)
    core = rectangle - holes
    assert len(_components(core)) == 1
    assert len(_components(holes)) == n, 'slots merge'
    expected, loose, local_colors = [], [], []
    for i, (plug, handle, label) in enumerate(destinations):
        pc, hc = labels[i], labels[(i + 1) % n]
        parity = i % 2 if witnesses and i < 2 else int(rng.integers(2))
        op = int(rng.integers(4)) + 4 * parity
        regions = _transform((plug, handle), op)
        loose.append(regions)
        local_colors.append((hc, pc) if parity else (pc, hc))
        expected.append((plug, handle, label, pc, hc))
    # A decoy may have a legitimate adjacent color pair but an incompatible plug.
    for di in range(decoys):
        if di == 0:
            color_pair = (labels[0], labels[1])
            # Both regions have area 1 or 2, whereas all socket plugs have >=6.
            regions = (frozenset(((0, 0), (1, 0))), frozenset(((1, 1),)))
        else:
            color_pair = (labels[0], int(rng.choice([c for c in range(10) if c not in labels and c not in (bg, frame)])))
            regions = (shared[0], shared[1])
        loose.append(_transform(regions, int(rng.integers(8))))
        local_colors.append(color_pair)
    extent = set(rectangle)
    for plug, handle, label, pc, hc in expected:
        extent.update(handle)
    er0, ec0, er1, ec1 = _bbox(extent)
    for attempt in range(18):
        if fallback or attempt >= 12:
            canvas = (30, 30)
        else:
            canvas = (int(rng.integers(max(18, er1 - er0 + 1), 31)),
                      int(rng.integers(max(18, ec1 - ec0 + 1), 31)))
        low = (-er0, -ec0)
        high = (canvas[0] - 1 - er1, canvas[1] - 1 - ec1)
        if any(a > b for a, b in zip(low, high)):
            continue
        origin = tuple(int(rng.choice((a, b))) if rng.random() < .35 else int(rng.integers(a, b + 1))
                       for a, b in zip(low, high))
        placements = _pack(rng, canvas, rectangle, origin, loose)
        if placements is None:
            continue
        inp = np.full(canvas, bg, dtype=np.int8)
        out = inp.copy()
        _paint(inp, _shift(core, *origin), frame)
        _paint(out, _shift(core, *origin), frame)
        for plug, handle, label, pc, hc in expected:
            _paint(inp, _shift(label, *origin), pc)
            _paint(out, _shift(label, *origin), pc)
            _paint(out, _shift(plug, *origin), pc)
            _paint(out, _shift(handle, *origin), hc)
        for regions, colors in zip(placements, local_colors):
            for region, color in zip(regions, colors):
                _paint(inp, region, color)
        if np.count_nonzero(inp == bg) <= inp.size // 2:
            continue
        info = _parse(inp)
        answers = _answers(info)
        if len(answers) != 1 or not np.array_equal(answers[0][0], out):
            continue
        # Geometric ambiguity, parity, and filtering are demonstrated explicitly.
        if witnesses:
            if len(_answers(info, use_addresses=False)) < 2:
                continue
            moves = answers[0][1]
            if {int(m['op'] >= 4) for m in moves} != {0, 1}:
                continue
            if any(np.array_equal(x[0], out) for x in _answers(info, swap_reflections=False)):
                continue
            if strong_witness and not _parity_geometry_witness(info, moves):
                continue
        return dict(input=inp, output=out)
    raise AssertionError('packing or uniqueness failed')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 4, replace=False)))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            n = (3 + i if split == 'train' and i < 2 else int(rng.integers(3, 5)))
            if split == 'train':
                indices = sorted(map(int, rng.choice(4, n, replace=False)))
            else:
                # A new directed edge palette[2] -> palette[1] cannot occur in
                # any training cycle that follows the base cyclic order.
                indices = [0, 2, 1, 3] if n == 4 else list(rng.choice(((0, 2, 1), (2, 1, 3))))
            cycle = [palette[j] for j in indices]
            decoys = (1 if i == 0 else int(rng.integers(3))) if split == 'train' else int(rng.integers(1, 3))
            for attempt in range(65):
                try:
                    pair = _build_pair(rng, cycle, n, split == 'train', decoys,
                                       fallback=attempt >= 50, strong_witness=split == 'train' and i == 0)
                except AssertionError:
                    continue
                key = _array_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded generation exhausted')
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_edges, test_edges, train_counts = set(), set(), set(), set()
    parity_set, shape_witness, filter_witness, mirror_witness = set(), False, False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _array_key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _parse(inp)
            answers = _answers(info)
            assert len(answers) == 1 and np.array_equal(answers[0][0], out), 'wrong or ambiguous answer'
            moves = answers[0][1]
            assert len(moves) == len(info['ports'])
            assert all(out[p] == inp[p] for p in info['core']), 'fixed hub changed'
            assert all(out[p] != info['bg'] for p in info['rectangle']), 'hub not completed'
            selected = {m['key'] for m in moves}
            assert len(selected) == len(moves)
            for move in moves:
                port = info['ports'][move['port']]
                assert all(out[p] == port['color'] for p in port['slot'])
                assert all(out[p] == port['next_color'] for p in move['full'] - port['slot'])
                original = info['keys'][move['key']]
                assert len(original['cells']) == len(move['full']), 'rigid key lost cells'
            edges = {(p['color'], p['next_color']) for p in info['ports']}
            (train_edges if split == 'train' else test_edges).update(edges)
            if split == 'train':
                train_counts.add(len(moves))
                parity_set.update(int(m['op'] >= 4) for m in moves)
                shape_witness |= len(_answers(info, use_addresses=False)) >= 2
                mirror_witness |= _parity_geometry_witness(info, moves)
                bad = _answers(info, swap_reflections=False)
                assert not any(np.array_equal(x[0], out) for x in bad), 'reflection has no visible consequence'
                # Exact-color decoy: correct neighboring colors, incompatible geometry.
                filter_witness |= any(set(k['colors']) == {p['color'], p['next_color']}
                    for i, k in enumerate(info['keys']) if i not in selected for p in info['ports'])
    assert train_counts == {3, 4}, 'demonstrate both circuit sizes'
    assert parity_set == {0, 1} and shape_witness and filter_witness and mirror_witness
    assert test_edges - train_edges, 'test must rewire the demonstrated color circuit'


def self_test():
    """Bounded streaming checks over 42 seeds; no puzzle or trace archive."""
    dimensions, slots, handles, palettes, operations, sizes, backgrounds = set(), set(), set(), set(), set(), set(), set()
    fingerprints, relations, side_sets, decoy_counts = set(), set(), set(), set()
    total_pairs, ablations, mirror_witnesses = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                info = _parse(inp)
                answers = _answers(info)
                assert len(answers) == 1 and np.array_equal(solve(inp.copy()), out)
                moves = answers[0][1]
                dimensions.add(inp.shape)
                palettes.add(tuple(p['color'] for p in info['ports']))
                backgrounds.add(info['bg'])
                side_sets.add(tuple(p['side'] for p in info['ports']))
                decoy_counts.add(len(info['keys']) - len(moves))
                sizes.add(len(moves))
                for move in moves:
                    plug, handle = move['regions'][move['plug_index']], move['regions'][1 - move['plug_index']]
                    slots.add(_canonical(plug))
                    handles.add(_canonical(handle))
                    relations.add(int(np.sign(len(plug) - len(handle))))
                    operations.add(move['op'])
                if split == 'train':
                    assert not any(np.array_equal(x[0], out) for x in _answers(info, clockwise=False)), 'counterclockwise shortcut'
                    ablations += len(_answers(info, use_addresses=False)) >= 2
                    mirror_witnesses += _parity_geometry_witness(info, moves)
                total_pairs += 1
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and total_pairs == 210
    assert len(dimensions) >= 35 and len(slots) >= 15 and len(handles) >= 45
    assert len(palettes) >= 80 and len(backgrounds) >= 8
    assert operations == set(range(8)) and sizes == {3, 4}
    assert relations == {-1, 0, 1} and decoy_counts == {0, 1, 2}
    assert len(side_sets) == 5 and ablations >= 42
    # All 30 legal explicit count combinations, including the tightest budget.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(11000 + 10 * a + b, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    first, again = generate_puzzle(913, 2, 1), generate_puzzle(913, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    # Whole-scene rotations preserve clockwise order; arbitrary numeric colors
    # must also preserve the answer after the same bijective relabeling.
    recolor = np.random.default_rng(391).permutation(10).astype(np.int8)
    pair = first['test'][0]
    for turns in range(4):
        inp = recolor[np.rot90(pair['input'], turns)]
        expected = recolor[np.rot90(pair['output'], turns)]
        assert np.array_equal(solve(inp), expected)
    changed = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    changed['test'][0]['output'][0, 0] = (int(changed['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return dict(seeds=42, pairs=total_pairs, grid_dimensions=len(dimensions), slot_shapes=len(slots),
                handle_shapes=len(handles), palettes=len(palettes), operations=sorted(operations),
                area_relations=sorted(relations), address_ablation_witnesses=ablations,
                mirror_geometry_witnesses=mirror_witnesses)
