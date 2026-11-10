"""Keyed edge-fed queues: geometric eligibility and stable edge priority interact.

Only standard-library Python and NumPy are used. No parent code is imported.
The solver reads roots, frames, pin geometry, and priority from the input alone.
"""
import json
from collections import Counter

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b):
        raise ValueError("counts must be integers")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=True)


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        q = min(todo)
        todo.remove(q)
        stack, found = [q], {q}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _decode(g):
    """Canonical horizontal decoder; pins may be disconnected from each other.

    Each pin is adjacent to the complete frame, so spatial components are whole
    cargos, even when two cargos have the same payload and frame colors.
    """
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= g.size:
        raise ValueError("background must occupy over half the input")
    objects, marker_colors = [], set()
    for comp in _components(map(tuple, np.argwhere(g != bg).tolist())):
        top = min(r for r, c in comp)
        bottom = max(r for r, c in comp)
        columns = sorted(c for r, c in comp if r == top)
        left, right = columns[0], columns[-1]
        h, w = bottom - top + 1, right - left + 1
        if h < 4 or w < 3 or columns != list(range(left, right + 1)):
            raise ValueError("no complete cargo rectangle")
        frame = int(g[top, left])
        patch = g[top:bottom + 1, left:right + 1].copy()
        if (not np.all(patch[0] == frame) or not np.all(patch[-1] == frame)
                or not np.all(patch[:, 0] == frame) or not np.all(patch[:, -1] == frame)
                or np.any((patch[1:-1, 1:-1] == frame) | (patch[1:-1, 1:-1] == bg))):
            raise ValueError("invalid frame or interior")
        rectangle = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}
        if not rectangle <= comp:
            raise ValueError("background hole inside cargo")
        extra = comp - rectangle
        low = sorted(r - top for r, c in extra if c == left - 1)
        high = sorted(r - top for r, c in extra if c == right + 1)
        if len(low) + len(high) != len(extra):
            raise ValueError("pins must lie on the two horizontal sides")
        for rows in (low, high):
            if rows and (len(rows) != 2 or not 1 <= rows[0] < rows[1] <= h - 2):
                raise ValueError("each port must contain two pins beside interior rows")
        marker_colors.update(int(g[q]) for q in extra)
        cells = {(r - top, c - left): int(g[r, c]) for r, c in comp}
        objects.append(dict(r=top, c=left, h=h, w=w, frame=frame, patch=patch,
                            low=tuple(low), high=tuple(high), cells=cells))
    if not objects or len(marker_colors) != 1:
        raise ValueError("one common pin color is required")
    marker = marker_colors.pop()
    if marker == bg or any(o['frame'] == marker or np.any(o['patch'] == marker) for o in objects):
        raise ValueError("pin color cannot be used inside a cargo")
    roots = {}
    for i, o in enumerate(objects):
        direction = None
        if o['c'] == 0 and not o['low']:
            direction = 0
        if o['c'] + o['w'] == g.shape[1] and not o['high']:
            if direction is not None:
                raise ValueError("root cannot span the canvas")
            direction = 1
        if direction is not None:
            if o['frame'] in roots:
                raise ValueError("multiple roots for a frame")
            roots[o['frame']] = (i, direction)
    if set(roots) != {o['frame'] for o in objects} or len(roots) > 2:
        raise ValueError("every frame class requires exactly one edge root")
    if len({d for i, d in roots.values()}) != len(roots):
        raise ValueError("at most one root at each edge")
    for frame, (root, direction) in roots.items():
        distances = []
        for i, o in enumerate(objects):
            if o['frame'] != frame or i == root:
                continue
            distance = o['c'] if direction == 0 else g.shape[1] - o['c'] - o['w']
            if distance <= 0 or not (o['low'] if direction == 0 else o['high']):
                raise ValueError("nonroot requires an inward port and an interior source position")
            distances.append(distance)
        if len(distances) != len(set(distances)):
            raise ValueError("source priorities may not tie within a frame class")
    return bg, marker, objects, roots


def _signature(rows):
    return tuple(r - rows[0] for r in rows) if rows else ()


def _render(shape, bg, marker, objects, positions, complete=True):
    out = np.full(shape, bg, dtype=np.int8)
    owners, solid = Counter(), set()
    for i, (r, c) in positions.items():
        o = objects[i]
        for (dr, dc), color in o['cells'].items():
            q = r + dr, c + dc
            if not (0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]):
                raise ValueError("docking exceeds the boundary")
            if color == marker:
                if q in solid:
                    raise ValueError("pin intersects solid cargo")
                owners[q] += 1
                if owners[q] > 2:
                    raise ValueError("more than two pin owners")
            else:
                if q in solid or owners[q]:
                    raise ValueError("solid cargo overlaps")
                solid.add(q)
            out[q] = color
    if complete and owners and set(owners.values()) != {2}:
        raise ValueError("all final pins must be shared by exactly two cargos")
    return out


def _assemble(g, decoded=None, priority_override=None):
    """One frontier per frame. Matching geometry filters the stable priority queue."""
    bg, marker, objects, roots = decoded if decoded is not None else _decode(g)
    positions, records = {}, []
    for frame, (root, direction) in sorted(roots.items()):
        inward, outward = ('low', 'high') if direction == 0 else ('high', 'low')
        pending = {i for i, o in enumerate(objects) if o['frame'] == frame and i != root}
        positions[root] = objects[root]['r'], objects[root]['c']
        current, order, skipped = root, [root], []
        distance = {i: (objects[i]['c'] if direction == 0 else
                       g.shape[1] - objects[i]['c'] - objects[i]['w']) for i in pending}
        if priority_override and frame in priority_override:
            distance = {i: priority_override[frame].index(i) for i in pending}
        while pending:
            port = objects[current][outward]
            if not port:
                raise ValueError("queue terminated before using all cargos")
            eligible = [i for i in pending if _signature(objects[i][inward]) == _signature(port)]
            if not eligible:
                raise ValueError("no matching cargo for exposed socket")
            chosen = min(eligible, key=distance.__getitem__)
            nearest = min(pending, key=distance.__getitem__)
            if nearest != chosen:
                skipped.append((nearest, chosen))
            o, parent = objects[chosen], objects[current]
            pr, pc = positions[current]
            joint_c = pc + parent['w'] if direction == 0 else pc - 1
            r = pr + port[0] - o[inward][0]
            c = joint_c + 1 if direction == 0 else joint_c - o['w']
            positions[chosen] = r, c
            pending.remove(chosen)
            current = chosen
            order.append(chosen)
        if objects[current][outward]:
            raise ValueError("unused terminal socket")
        records.append(dict(frame=frame, direction=direction, root=root,
                            order=order, skipped=skipped))
    output = _render(g.shape, bg, marker, objects, positions)
    return output, positions, records


def _read(g):
    candidates = []
    for axis, canonical in ((1, g), (0, g.T)):
        try:
            decoded = _decode(canonical)
        except ValueError:
            continue
        candidates.append((axis, canonical, decoded))
    if len(candidates) != 1:
        raise ValueError("input must identify one movement axis")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    axis, canonical, decoded = _read(g)
    out, _, _ = _assemble(canonical, decoded)
    return out.copy() if axis == 1 else out.T.copy()


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _cargo(rng, frame, marker, bg, palette, h, w, low, high):
    patch = np.full((h, w), frame, dtype=np.int8)
    interior = np.full((h - 2, w - 2), _pick(rng, palette), dtype=np.int8)
    if len(palette) > 1 and interior.size > 1 and rng.random() < 0.4:
        rr, cc = np.indices(interior.shape)
        a, b = rng.choice(palette, 2, replace=False)
        interior[:] = np.where((rr + cc) % 2, a, b)
    patch[1:-1, 1:-1] = interior
    cells = {(r, c): int(patch[r, c]) for r in range(h) for c in range(w)}
    cells.update({(r, -1): marker for r in low})
    cells.update({(r, w): marker for r in high})
    return dict(r=0, c=0, h=h, w=w, frame=frame, patch=patch,
                low=tuple(low), high=tuple(high), cells=cells)


def _solid_interior(o, color):
    o['patch'][1:-1, 1:-1] = color
    for r in range(1, o['h'] - 1):
        for c in range(1, o['w'] - 1):
            o['cells'][r, c] = int(color)


def _blueprint(rng, n, direction, frame, marker, bg, palette, purpose):
    if purpose == 'repeat':
        gaps = [2] * (n - 1)
    elif purpose == 'switch':
        gaps = [1, 2, 1]  # n=4: a closer gap-2 cargo must wait initially.
    else:
        gaps = [int(x) for x in rng.integers(1, 4, size=n - 1)]
    objects = []
    for i in range(n):
        h, w = int(rng.integers(5, 10)), int(rng.integers(3, 6))
        h = max(h, (gaps[i - 1] + 3 if i else 5),
                (gaps[i] + 3 if i < n - 1 else 5))
        # Small widths are useful but not a fixed output lattice.
        if rng.random() < 0.55:
            w = 3
        ports = []
        for gap in (gaps[i - 1] if i else None, gaps[i] if i < n - 1 else None):
            if gap is None:
                ports.append(())
            else:
                r = int(rng.integers(1, h - gap - 1))
                ports.append((r, r + gap))
        inside, outside = ports
        low, high = (inside, outside) if direction == 0 else (outside, inside)
        objects.append(_cargo(rng, frame, marker, bg, palette, h, w, low, high))
    if purpose == 'repeat':
        # Different carried contents make exchanging the two eligible middles visible.
        for i, color in ((1, palette[0]), (2, palette[1])):
            _solid_interior(objects[i], color)
    return objects


def _logical_positions(objects, direction, height, width, rng):
    inward, outward = ('low', 'high') if direction == 0 else ('high', 'low')
    relative = [(0, 0)]
    for a, b in zip(objects, objects[1:]):
        r, c = relative[-1]
        r += a[outward][0] - b[inward][0]
        c += a['w'] + 1 if direction == 0 else -b['w'] - 1
        relative.append((r, c))
    top = min(r for r, c in relative)
    bottom = max(r + o['h'] for o, (r, c) in zip(objects, relative))
    if bottom - top > height:
        raise ValueError("queue is too tall")
    base_r = int(rng.integers(-top, height - bottom + 1))
    base_c = 0 if direction == 0 else width - objects[0]['w']
    return [(r + base_r, c + base_c) for r, c in relative]


def _scatter(rng, shape, groups, logical, purposes, bg, marker):
    """Scattered source ranks preserve within-key order, not global chain order."""
    height, width = shape
    objects = [o for group in groups for o in group]
    starts, total = [], 0
    for group in groups:
        starts.append(total)
        total += len(group)
    for attempt in range(150):
        positions, forbidden = {}, set()
        for direction, group in enumerate(groups):
            if not group:
                continue
            i = starts[direction]
            positions[i] = logical[i]
            r, c = positions[i]
            cells = {(r + a, c + b) for a, b in group[0]['cells']}
            if cells & forbidden:
                break
            forbidden |= {(a + dr, b + dc) for a, b in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        else:
            planned = []
            for direction, group in enumerate(groups):
                if len(group) <= 1:
                    continue
                sequence = list(range(1, len(group)))
                rng.shuffle(sequence)
                inward = 'low' if direction == 0 else 'high'
                # Preserve source priority only among equal inlet geometries.
                for key in sorted({_signature(o[inward]) for o in group[1:]}):
                    slots = [j for j, i in enumerate(sequence) if _signature(group[i][inward]) == key]
                    members = sorted(sequence[j] for j in slots)
                    for j, i in zip(slots, members):
                        sequence[j] = i
                if purposes[direction] == 'switch':
                    sequence = [2, 1, 3]
                available = np.arange(2, width - max(o['w'] for o in group) - 1)
                distances = sorted(int(x) for x in rng.choice(available, len(sequence), replace=False))
                for i, distance in zip(sequence, distances):
                    o = group[i]
                    c = distance if direction == 0 else width - o['w'] - distance
                    planned.append((starts[direction] + i, c))
            rng.shuffle(planned)
            planned.sort(key=lambda x: -len(objects[x[0]]['cells']))
            for i, c in planned:
                o = objects[i]
                for trial in range(160):
                    r = int(rng.integers(height - o['h'] + 1))
                    cells = {(r + a, c + b) for a, b in o['cells']}
                    if cells & forbidden:
                        continue
                    positions[i] = r, c
                    forbidden |= {(a + dr, b + dc) for a, b in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                    break
                else:
                    break
            if len(positions) == len(objects):
                inp = _render(shape, bg, marker, objects, positions, complete=False)
                if np.count_nonzero(inp == bg) * 2 > inp.size:
                    return inp
    raise ValueError("no readable source scatter")


def _a_witness(canonical, decoded, records):
    """B alone permits exchanging two repeated-key middles; A selects their order."""
    expected = _assemble(canonical, decoded)[0]
    objects = decoded[2]
    for rec in records:
        order, d = rec['order'], rec['direction']
        inward, outward = ('low', 'high') if d == 0 else ('high', 'low')
        for k in range(1, len(order) - 2):
            a, b = objects[order[k]], objects[order[k + 1]]
            keys = [_signature(a[inward]), _signature(a[outward]),
                    _signature(b[inward]), _signature(b[outward])]
            if not keys[0] or len(set(keys)) != 1:
                continue
            alternative_order = order[1:].copy()
            alternative_order[k - 1], alternative_order[k] = alternative_order[k], alternative_order[k - 1]
            try:
                alt = _assemble(canonical, decoded, {rec['frame']: alternative_order})[0]
            except ValueError:
                continue
            if not np.array_equal(expected, alt):
                return True
    return False


def _make_pair(rng, params, purpose, main, test=False):
    frames, marker, bgs, palette, axis = params
    for attempt in range(250):
        bg = _pick(rng, bgs)
        colors = [c for c in palette if c != bg]
        if purpose in ('repeat', 'switch'):
            sizes = [2, 2]
            sizes[main] = 4
        elif test:
            sizes = [int(rng.integers(0, 4)), int(rng.integers(0, 4))]
            sizes[main] = int(rng.integers(5, 7))
        else:
            sizes = [int(rng.integers(0, 5)), int(rng.integers(0, 5))]
            if sum(sizes) < 2 or max(sizes) < 2:
                continue
        purposes = ['free', 'free']
        purposes[main] = purpose
        groups = [_blueprint(rng, n, d, frames[d], marker, bg, colors, purposes[d]) if n else []
                  for d, n in enumerate(sizes)]
        if purpose == 'repeat':
            # The same carried hue follows both edge instructions.
            _solid_interior(groups[1 - main][1], colors[0])
        objects = [o for group in groups for o in group]
        height = int(rng.integers(20, 31))
        width = int(rng.integers(22, 31))
        if rng.random() < 0.2:
            height = width = max(height, width)
        logical, offset = {}, 0
        try:
            for d, group in enumerate(groups):
                if group:
                    for j, pos in enumerate(_logical_positions(group, d, height, width, rng)):
                        logical[offset + j] = pos
                offset += len(group)
            out = _render((height, width), bg, marker, objects, logical)
            inp = _scatter(rng, (height, width), groups, logical, purposes, bg, marker)
            decoded = _decode(inp)
            predicted, positions, records = _assemble(inp, decoded)
            if not np.array_equal(predicted, out) or np.array_equal(inp, out):
                continue
            if purpose == 'repeat' and not _a_witness(inp, decoded, records):
                continue
            if purpose == 'switch' and not any(r['skipped'] for r in records):
                continue
            if axis == 0:
                inp, out = inp.T.copy(), out.T.copy()
            return dict(input=inp, output=out)
        except ValueError:
            continue
    raise RuntimeError("bounded construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    role_colors = [int(x) for x in rng.permutation(10)]
    frames = role_colors[:2]
    marker = role_colors[2]
    backgrounds = role_colors[3:3 + int(rng.integers(1, 3))]
    palette = role_colors[5:]
    params = frames, marker, backgrounds, palette, int(rng.integers(2))
    main = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for index in range(count):
            purpose = 'repeat' if split == 'train' and index == 0 else 'switch' if split == 'train' and index == 1 else 'free'
            direction = main if index % 2 == 0 else 1 - main
            for attempt in range(40):
                pair = _make_pair(rng, params, purpose, direction, split == 'test')
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            g = _grid(pair['input'])
            axis, canonical, decoded = _read(g)
            bg, marker, objects, roots = decoded
            predicted, positions, queues = _assemble(canonical, decoded)
            records.append(dict(split=split, axis=axis, shape=list(g.shape), background=bg,
                                marker=marker, frames=sorted(roots), count=len(objects),
                                queue_lengths=sorted(len(r['order']) for r in queues),
                                gaps=sorted({rows[1] - rows[0] for o in objects for rows in (o['low'], o['high']) if rows}),
                                footprints=sorted({(o['h'], o['w']) for o in objects}),
                                skipped=sum(len(r['skipped']) for r in queues),
                                a_witness=_a_witness(canonical, decoded, queues),
                                row_shifts=sum(positions[i][0] != o['r'] for i, o in enumerate(objects)),
                                payloads=sorted({int(c) for o in objects for c in np.unique(o['patch'][1:-1, 1:-1])})))
    return records


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, axes, conventions = set(), set(), []
    witness_a, witness_b, shared_payload = False, False, False
    max_train, min_test = 0, 100
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            key = g.shape, g.tobytes()
            assert key not in seen, 'duplicate input'
            seen.add(key)
            axis, canonical, decoded = _read(g)
            axes.add(axis)
            bg, marker, objects, roots = decoded
            expected, positions, records = _assemble(canonical, decoded)
            actual = out if axis == 1 else out.T
            assert np.array_equal(expected, actual), 'wrong keyed queue assembly'
            assert np.count_nonzero(g == marker) == 2 * np.count_nonzero(out == marker)
            # Independently inspect all shared joint coordinates and rigid patches.
            pin_owners = Counter()
            solids = set()
            for i, o in enumerate(objects):
                r, c = positions[i]
                assert np.array_equal(actual[r:r + o['h'], c:c + o['w']], o['patch'])
                solid = {(r + a, c + b) for a in range(o['h']) for b in range(o['w'])}
                assert not solids & solid
                solids |= solid
                for rows, col in ((o['low'], c - 1), (o['high'], c + o['w'])):
                    for row in rows:
                        pin_owners[r + row, col] += 1
            assert set(pin_owners.values()) == {2}
            assert not set(pin_owners) & solids
            for frame, (root, direction) in roots.items():
                assert positions[root] == (objects[root]['r'], objects[root]['c'])
            for rec in records:
                inward, outward = ('low', 'high') if rec['direction'] == 0 else ('high', 'low')
                remaining = set(rec['order'][1:])
                for parent, chosen in zip(rec['order'], rec['order'][1:]):
                    eligible = [i for i in remaining if _signature(objects[i][inward])
                                == _signature(objects[parent][outward])]
                    distances = {i: (objects[i]['c'] if rec['direction'] == 0 else
                                     canonical.shape[1] - objects[i]['c'] - objects[i]['w'])
                                 for i in eligible}
                    assert chosen == min(eligible, key=distances.__getitem__), 'unstable eligible priority'
                    remaining.remove(chosen)
            mapping = {frame: d for frame, (i, d) in roots.items()}
            conventions.append((marker, mapping))
            length = max(len(r['order']) for r in records)
            if split == 'train':
                max_train = max(max_train, length)
                witness_a |= _a_witness(canonical, decoded, records)
                witness_b |= any(r['skipped'] for r in records)
                hues = {}
                for o in objects:
                    for color in np.unique(o['patch'][1:-1, 1:-1]):
                        hues.setdefault(int(color), set()).add(o['frame'])
                shared_payload |= any(len(frames) == 2 for frames in hues.values())
            else:
                min_test = min(min_test, length)
    assert len(axes) == 1, 'axis changes within puzzle'
    assert len({m for m, mapping in conventions}) == 1, 'marker convention changes'
    frame_map = {}
    for marker, mapping in conventions:
        for frame, direction in mapping.items():
            assert frame not in frame_map or frame_map[frame] == direction
            frame_map[frame] = direction
    assert len(frame_map) == 2
    assert witness_a and witness_b, 'training must distinguish both inherited concepts'
    assert shared_payload, 'training must separate payload hue from edge instruction'
    assert min_test > max_train, 'tests must extend demonstrated queue length'


def _fingerprint(p):
    return tuple((s, np.asarray(q['input']).shape, np.asarray(q['input']).tobytes(),
                  np.asarray(q['output']).tobytes()) for s in ('train', 'test') for q in p[s])


def self_test() -> dict:
    """Bounded 42-seed suite, structural measurements and negative checks."""
    corpus = [generate_puzzle(s) for s in range(42)]
    features = [measure_puzzle(p) for p in corpus]
    rows = [r for group in features for r in group]
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert {r['axis'] for r in rows} == {0, 1}
    assert len({tuple(r['shape']) for r in rows}) >= 35
    assert len({tuple(r['frames']) for r in rows}) >= 20
    assert len({r['count'] for r in rows}) >= 5
    assert {g for r in rows for g in r['gaps']} == {1, 2, 3}
    assert len({tuple(f) for r in rows for f in r['footprints']}) >= 12
    assert {1, 2, 3, 4, 5, 6} <= {n for r in rows for n in r['queue_lengths']}
    assert any(len(r['queue_lengths']) == 1 for r in rows)
    assert all(any(r['a_witness'] for r in group if r['split'] == 'train') for group in features)
    assert all(any(r['skipped'] for r in group if r['split'] == 'train') for group in features)
    assert all(len({tuple(r['shape']) for r in group}) > 1 for group in features)
    assert any(r['row_shifts'] >= 4 for r in rows)
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = corpus[0]
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    q = bad['test'][0]
    axis, canonical, decoded = _read(q['input'])
    marker = decoded[1]
    coord = tuple(np.argwhere(q['output'] == marker)[0])
    q['output'][coord] = decoded[0]
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('erased docking pin accepted')
    return dict(seeds=42, pairs=len(rows), shapes=len({tuple(r['shape']) for r in rows}),
                frame_pairs=len({tuple(r['frames']) for r in rows}),
                counts=sorted({r['count'] for r in rows}),
                queue_lengths=sorted({n for r in rows for n in r['queue_lengths']}),
                footprints=len({tuple(f) for r in rows for f in r['footprints']}),
                skipped_pairs=sum(r['skipped'] > 0 for r in rows),
                repeated_key_witnesses=sum(r['a_witness'] for r in rows))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
