"""Cut a connected two-colored source into corner-owned, rigid socket keys.

Only NumPy and the standard library are used. No file I/O occurs during generation,
solving, validation or self_test; all searches and diagnostic storage are bounded.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))  # TL, TR, BR, BL
CORNER_OPS = ((0, 7), (1, 4), (2, 5), (3, 6))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(seed, num_train, num_test):
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(x):
    a = np.asarray(x)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("invalid ARC grid")
    return a


def _cells(mask):
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _neighbors(p):
    return tuple((p[0] + dr, p[1] + dc) for dr, dc in DIR4)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        found, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in todo:
                    todo.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(frozenset(found))
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _shift(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _raw(p, op):
    r, c = p
    return ((r, c), (c, -r), (-r, -c), (-c, r),
            (r, -c), (-c, -r), (-r, c), (c, r))[op]


def _transform(regions, op):
    raw = [frozenset(_raw(p, op) for p in part) for part in regions]
    r0, _, c0, _ = _bbox(frozenset().union(*raw))
    return tuple(_shift(part, -r0, -c0) for part in raw)


def _signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted(_shift(cells, -r0, -c0)))


def _canonical(cells):
    return min(_signature(_transform((cells,), op)[0]) for op in range(8))


def _orbit(cells):
    return len({_signature(_transform((cells,), op)[0]) for op in range(8)})


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _analyze(a):
    a = _grid(a)
    h, w = a.shape
    assert min(h, w) >= 9, "four sockets need room"
    rim = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    vals, counts = np.unique(rim, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(vals[np.argmax(counts)])
    target = int(a[0, 0])
    assert target != bg and all(int(a[r, c]) == target
                               for r, c in ((0, 0), (0, w-1), (h-1, w-1), (h-1, 0)))
    marks = set(map(int, vals)) - {bg, target}
    assert len(marks) == 1, "marker is the remaining border color"
    marker = marks.pop()
    mp = _cells(a == marker)
    assert 1 <= len(mp) <= 2
    axes, sides = [None, None], set()
    for r, c in sorted(mp):
        flags = [s for yes, s in ((r == 0, 0), (c == w-1, 1),
                                  (r == h-1, 2), (c == 0, 3)) if yes]
        assert len(flags) == 1 and flags[0] not in sides
        side = flags[0]
        sides.add(side)
        axes[1 if side % 2 == 0 else 0] = c if side % 2 == 0 else r
    assert len(sides) == 1 or ({s % 2 for s in sides} == {0, 1})
    colors = tuple(sorted(set(map(int, np.unique(a))) - {bg, target, marker}))
    assert len(colors) == 2
    source = _cells(np.isin(a, colors))
    assert source and len(_components(source)) == 1
    assert all(0 < r < h-1 and 0 < c < w-1 for r, c in source)
    ar, ac = axes
    fixed = frozenset(p for p in source if p[0] == ar or p[1] == ac)
    assert fixed
    groups = {}
    for r, c in source - fixed:
        vertical = (int(r > ar) if ar is not None else int(2 in sides))
        horizontal = (int(c > ac) if ac is not None else int(1 in sides))
        groups.setdefault((vertical, horizontal), set()).add((r, c))
    assert len(groups) == (2 if len(sides) == 1 else 4)
    for corner, cells in groups.items():
        regions = tuple(frozenset(p for p in cells if int(a[p]) == col) for col in colors)
        assert all(regions) and all(len(_components(x)) == 1 for x in regions)
        assert len(_components(cells)) == 1
        groups[corner] = regions
    active = {corner for corner in groups
              if (0 if corner[0] == 0 else 2) in sides
              or (3 if corner[1] == 0 else 1) in sides}
    assert len(active) == (2 if len(sides) == 1 else 3)
    targets, slots, rectangles = {}, {}, {}
    all_targets = _cells(a == target)
    for corner in CORNERS:
        part = frozenset((r, c) for r, c in all_targets
                         if (int(2*r >= h), int(2*c >= w)) == corner)
        assert part and len(_components(part)) == 2
        r0, r1, c0, c1 = _bbox(part)
        assert 2 <= r1-r0+1 < h/2 and 2 <= c1-c0+1 < w/2
        assert (r0 == 0 if corner[0] == 0 else r1 == h-1)
        assert (c0 == 0 if corner[1] == 0 else c1 == w-1)
        rect = frozenset(itertools.product(range(r0, r1+1), range(c0, c1+1)))
        slot = rect - part
        assert slot and len(_components(slot)) == 1
        assert all(int(a[p]) == bg for p in slot)
        targets[corner], slots[corner], rectangles[corner] = part, slot, rect
    paths, internal_gaps = [], 0
    for side in sorted(sides):
        axis = 0 if side % 2 == 0 else 1
        coordinate = ac if axis == 0 else ar
        hits = [p[axis] for p in source if p[1-axis] == coordinate]
        assert hits
        internal_gaps += sum(v not in hits for v in range(min(hits), max(hits)+1))
        start, stop = ((0, max(hits)) if side in (0, 3)
                       else (min(hits), (h if axis == 0 else w)-1))
        paths.append(frozenset((v, coordinate) if axis == 0 else (coordinate, v)
                               for v in range(start, stop+1)))
    info = dict(grid=a, bg=bg, target=target, marker=marker, colors=colors,
                source=source, fixed=fixed, groups=groups, active=active,
                targets=targets, slots=slots, rectangles=rectangles, paths=paths,
                axes=axes, sides=sides, internal_gaps=internal_gaps)
    base = np.full(a.shape, bg, dtype=np.int8)
    for p in all_targets | fixed:
        base[p] = a[p]
    for path in paths:
        for p in path:
            assert int(base[p]) in (bg, marker, *colors), "guide hits a socket"
            if int(base[p]) == bg:
                base[p] = marker
    info['base'] = base
    return info


def _fits(info, regions, corner):
    """All whole-key fits to this socket; deduplicate equivalent colored moves."""
    slot, rect, base = info['slots'][corner], info['rectangles'][corner], info['base']
    sr, _, sc, _ = _bbox(slot)
    h, w = base.shape
    results, seen = [], set()
    for op in range(8):
        parts = _transform(regions, op)
        for plug in (0, 1):
            pr, _, pc, _ = _bbox(parts[plug])
            placed = tuple(_shift(part, sr-pr, sc-pc) for part in parts)
            if placed[plug] != slot or placed[1-plug] & rect:
                continue
            full = placed[0] | placed[1]
            if not all(0 <= r < h and 0 <= c < w and int(base[r, c]) == info['bg']
                       for r, c in full):
                continue
            colors = info['colors'][::-1] if op >= 4 else info['colors']
            signature = tuple(sorted((r, c, col) for part, col in zip(placed, colors) for r, c in part))
            if signature not in seen:
                seen.add(signature)
                results.append(dict(parts=placed, op=op, plug=plug, corner=corner, full=full))
    return results


def _paint(info, moves, policy=(False, True)):
    out = info['base'].copy()
    for move in moves:
        colors = info['colors'][::-1] if policy[int(move['op'] >= 4)] else info['colors']
        for part, color in zip(move['parts'], colors):
            for p in part:
                assert int(out[p]) == info['bg']
                out[p] = color
    return out


def _solve_details(input_grid):
    info = _analyze(np.asarray(input_grid))
    choices = []
    for corner in sorted(info['active']):
        fits = _fits(info, info['groups'][corner], corner)
        assert fits, "cut key cannot fit its assigned corner"
        choices.append(fits)
    solutions = {}

    def search(i, occupied, moves):
        if len(solutions) > 1:
            return
        if i == len(choices):
            out = _paint(info, moves)
            solutions.setdefault(_key(out), (out, tuple(moves)))
            return
        for move in choices[i]:
            if not move['full'] & occupied:
                search(i+1, occupied | move['full'], moves + [move])

    search(0, frozenset(), [])
    assert len(solutions) == 1, "require one final colored answer, not an arbitrary symmetry tie"
    out, moves = next(iter(solutions.values()))
    return out, info, moves


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Determine cuts, destinations, fits, parity and rays from this input only."""
    return _solve_details(input_grid)[0]


def _slot_shape(rng, large=False, asymmetric=False):
    for _ in range(80):
        h = int(rng.integers(4 if large else 3, 6 if large else 5))
        w = 5 if large else int(rng.integers(3, 5))
        y = int(rng.integers(1, h-1))
        slot = {(y, 0)}
        for c in range(1, w):
            ny = int(np.clip(y + int(rng.integers(-1, 2)), 1, h-2))
            slot.update((r, c) for r in range(min(y, ny), max(y, ny)+1))
            y = ny
        # Widen or branch the separator without predetermining its complement.
        if rng.random() < .55:
            candidates = sorted({q for p in slot for q in _neighbors(p)
                                 if 0 < q[0] < h-1 and 0 <= q[1] < w} - slot)
            rng.shuffle(candidates)
            slot.update(candidates[:int(rng.integers(0, min(4, len(candidates))+1))])
        rect = frozenset(itertools.product(range(h), range(w)))
        target = rect - slot
        if len(_components(target)) != 2 or _bbox(target) != (0, h-1, 0, w-1):
            continue
        if asymmetric and _orbit(slot) != 8:
            continue
        return frozenset(target), frozenset(slot)
    # Asymmetric, non-square four-by-four separator, two connected banks.
    width = 5 if large else 4
    slot = frozenset({(1, 0), (1, 1)} | {(2, c) for c in range(1, width)})
    rect = frozenset(itertools.product(range(4), range(width)))
    return rect-slot, slot


def _handle(rng, target, slot, large):
    r0, r1, c0, c1 = _bbox(target)
    assert (r0, c0) == (0, 0)
    # The untransformed socket is at TL; exterior growth goes inward from it.
    starts = sorted((r, c1+1) for r, c in slot if c == c1)
    handle = {starts[int(rng.integers(len(starts)))]}
    maximum = 7 if large else 6
    requested = int(rng.integers(2, 11 if large else 8))
    style = int(rng.integers(3))
    tip = next(iter(handle))
    for _ in range(requested-1):
        frontier = sorted({q for p in handle for q in _neighbors(p)
                           if 0 <= q[0] < maximum and 0 <= q[1] < maximum
                           and not (q[0] <= r1 and q[1] <= c1)} - handle)
        if not frontier:
            break
        weights = []
        for p in frontier:
            degree = sum(q in handle for q in _neighbors(p))
            distance = abs(p[0]-tip[0]) + abs(p[1]-tip[1])
            weights.append((degree**2 if style == 0 else
                            (8 if distance == 1 else .2) / degree**2 if style == 1 else 1/degree))
        weights = np.asarray(weights, dtype=float)
        tip = frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))]
        handle.add(tip)
    return frozenset(handle)


def _corner_assembly(target, slot, handle, corner_index, rng):
    op = CORNER_OPS[corner_index][int(rng.integers(2))]
    parts = _transform((target, slot, handle), op)
    # Keep coordinates relative to the target rectangle, not the complete key.
    r0, _, c0, _ = _bbox(parts[0])
    return tuple(_shift(part, -r0, -c0) for part in parts)


def _make_pair(rng, double, large, roles, shared, parity_offset, require_gap=False):
    """Bounded proposals; expected answers are painted from constructed geometry."""
    bg, tc, marker, pc, hc = roles
    for attempt in range(180):
        common = _slot_shape(rng, large, asymmetric=shared or attempt >= 100)
        systems = {}
        source_local = {}
        # Canonical markers are top and right. Global rotations are applied later.
        owners = CORNERS if double else ((0, 0), (0, 1))
        for j, corner in enumerate(CORNERS):
            target, slot = common if shared else _slot_shape(rng, large, asymmetric=attempt >= 100)
            handle = _handle(rng, target, slot, large)
            systems[corner] = _corner_assembly(target, slot, handle, j, rng)
            if corner in owners:
                parity = ((j + parity_offset) % 2 if parity_offset is not None
                          else int(rng.integers(2)))
                op = int(rng.integers(4)) + 4*parity
                source_local[corner] = (_transform(systems[corner][1:], op), parity)
        # Room for the source and sockets is selected from actual extents.
        source_h = max(_bbox(p[0][0] | p[0][1])[1]+1 for p in source_local.values())
        source_w = max(_bbox(p[0][0] | p[0][1])[3]+1 for p in source_local.values())
        extents = [(_bbox(t | s | k)[1]-min(0, _bbox(t | s | k)[0])+1,
                    _bbox(t | s | k)[3]-min(0, _bbox(t | s | k)[2])+1)
                   for t, s, k in systems.values()]
        outer_h, outer_w = max(x[0] for x in extents), max(x[1] for x in extents)
        low_h = max(17, 2*outer_h+3, (2*source_h if double else source_h)+5)
        low_w = max(19, 2*outer_w+3, 2*source_w+5)
        if max(low_h, low_w) > 30:
            continue
        h = int(rng.integers(low_h, 31))
        w = int(rng.integers(low_w, 31))
        ar = h//2 + int(rng.integers(-1, 2)) if double else None
        ac = w//2 + int(rng.integers(-1, 2))
        inp = np.full((h, w), bg, dtype=np.int8)
        destinations = {}
        target_cells = set()
        for corner, parts in systems.items():
            r0, r1, c0, c1 = _bbox(parts[0])
            dr = -r0 if corner[0] == 0 else h-1-r1
            dc = -c0 if corner[1] == 0 else w-1-c1
            placed = tuple(_shift(part, dr, dc) for part in parts)
            if not all(0 <= r < h and 0 <= c < w for part in placed for r, c in part):
                break
            destinations[corner] = placed
            target_cells.update(placed[0])
            for p in placed[0]:
                inp[p] = tc
        else:
            source_cells, source_parts = set(), {}
            for corner, (parts, parity) in source_local.items():
                ph = _bbox(parts[0] | parts[1])[1]+1
                pw = _bbox(parts[0] | parts[1])[3]+1
                rr = (ar-ph if corner[0] == 0 else ar+1) if double else int(rng.integers(2, h-ph-1))
                cc = ac-pw if corner[1] == 0 else ac+1
                absolute = tuple(_shift(part, rr, cc) for part in parts)
                source_parts[corner] = absolute
                source_cells.update(absolute[0] | absolute[1])
                for part, color in zip(absolute, (pc, hc)):
                    for p in part:
                        if 0 < p[0] < h-1 and 0 < p[1] < w-1:
                            inp[p] = color
            if not all(0 < r < h-1 and 0 < c < w-1 for r, c in source_cells):
                continue
            minr, maxr, minc, maxc = _bbox(source_cells)
            spine = {(r, ac) for r in range(minr, maxr+1)}
            if double:
                spine.update((ar, c) for c in range(minc, maxc+1))
            # Remove cells only if an off-axis detour preserves source connectivity.
            removal = sorted(spine)
            rng.shuffle(removal)
            removed = 0
            for p in removal:
                if removed >= (int(rng.integers(1, 4)) if require_gap else 2):
                    break
                if p[0] in (minr, maxr) or p[1] in (minc, maxc) or p == (ar, ac):
                    continue
                if rng.random() < (.95 if require_gap else .3):
                    trial = (spine - {p}) | source_cells
                    if len(_components(trial)) == 1:
                        spine.remove(p)
                        removed += 1
            all_source = source_cells | spine
            # Separate source from fixed sockets and from every potential docking.
            dock_cells = set().union(*(set(s) | set(k) for t, s, k in destinations.values()))
            moat = target_cells | {q for p in target_cells for q in _neighbors(p)}
            if all_source & (moat | dock_cells):
                continue
            for p in spine:
                inp[p] = pc if rng.random() < .75 else hc
            inp[0, ac] = marker
            if double:
                inp[ar, -1] = marker
            try:
                info = _analyze(inp)
                if require_gap and not info['internal_gaps']:
                    continue
                expected = info['base'].copy()
                for corner in info['active']:
                    parts = destinations[corner][1:]
                    cols = (hc, pc) if source_local[corner][1] else (pc, hc)
                    for part, color in zip(parts, cols):
                        for p in part:
                            assert int(expected[p]) == bg
                            expected[p] = color
                actual, _, _ = _solve_details(inp)
                if not np.array_equal(actual, expected):
                    continue
                return {'input': inp, 'output': expected}
            except AssertionError:
                continue
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    roles = tuple(int(v) for v in rng.choice(10, 5, replace=False))
    puzzle, seen = {'train': [], 'test': []}, set()
    train_turns = [int(rng.integers(4)) for _ in range(num_train)]
    unseen = [v for v in range(4) if v not in train_turns]
    test_turns = [int(rng.choice(unseen or list(range(4)))) for _ in range(num_test)]
    for split, number, turns in (('train', num_train, train_turns), ('test', num_test, test_turns)):
        for i in range(number):
            # Both arities and both parity classes are established at 2/1.
            double = (i == 1 or (i > 1 and rng.random() < .5)) if split == 'train' else True
            shared = (split == 'train' and i == 0) or bool(rng.random() < .55)
            large = split == 'test'
            if not stable:
                roles = tuple(int(v) for v in rng.choice(10, 5, replace=False))
            for _ in range(8):
                witness = split == 'train' and i == 0
                pair = _make_pair(rng, double, large, roles, shared,
                                  int(rng.integers(2)) if witness else None,
                                  require_gap=split == 'train' and i == 1)
                pair = {k: np.rot90(v, turns[i]).copy() for k, v in pair.items()}
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _routing_witness(info, moves):
    """Does swapping owners admit a different colored answer with the same cuts?"""
    if len(moves) != 2:
        return False
    corners = sorted(info['active'])
    choices = [_fits(info, info['groups'][corners[1-i]], corners[i]) for i in range(2)]
    correct = _paint(info, moves)
    for first in choices[0]:
        for second in choices[1]:
            if not first['full'] & second['full']:
                if not np.array_equal(_paint(info, (first, second)), correct):
                    return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(0, len(puzzle['train']), len(puzzle['test']))
    seen, arities, parities = set(), set(), set()
    routing, deletion, gap = False, False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            solved, info, moves = _solve_details(inp)
            assert np.array_equal(solved, out)
            assert all(int(out[p]) == int(inp[p]) for part in info['targets'].values() for p in part)
            assert all(int(out[p]) == int(inp[p]) for p in info['fixed'])
            moving = info['source'] - info['fixed']
            assert all(int(out[p]) == info['bg'] for p in moving)
            expected_area = len(info['fixed']) + sum(sum(map(len, info['groups'][c])) for c in info['active'])
            assert int(np.count_nonzero(np.isin(out, info['colors']))) == expected_area
            assert np.count_nonzero(out == info['target']) == np.count_nonzero(inp == info['target'])
            for corner in set(info['groups']) - info['active']:
                assert all(int(out[p]) == info['bg'] for p in info['slots'][corner])
                # A genuine deletion witness: even this unused key could fit.
                fits = _fits(info, info['groups'][corner], corner)
                assert any(not m['full'] & frozenset().union(*(x['full'] for x in moves)) for m in fits)
                if split == 'train':
                    deletion = True
            if split == 'train':
                arities.add(len(info['sides']))
                parities.update(int(m['op'] >= 4) for m in moves)
                routing |= _routing_witness(info, moves)
                gap |= info['internal_gaps'] > 0
    assert arities == {1, 2} and parities == {0, 1}
    assert deletion, "missing selection/deletion demonstration"
    assert routing, "demonstrations must distinguish ownership from shape-only matching"
    assert gap, "missing internal-gap / farthest-hit guide demonstration"


def self_test():
    """Stream 42 seeds, check semantic witnesses and measured structural variety."""
    dimensions, slots, handles, sources, directions = set(), set(), set(), set(), set()
    backgrounds, operations, relations, palettes = set(), set(), set(), set()
    stats = dict(seeds=42, pairs=0, gapped_guides=0, symmetric_slots=0,
                 reflection_witnesses=0, rotation_witnesses=0, deletion_witnesses=0,
                 routing_witnesses=0, branched_handles=0)
    digest = hashlib.sha256()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                solved, info, moves = _solve_details(inp.copy())
                assert np.array_equal(solved, out)
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                stats['pairs'] += 1
                dimensions.add(inp.shape)
                backgrounds.add(info['bg'])
                palettes.add((info['bg'], info['target'], info['marker'], info['colors']))
                sources.add(_canonical(info['source']))
                directions.add(tuple(sorted(info['sides'])))
                stats['gapped_guides'] += bool(info['internal_gaps'])
                stats['deletion_witnesses'] += len(info['sides']) == 2
                stats['routing_witnesses'] += _routing_witness(info, moves)
                # These alternatives keep all fitted coordinates, isolating color parity.
                stats['reflection_witnesses'] += not np.array_equal(_paint(info, moves, (False, False)), out)
                stats['rotation_witnesses'] += not np.array_equal(_paint(info, moves, (True, True)), out)
                for move in moves:
                    plug = move['parts'][move['plug']]
                    handle = move['parts'][1-move['plug']]
                    slots.add(_canonical(plug))
                    handles.add(_canonical(handle))
                    operations.add(move['op'])
                    relations.add(int(np.sign(len(plug)-len(handle))))
                    stats['symmetric_slots'] += _orbit(plug) < 8
                    stats['branched_handles'] += any(sum(q in handle for q in _neighbors(p)) >= 3 for p in handle)
    assert stats['pairs'] == 210
    assert len(dimensions) >= 30 and len(slots) >= 12 and len(handles) >= 35
    assert len(sources) >= 160 and len(palettes) >= 30
    assert backgrounds == set(range(10)) and operations == set(range(8))
    assert len(directions) == 8 and relations == {-1, 0, 1}
    assert stats['gapped_guides'] > 0 and stats['symmetric_slots'] > 0
    assert stats['branched_handles'] > 10
    assert stats['routing_witnesses'] >= 42
    assert stats['reflection_witnesses'] >= 42 and stats['rotation_witnesses'] >= 42
    first, second = generate_puzzle(123, 5, 3), generate_puzzle(123, 5, 3)
    assert all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    # Conjugating a fit by a global rigid symmetry preserves its parity.
    for pair in first['train'][:2] + first['test'][:1]:
        for reflection in (False, True):
            for turns in range(4):
                inp = np.fliplr(pair['input']) if reflection else pair['input']
                out = np.fliplr(pair['output']) if reflection else pair['output']
                transformed = np.rot90(inp, turns).copy()
                before = transformed.copy()
                assert np.array_equal(solve(transformed), np.rot90(out, turns))
                assert np.array_equal(transformed, before), "solve must not mutate its input"
    # Empty axis crossings obey the same rule; require an actually connected witness.
    empty_crossing = False
    for pair in first['train'] + first['test']:
        _, info, _ = _solve_details(pair['input'])
        ar, ac = info['axes']
        if ar is not None and ac is not None:
            candidate = pair['input'].copy()
            candidate[ar, ac] = info['bg']
            try:
                _, changed, _ = _solve_details(candidate)
            except AssertionError:
                continue
            assert (ar, ac) not in changed['fixed']
            assert int(solve(candidate)[ar, ac]) == info['marker']
            empty_crossing = True
            break
    assert empty_crossing, "empty-crossing extension lacks a checked witness"
    # A straight slot and bent attachment permit different reflected colored answers.
    ambiguous = np.zeros((25, 25), dtype=np.int8)
    for r in (0, 22):
        for c in (0, 22):
            ambiguous[r, c:c+3] = 8
            ambiguous[r+2, c:c+3] = 8
    ambiguous[0, 12] = 9
    ambiguous[8:15, 12] = 2
    ambiguous[8, 8:11] = 2
    ambiguous[8:10, 11] = 3
    ambiguous[13, 13:16] = 2
    ambiguous[13:15, 16] = 3
    try:
        solve(ambiguous)
    except AssertionError as error:
        assert 'one final colored answer' in str(error)
    else:
        raise AssertionError("ambiguous reflection was silently tie-broken")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(700+n*10+t, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1),
                 (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    stats.update(dimensions=len(dimensions), canonical_slots=len(slots), canonical_handles=len(handles),
                 canonical_sources=len(sources), palettes=len(palettes), directions=len(directions),
                 operations=sorted(operations), plug_handle_area_relations=sorted(relations),
                 symmetry_equivariance_cases=24, ambiguous_fit_rejected=True,
                 empty_crossing_checked=empty_crossing,
                 streamed_sha256=digest.hexdigest())
    return stats
