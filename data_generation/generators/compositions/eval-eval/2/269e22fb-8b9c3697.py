"""Witness-card docking: exact crop recognition and geometric pose selection.

Black (0) is background, gray (9) forms the reference frame and receivers,
and azure (8) records the completed card's swept footprint. Patterns use 1..7.
The solver obtains its reference from the current input, never from a seed.
"""
from __future__ import annotations

from collections import deque
import numpy as np


BACKGROUND, STRUCTURE, TRAIL = 0, 9, 8
_DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))
HOST_SEEDS = (
    0, 459224226, 263621361, 371492979, 1640759, 339465362,
    185994304, 844139708, 802249266, 984538885, 753845132,
    216253355, 898647249, 589240504, 715149326, 797764069,
    927223064, 493721679, 765297776, 583578524, 657958653,
    134129526, 909529950, 728345457, 560373927, 734016059,
    512270315, 266701336, 513056810, 690079381, 862263650,
    978461237, 775869287, 345205064, 489263069, 48677735,
    715169998, 199920706, 460359847, 84660524, 597527969, 49505258,
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _transform(a, index):
    """Four rotations, followed by four rotations of a mirror image."""
    return np.rot90(np.fliplr(a) if index >= 4 else a, index % 4).copy()


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _canonical(a):
    return min(_key(_transform(a, i) ^ s) for i in range(8) for s in (0, 1))


def _binary(a):
    colors = tuple(map(int, np.unique(a)))
    _require(len(colors) == 2 and all(1 <= c <= 7 for c in colors),
             "a card must contain exactly two colors from 1..7")
    return (a == colors[1]).astype(np.int8), colors


def _colorize(a, palette):
    return np.where(a == 0, palette[0], palette[1]).astype(np.int8)


def _matches(reference, patch):
    """All poses, role exchanges, and exact submatrix offsets; no state."""
    result = []
    for i in range(8):
        a = _transform(reference, i)
        if any(x > y for x, y in zip(patch.shape, a.shape)):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(a, patch.shape)
        for swap in (0, 1):
            rr, cc = np.nonzero(np.all(windows == (patch ^ swap), axis=(-1, -2)))
            result.extend((i, swap, int(r), int(c)) for r, c in zip(rr, cc))
    return result


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        pending, component = deque([start]), {start}
        while pending:
            r, c = pending.popleft()
            for dr, dc in _DIRECTIONS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    pending.append(p)
        result.append(component)
    return result


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}


def _frame(cells):
    r, c, r1, c1 = _bounds(cells)
    if r1 - r < 4 or c1 - c < 4:
        return None
    border = _rect(r, c, r1 - r, c1 - c) - _rect(r + 1, c + 1, r1 - r - 2, c1 - c - 2)
    return (r, c, r1, c1) if cells == border else None


def _fork(cells):
    parses = []
    for dr, dc in _DIRECTIONS:
        out, perp = (dr, dc), (-dc, dr)
        kmin = min(r * dr + c * dc for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * dr + c * dc - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)} | {
            (k, q) for q, n in enumerate(lengths) for k in range(1, n + 1)}
        if local != expected:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or any(runs[j][1] >= runs[j + 1][0] for j in (0, 1)):
            continue
        left, middle, right = runs
        if left[0] != 0 or right[1] != width or middle[0] + middle[1] != width:
            continue
        if left[1] - left[0] != right[1] - right[0]:
            continue
        parses.append(dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                           q0=middle[0], span=middle[1] - middle[0],
                           stop=middle[2] + 1, lengths=tuple(x[2] for x in runs),
                           width=width, cells=cells))
    _require(len(parses) == 1, "a receiver must have one base and three positive prongs")
    return parses[0]


def _cell(fork, k, q):
    k, q = k + fork['kmin'], q + fork['qmin']
    return (k * fork['out'][0] + q * fork['perp'][0],
            k * fork['out'][1] + q * fork['perp'][1])


def _local_card(grid, fork, token):
    points = {(r * fork['out'][0] + c * fork['out'][1] - fork['kmin'],
               r * fork['perp'][0] + c * fork['perp'][1] - fork['qmin'])
              for r, c in token['cells']}
    k, q, k1, q1 = _bounds(points)
    patch = np.array([[grid[_cell(fork, kk, qq)] for qq in range(q, q1)]
                      for kk in range(k, k1)], dtype=np.int8)
    return k, q, patch


def _proposal(grid, reference, fork, token, pose, k, q, palette):
    i, swap, r, c = pose
    full = _transform(reference, i) ^ swap
    h, w = full.shape
    if w != fork['span'] or q - c != fork['q0']:
        return None
    original_k = k - r
    if original_k <= fork['stop']:
        return None
    route = {_cell(fork, kk, qq)
             for kk in range(fork['stop'], original_k + h)
             for qq in range(fork['q0'], fork['q0'] + w)}
    for rr, cc in route:
        if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
            return None
        if (rr, cc) not in token['cells'] and grid[rr, cc] != BACKGROUND:
            return None
    final = {_cell(fork, fork['stop'] + rr, fork['q0'] + cc)
             for rr in range(h) for cc in range(w)}
    source = {_cell(fork, original_k + rr, fork['q0'] + cc)
              for rr in range(h) for cc in range(w)}
    return dict(full=_colorize(full, palette), offset=(r, c),
                pose=(i, swap), original_k=original_k, route=route,
                final=final, source=source, distance=original_k - fork['stop'])


def _analyze(input_grid):
    grid = _grid(input_grid)
    _require(not np.any(grid == TRAIL), "trail color is absent from inputs")
    structural = _components(grid == STRUCTURE)
    frames = [(cells, _frame(cells)) for cells in structural if _frame(cells) is not None]
    _require(len(frames) == 1, "one closed reference frame is required")
    frame_cells, (r, c, r1, c1) = frames[0]
    master = grid[r + 1:r1 - 1, c + 1:c1 - 1]
    reference, _ = _binary(master)
    _require(len({_key(_transform(reference, i) ^ s)
                  for i in range(8) for s in (0, 1)}) == 16,
             "reference must distinguish all geometric and color poses")
    forks = [_fork(cells) for cells in structural if cells != frame_cells]
    _require(bool(forks), "at least one receiver is required")
    token_mask = (grid != BACKGROUND) & (grid != STRUCTURE)
    token_mask[r:r1, c:c1] = False
    tokens = []
    for cells in _components(token_mask):
        tr, tc, tr1, tc1 = _bounds(cells)
        _require(cells == _rect(tr, tc, tr1 - tr, tc1 - tc), "fragments must be solid rectangles")
        _require(min(tr1 - tr, tc1 - tc) >= 2, "fragments have both sides at least two")
        _binary(grid[tr:tr1, tc:tc1])
        tokens.append(dict(cells=cells, bounds=(tr, tc, tr1, tc1)))
    _require(bool(tokens), "at least one fragment is required")
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        proposals = []
        for ti, token in enumerate(tokens):
            k, q, patch = _local_card(grid, fork, token)
            binary_patch, palette = _binary(patch)
            poses = _matches(reference, binary_patch)
            for pose in poses:
                proposal = _proposal(grid, reference, fork, token, pose, k, q, palette)
                if proposal is not None:
                    proposals.append(dict(proposal, fi=fi, ti=ti, all_pose_count=len(poses)))
        _require(len(proposals) <= 1, "receiver has competing complete-card poses")
        if proposals:
            match = proposals[0]
            _require(match['ti'] not in used, "one fragment cannot activate two receivers")
            _require(swept.isdisjoint(match['route']), "successful routes must be disjoint")
            used.add(match['ti'])
            swept.update(match['route'])
            matches.append(match)
    return dict(grid=grid, reference=reference, frame=(r, c, r1, c1),
                frame_cells=frame_cells, forks=forks, tokens=tokens,
                matches=matches, unmatched=set(range(len(tokens))) - used)


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _render(analysis, mode='full'):
    output = analysis['grid'].copy()
    for token in analysis['tokens']:
        _paint(output, token['cells'], BACKGROUND)
    if mode == 'omit_A':
        # Literal fragments are narrower than the stop, so B alone rejects them.
        return output
    for match in analysis['matches']:
        fork = analysis['forks'][match['fi']]
        if mode == 'full':
            _paint(output, match['route'] - match['final'], TRAIL)
            start = fork['stop']
        elif mode == 'omit_B':
            # Restore the selected reference at its original, unmoved extent.
            start = match['original_k']
        else:
            raise ValueError('unknown rendering mode')
        for (rr, cc), value in np.ndenumerate(match['full']):
            output[_cell(fork, start + rr, fork['q0'] + cc)] = value
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the framed reference, jointly infer card poses, and dock them."""
    return _render(_analyze(input_grid))


def _crop_bank(reference):
    """Crops whose perpendicular offset and full span single out one pose.

    This is generation-time sampling evidence only. solve exhaustively searches
    the actual input reference and also checks boundaries and route occupancy.
    """
    h0, w0 = reference.shape
    bank = []
    for h in range(2, h0 + 1):
        for w in range(2, w0):
            for r in range(h0 - h + 1):
                for c in range(w0 - w + 1):
                    patch = reference[r:r + h, c:c + w]
                    if len(np.unique(patch)) != 2:
                        continue
                    poses = _matches(reference, patch)
                    compatible = [p for p in poses
                                  if _transform(reference, p[0]).shape[1] == w0 and p[3] == c]
                    if len(compatible) == 1:
                        bank.append((r, c, h, w, len(poses)))
    return bank


def _draw_reference(rng, h, w):
    a = np.zeros((h, w), dtype=np.int8)
    # Rectangles, bends, loops, and staircases combine and sometimes overlap.
    kinds = list(rng.permutation(4)) + list(rng.integers(4, size=int(rng.integers(1, 5))))
    for kind in kinds:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        hh, ww = int(rng.integers(1, h - r + 1)), int(rng.integers(1, w - c + 1))
        if kind == 0:
            a[r:r + hh, c:c + ww] = 1
        elif kind == 1:
            a[r:r + hh, c] = 1
            a[r, c:c + ww] = 1
        elif kind == 2:
            a[r:r + hh, c] = a[r:r + hh, c + ww - 1] = 1
            a[r, c:c + ww] = a[r + hh - 1, c:c + ww] = 1
        else:
            for step in range(max(hh, ww)):
                a[r + min(step, hh - 1), c + min(step, ww - 1)] = 1
    return a


def _reference(rng):
    h, w = int(rng.integers(5, 8)), int(rng.integers(4, 8))
    for _ in range(100):
        a = _draw_reference(rng, h, w)
        if not .2 <= a.mean() <= .8 or len({_key(_transform(a, i) ^ s)
                                         for i in range(8) for s in (0, 1)}) != 16:
            continue
        bank = _crop_bank(a)
        anchors = [p for p in bank if p[0] > 0 and p[4] > 1 and
                   2 * p[1] + p[3] != w and .15 <= p[2] * p[3] / a.size <= .65]
        edges = [p for p in bank if p[0] == 0]
        small = [p for p in bank if p[2] * p[3] / a.size <= .35]
        if anchors and edges and small:
            turned = _transform(a, 1)
            turned_bank = _crop_bank(turned)
            turned_edges = [p for p in turned_bank if p[0] == 0]
            turned_small = [p for p in turned_bank if p[2] * p[3] / a.size <= .35]
            if turned_edges and turned_small:
                return a, ((a, bank, anchors, edges, small),
                           (turned, turned_bank, [], turned_edges, turned_small))
    # A constructive, dimension-independent fallback is still a new pattern at
    # the sampled size. It is procedural, not any parent's source bitmap.
    a = np.zeros((h, w), dtype=np.int8)
    a[0, :w - 1] = 1
    a[:h - 1, 0] = 1
    a[2:, w - 2] = 1
    a[h - 2, 1:3] = 1
    a[1, 2] = 1
    bank = _crop_bank(a)
    anchors = [p for p in bank if p[0] > 0 and p[4] > 1 and
               2 * p[1] + p[3] != w and .15 <= p[2] * p[3] / a.size <= .65]
    edges = [p for p in bank if p[0] == 0]
    small = [p for p in bank if p[2] * p[3] / a.size <= .35]
    _require(bool(anchors and edges and small), 'fallback reference lacks distinguishing crops')
    turned = _transform(a, 1)
    turned_bank = _crop_bank(turned)
    turned_edges = [p for p in turned_bank if p[0] == 0]
    turned_small = [p for p in turned_bank if p[2] * p[3] / a.size <= .35]
    _require(bool(turned_edges and turned_small), 'turned fallback lacks distinguishing crops')
    return a, ((a, bank, anchors, edges, small),
               (turned, turned_bank, [], turned_edges, turned_small))


def _palette(rng, excluded=()):
    colors = [c for c in range(1, 8) if c not in excluded]
    return tuple(map(int, rng.choice(colors, size=2, replace=False)))


def _module(rng, reference, crop, active, pose, mode):
    h, w = reference.shape
    thickness, clearance = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    width = w + 2 * (thickness + clearance)
    q0 = thickness + clearance
    if mode == 'standard':
        center = int(rng.integers(1, 3))
        left = right = int(rng.integers(center + 1, 5))
    elif mode == 'long_center':
        center = int(rng.integers(3, 5))
        left, right = map(int, rng.integers(1, center, size=2))
    else:
        left, center, right = map(int, rng.integers(1, 5, size=3))
    stop = center + 1
    start = max(left, center, right) + 1 + int(rng.integers(1, 5))
    depth = start + h if active else max(left, center, right) + 1
    inp = np.zeros((depth, width), dtype=np.int8)
    inp[0, :] = STRUCTURE
    inp[1:left + 1, :thickness] = STRUCTURE
    inp[1:center + 1, q0:q0 + w] = STRUCTURE
    inp[1:right + 1, width - thickness:] = STRUCTURE
    out = inp.copy()
    if active:
        r, c, ch, cw, _ = crop
        full = _colorize(reference, _palette(rng))
        inp[start + r:start + r + ch, q0 + c:q0 + c + cw] = full[r:r + ch, c:c + cw]
        # Ground truth comes from construction coordinates, not from solve().
        out[stop + h:start + h, q0:q0 + w] = TRAIL
        out[stop:stop + h, q0:q0 + w] = full
    return _transform(inp, pose), _transform(out, pose)


def _available(blocked, h, w):
    H, W = blocked.shape
    if h > H or w > W:
        return []
    sums = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return list(zip(*np.nonzero(counts == 0)))


def _reserve(blocked, r, c, h, w):
    blocked[max(0, r - 1):min(blocked.shape[0], r + h + 1),
            max(0, c - 1):min(blocked.shape[1], c + w + 1)] = True


def _pair(rng, reference_data, split, index, anchor_pose, seen):
    reference, variants = reference_data
    bank = variants[0][1]
    is_test = split == 'test'
    count = 2 if is_test or index == 1 else int(rng.integers(1, 3))
    active_count = 2 if is_test else 1
    for attempt in range(60):
        variant_index = index if not is_test and index < 2 else int(rng.integers(2))
        blocks = []
        for j in range(count):
            vi = (variant_index + j) % 2
            card_reference, card_bank, anchors, edges, small = variants[vi]
            if not is_test and index == 0 and j == 0:
                pool = anchors
            elif not is_test and index == 1 and j == 0:
                pool = edges
            else:
                pool = small if is_test else card_bank
            selected = pool[int(rng.integers(len(pool)))]
            if not is_test and index < 2 and j == 0:
                pose = (anchor_pose + index) % 4
                mode = 'standard' if index == 0 else 'long_center'
            else:
                pose, mode = int(rng.integers(8)), 'independent'
            blocks.append(_module(rng, card_reference, selected, j < active_count, pose, mode))
        active_colors = {int(c) for a, _ in blocks for c in np.unique(a) if 1 <= c <= 7}
        master = _colorize(_transform(reference, int(rng.integers(8))), _palette(rng, active_colors))
        framed = np.pad(master, 1, constant_values=STRUCTURE)
        blocks.append((framed, framed.copy()))
        H, W = (int(rng.integers(26, 31)), int(rng.integers(26, 31))) if is_test else (
            int(rng.integers(23, 31)), int(rng.integers(23, 31)))
        if attempt >= 30:
            H = W = 30
        inp = np.zeros((H, W), dtype=np.int8)
        out, blocked = inp.copy(), np.zeros((H, W), dtype=bool)
        order = sorted(range(len(blocks)), key=lambda j: -blocks[j][0].size)
        failed = False
        for j in order:
            a, b = blocks[j]
            h, w = a.shape
            positions = _available(blocked, h, w)
            if not positions:
                failed = True
                break
            r, c = positions[int(rng.integers(len(positions)))]
            inp[r:r + h, c:c + w] = a
            out[r:r + h, c:c + w] = b
            _reserve(blocked, r, c, h, w)
        if failed:
            continue
        try:
            analysis = _analyze(inp)
        except ValueError:
            continue
        if len(analysis['matches']) != active_count or not np.array_equal(_render(analysis), out):
            continue
        # Distractors are genuine reference fragments whose virtual extents
        # cannot dock. No arbitrary noise or markers are used for uniqueness.
        extras = int(rng.integers(2, 5) if is_test else rng.integers(1, 3))
        for _ in range(extras):
            placed = False
            for trial in range(80):
                r0, c0, ch, cw, _ = bank[int(rng.integers(len(bank)))]
                patch = _colorize(reference[r0:r0 + ch, c0:c0 + cw], _palette(rng))
                patch = _transform(patch, int(rng.integers(8)))
                ch, cw = patch.shape
                positions = _available(blocked, ch, cw)
                if not positions:
                    continue
                r, c = positions[int(rng.integers(len(positions)))]
                candidate = inp.copy()
                candidate[r:r + ch, c:c + cw] = patch
                try:
                    a = _analyze(candidate)
                except ValueError:
                    continue
                if len(a['matches']) != active_count or not np.array_equal(_render(a), out):
                    continue
                inp = candidate
                _reserve(blocked, r, c, ch, cw)
                placed = True
                break
            if not placed:
                failed = True
                break
        if failed or _key(inp) in seen:
            continue
        seen.add(_key(inp))
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded layout construction exhausted')


def _count(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
             and low <= int(value) <= high, f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create one shared-reference puzzle, with explicit legal pair counts."""
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _require(nt > nq, 'num_train must be greater than num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    reference_data = _reference(rng)
    anchor_pose = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            puzzle[split].append(_pair(rng, reference_data, split, index, anchor_pose, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check observed geometry, exact reconstruction, and both ablations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'num_train', 2, 10)
    _count(nq, 'num_test', 1, 4)
    _require(nt > nq, 'illegal split counts')
    seen, canonical = set(), None
    directions, first_two = set(), []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output must preserve canvas dimensions')
            _require(_key(inp) not in seen, 'inputs must be distinct')
            seen.add(_key(inp))
            a = _analyze(inp)
            current = _canonical(a['reference'])
            if canonical is None:
                canonical = current
            _require(current == canonical, 'one reference content must be shared within the puzzle')
            _require(np.array_equal(_render(a), out), 'output disagrees with the input-only rule')
            _require(bool(a['unmatched']), 'demonstrate erasure of unmatched reference fragments')
            _require(len(a['matches']) == (1 if split == 'train' else 2), 'incorrect activity schedule')
            _require(not np.array_equal(_render(a, 'omit_A'), out), 'A must affect the answer')
            _require(not np.array_equal(_render(a, 'omit_B'), out), 'B must affect the answer')
            r, c, r1, c1 = a['frame']
            _require(np.array_equal(inp[r:r1, c:c1], out[r:r1, c:c1]), 'reference must be preserved')
            master_colors = set(map(int, np.unique(inp[r + 1:r1 - 1, c + 1:c1 - 1])))
            trail_cells = set()
            for fork in a['forks']:
                _require(min(fork['lengths']) >= 1, 'all prongs have positive lengths beyond the base')
                _require(all(out[p] == STRUCTURE for p in fork['cells']), 'structures must be preserved')
            for m in a['matches']:
                fork, token = a['forks'][m['fi']], a['tokens'][m['ti']]
                k, q, patch = _local_card(inp, fork, token)
                h, w = patch.shape
                _require(master_colors.isdisjoint(map(int, np.unique(patch))),
                         'demonstrate color-role transfer using a different fragment palette')
                _require(w < fork['span'], 'visible fragment must be narrower than the full-card span')
                rr, cc = m['offset']
                _require(np.array_equal(m['full'][rr:rr + h, cc:cc + w], patch), 'crop must be exact')
                _require(len(m['route'] - m['final']) == m['distance'] * fork['span'],
                         'trail must cover the entire virtual-card sweep')
                trail_cells.update(m['route'] - m['final'])
                if split == 'train':
                    directions.add(fork['out'])
            _require(set(map(tuple, np.argwhere(out == TRAIL))) == trail_cells, 'unexpected trail cells')
            for ti in a['unmatched']:
                _require(all(out[p] == BACKGROUND for p in a['tokens'][ti]['cells']), 'unmatched fragment survives')
            if split == 'train' and index < 2:
                first_two.append(a)
    m0, m1 = first_two[0]['matches'][0], first_two[1]['matches'][0]
    _require(m0['all_pose_count'] > 1 and m0['offset'][0] > 0,
             'first demonstration must require geometry to resolve a crop and recover a hidden leading edge')
    f0, f1 = first_two[0]['forks'][m0['fi']], first_two[1]['forks'][m1['fi']]
    _, q, first_patch = _local_card(first_two[0]['grid'], f0,
                                   first_two[0]['tokens'][m0['ti']])
    _require(2 * (q - f0['q0']) + first_patch.shape[1] != f0['span'],
             'first fragment must be off the full-card centerline')
    _require(m1['offset'][0] == 0, 'second demonstration contrasts a visible leading edge')
    _require(m0['full'].shape == m1['full'].shape[::-1],
             'first two examples must expose both reference axis choices')
    _require(f0['lengths'][1] < min(f0['lengths'][0], f0['lengths'][2]), 'show a short central stop')
    _require(f1['lengths'][1] > max(f1['lengths'][0], f1['lengths'][2]), 'show a long central stop')
    _require(len(first_two[1]['forks']) == 2 and len(first_two[1]['matches']) == 1,
             'second demonstration contains an inactive receiver')
    _require(f0['out'][0] * f1['out'][0] + f0['out'][1] * f1['out'][1] == 0,
             'first two demonstrated approach axes must be perpendicular')


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _analyze(pair['input'])
            records.append(dict(split=split, shape=list(a['grid'].shape),
                                reference_shape=list(a['reference'].shape),
                                forks=len(a['forks']), active=len(a['matches']),
                                distractors=len(a['unmatched']),
                                prongs=[list(f['lengths']) for f in a['forks']],
                                spans=[f['span'] for f in a['forks']],
                                directions=[list(f['out']) for f in a['forks']],
                                distances=[int(m['distance']) for m in a['matches']],
                                offsets=[[int(x) for x in m['offset']] for m in a['matches']],
                                crop_pose_counts=[m['all_pose_count'] for m in a['matches']],
                                areas=[len(a['tokens'][m['ti']]['cells']) / m['full'].size
                                       for m in a['matches']]))
    return records


def self_test():
    """Bounded corpus, count boundaries, corruption rejection, diversity."""
    corpus = [generate_puzzle(seed) for seed in HOST_SEEDS]
    _require(_signature(corpus[0]) == _signature(generate_puzzle(HOST_SEEDS[0])), 'nondeterminism')
    references, structures, shapes, offsets, distances, prongs, directions = set(), set(), set(), set(), set(), set(), set()
    for puzzle in corpus:
        references.add(_canonical(_analyze(puzzle['train'][0]['input'])['reference']))
        structures.add(_signature(puzzle))
        for record in measure_puzzle(puzzle):
            shapes.add(tuple(record['shape']))
            offsets.update(map(tuple, record['offsets']))
            distances.update(record['distances'])
            prongs.update(map(tuple, record['prongs']))
            directions.update(map(tuple, record['directions']))
    _require(len(references) >= 38 and len(structures) == 42, 'insufficient reference diversity')
    _require(len(shapes) >= 25 and len(offsets) >= 12 and len(prongs) >= 35,
             'insufficient geometric variation')
    _require(directions == set(_DIRECTIONS) and len(distances) >= 6, 'motion diversity missing')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'counts ignored')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in corpus[0][s]]
                 for s in ('train', 'test')}
    pair = corrupted['test'][0]
    p = min(_analyze(pair['input'])['matches'][0]['final'])
    pair['output'][p] = TRAIL
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted completion accepted')
    malformed = _rect(0, 0, 1, 8) | _rect(1, 0, 3, 1) | _rect(1, 7, 3, 1)
    try:
        _fork(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('zero-length central prong accepted')
    # Check every fallback size directly, rather than hoping rejection reaches it.
    for h in range(5, 8):
        for w in range(4, 8):
            a = np.zeros((h, w), dtype=np.int8)
            a[0, :w - 1] = a[:h - 1, 0] = a[2:, w - 2] = a[h - 2, 1:3] = a[1, 2] = 1
            bank = _crop_bank(a)
            _require(len({_key(_transform(a, i) ^ s) for i in range(8) for s in (0, 1)}) == 16,
                     'fallback must be asymmetric')
            _require(any(p[0] > 0 and p[4] > 1 and 2 * p[1] + p[3] != w
                         and .15 <= p[2] * p[3] / a.size <= .65 for p in bank),
                     'fallback does not support the mandatory anchor')
            turned_bank = _crop_bank(_transform(a, 1))
            _require(any(p[0] == 0 for p in bank) and any(p[0] == 0 for p in turned_bank)
                     and any(p[2] * p[3] / a.size <= .35 for p in turned_bank),
                     'fallback must also support the turned card and small crops')
    return {'seeds_checked': 42, 'distinct_reference_classes': len(references),
            'canvas_shapes': len(shapes), 'crop_offsets': len(offsets),
            'prong_geometries': len(prongs), 'motion_distances': sorted(distances)}
