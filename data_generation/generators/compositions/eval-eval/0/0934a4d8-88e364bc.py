"""Veiled queues: a symmetric static maze, asymmetric beads, and a local arrow.

No parent imports or filesystem access. Construction labels use lane packing;
the input-only solver uses front-to-back collision-aware motion instead.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _require(np.all((a >= 0) & (a <= 9)), 'invalid color')
    return a.astype(np.int8, copy=False)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    seen = ~mask.copy()
    result = []
    for rr, cc in np.argwhere(mask):
        p = int(rr), int(cc)
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _find_board(a, background):
    """Largest complete rectangular frame; the compact glyph is smaller."""
    frames = []
    h, w = a.shape
    for color in np.unique(a):
        if color == background:
            continue
        for r in range(h - 8):
            row = a[r] == color
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            ends = np.flatnonzero(row & ~np.r_[row[1:], False]) + 1
            for c, d in zip(starts, ends):
                if d - c < 9:
                    continue
                for s in range(r + 9, h + 1):
                    if (np.all(a[s - 1, c:d] == color)
                            and np.all(a[r:s, c] == color)
                            and np.all(a[r:s, d - 1] == color)):
                        frames.append(((s - r) * (d - c), int(color), (r, s, int(c), int(d))))
    _require(frames, 'no framed maze')
    area = max(x[0] for x in frames)
    best = [x for x in frames if x[0] == area]
    _require(len(best) == 1, 'ambiguous largest frame')
    return best[0][1:]


def _decode_key(key, wall):
    border = np.r_[key[0], key[-1], key[:, 0], key[:, -1]]
    _require(np.all(border == wall), 'key carrier differs from maze')
    others = set(map(int, np.unique(key))) - {wall}
    _require(len(others) == 2, 'key needs trail and tips')
    interpretations = []
    for bead in sorted(others):
        trail = next(iter(others - {bead}))
        tips = set(map(tuple, np.argwhere(key == bead)))
        tracks = set(map(tuple, np.argwhere(key == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                r, c = r - dr, c - dc
                while (r, c) in tracks:
                    stroke.append((r, c))
                    r, c = r - dr, c - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == tracks:
                interpretations.append((bead, trail, (dr, dc), tuple(lengths)))
    _require(len(interpretations) == 1, 'ambiguous trail-to-tip key')
    return interpretations[0]


def _orbit(shape, r, c):
    h, w = shape
    return sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c), (h - 1 - r, w - 1 - c)})


def _restore(static):
    """-1 is veiled terrain. Beads have already been replaced by known floor."""
    full = static.copy()
    witnesses = 0
    for r in range((full.shape[0] + 1) // 2):
        for c in range((full.shape[1] + 1) // 2):
            orbit = _orbit(full.shape, r, c)
            visible = {int(static[p]) for p in orbit if static[p] >= 0}
            _require(len(visible) == 1, 'erased orbit or conflicting mirror evidence')
            value = next(iter(visible))
            for p in orbit:
                full[p] = value
            witnesses += sum(static[p] >= 0 for p in orbit) >= 2
    _require(witnesses >= 10, 'insufficient visible mirror support')
    return full


def _read(input_grid):
    a = _grid(input_grid)
    background = int(a[0, 0])
    _require(np.all(np.r_[a[0], a[-1], a[:, 0], a[:, -1]] == background),
             'scene needs an empty outer margin')
    wall, board_box = _find_board(a, background)
    r, s, c, d = board_box
    outside = a != background
    outside[r:s, c:d] = False
    objects = _components(outside)
    _require(len(objects) == 1, 'exactly one external key is required')
    kr, ks, kc, kd = _bbox(objects[0])
    _require(len(objects[0]) == (ks - kr) * (kd - kc), 'key must be a filled rectangle')
    bead, trail, direction, lengths = _decode_key(a[kr:ks, kc:kd], wall)
    _require(len({background, wall, bead, trail}) == 4, 'overlapping local color roles')
    board = a[r:s, c:d].copy()
    unknown = set(map(int, np.unique(board))) - {background, wall, bead}
    _require(len(unknown) == 1, 'exactly one veil color is required')
    veil = next(iter(unknown))
    _require(veil != trail, 'veil and trail roles must differ')
    vr, vs, vc, vd = _bbox(list(map(tuple, np.argwhere(board == veil))))
    _require(0 < vr < vs < board.shape[0] and 0 < vc < vd < board.shape[1],
             'veil must be inside the maze frame')
    sub = board[vr:vs, vc:vd]
    _require(sub.size >= 2 and np.all((sub == veil) | (sub == bead)), 'nonrectangular veil')
    starts = sorted(map(tuple, np.argwhere(board == bead)))
    _require(starts, 'no live beads')
    static = np.where(board == wall, 1, 0).astype(np.int8)
    static[board == veil] = -1
    terrain = _restore(static)
    _require(all(terrain[p] == 0 for p in starts), 'bead lies on reconstructed wall')
    _require(np.all(terrain[0]) and np.all(terrain[-1])
             and np.all(terrain[:, 0]) and np.all(terrain[:, -1]), 'open maze frame')
    return dict(background=background, wall=wall, bead=bead, trail=trail, veil=veil,
                board=board, board_box=board_box, key_shape=(ks - kr, kd - kc),
                key_box=(kr, ks, kc, kd), lengths=lengths, direction=direction,
                box=(vr, vs, vc, vd), static=static, terrain=terrain, starts=starts)


def _step(terrain, p, direction, corner=True):
    r, c = p
    dr, dc = direction
    nr, nc = r + dr, c + dc
    if not (0 <= nr < terrain.shape[0] and 0 <= nc < terrain.shape[1]):
        return None
    if terrain[nr, nc]:
        return None
    if corner and dr and dc and (terrain[nr, c] or terrain[r, nc]):
        return None
    return nr, nc


def _settle(terrain, starts, direction, *, corner=True, collide=True):
    """Downstream first equals repeated simultaneous one-cell steps to rest.

    Walls block side corners; point beads block destinations, not side cells.
    Ties are on parallel lanes and cannot interact.
    """
    dr, dc = direction
    occupied, moves = set(), []
    order = sorted(starts, key=lambda p: (-(p[0] * dr + p[1] * dc), p))
    for start in order:
        p, distance = start, 0
        while True:
            q = _step(terrain, p, direction, corner)
            if q is None or (collide and q in occupied):
                break
            p, distance = q, distance + 1
        occupied.add(p)
        moves.append((start, p, distance))
    return occupied, moves


def _crop(terrain, endpoints, info):
    full = np.where(terrain, info['wall'], info['background']).astype(np.int8)
    for p in endpoints:
        full[p] = info['bead']
    r, s, c, d = info['box']
    return full[r:s, c:d].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    endpoints, _ = _settle(info['terrain'], info['starts'], info['direction'])
    return _crop(info['terrain'], endpoints, info)


def _lanes(terrain, direction):
    """Partition free cells into maximal directed, corner-safe line segments."""
    dr, dc = direction
    result = []
    reverse = -dr, -dc
    for rr, cc in np.argwhere(terrain == 0):
        p = int(rr), int(cc)
        if _step(terrain, p, reverse) is not None:
            continue
        lane = [p]
        while True:
            p = _step(terrain, p, direction)
            if p is None:
                break
            lane.append(p)
        result.append(lane)
    return result


def _terrain(rng, h, w, style, fallback=False):
    terrain = np.zeros((h, w), np.int8)
    terrain[0] = terrain[-1] = 1
    terrain[:, 0] = terrain[:, -1] = 1
    if fallback:
        # Interior posts and unequal bars, then orthogonal mirror closure below.
        terrain[3, 5] = 1
        terrain[6, 3:5] = 1
    else:
        for _ in range(int(rng.integers(2, 7))):
            r = int(rng.integers(2, (h + 1) // 2 + 1))
            c = int(rng.integers(2, (w + 1) // 2 + 1))
            if style == 0:
                dh, dw = 1, int(rng.integers(1, 4))
                if rng.integers(2):
                    dh, dw = dw, dh
            elif style == 1:
                dh, dw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            else:
                dh, dw = 1, int(rng.integers(2, 4))
                terrain[r:min(h - 1, r + 3), c] = 1
            terrain[r:min(h - 1, r + dh), c:min(w - 1, c + dw)] = 1
    terrain |= terrain[::-1]
    terrain |= terrain[:, ::-1]
    return terrain


def _make_key(rng, direction, wall, trail, bead):
    dr, dc = direction
    pr, pc = -dc, dr
    count = 2 if dr and dc else int(rng.integers(2, 4))
    length = int(rng.integers(2, 4))
    points = []
    for i in range(count):
        stagger = int(rng.integers(2))
        for j in range(length + 1):
            points.append((i * 2 * pr + (j + stagger) * dr,
                           i * 2 * pc + (j + stagger) * dc,
                           bead if j == length else trail))
    r, s, c, d = _bbox([(p[0], p[1]) for p in points])
    key = np.full((s - r + 2, d - c + 2), wall, np.int8)
    for rr, cc, color in points:
        key[rr - r + 1, cc - c + 1] = color
    _decode_key(key, wall)
    return key


def _scene(rng, board, key, background):
    h, w = board.shape
    kh, kw = key.shape
    choices = []
    for gap in (1, 2):
        if w + kw + gap + 2 <= 30 and max(h, kh) + 2 <= 30:
            choices.extend((('left', gap), ('right', gap)))
        if h + kh + gap + 2 <= 30 and max(w, kw) + 2 <= 30:
            choices.extend((('top', gap), ('bottom', gap)))
    if not choices:
        return None
    side, gap = choices[int(rng.integers(len(choices)))]
    if side in ('left', 'right'):
        hh, ww = max(h, kh) + 2, w + kw + gap + 2
        br, kr = int(rng.integers(1, hh - h)), int(rng.integers(1, hh - kh))
        bc, kc = (kw + gap + 1, 1) if side == 'left' else (1, w + gap + 1)
    else:
        hh, ww = h + kh + gap + 2, max(w, kw) + 2
        bc, kc = int(rng.integers(1, ww - w)), int(rng.integers(1, ww - kw))
        br, kr = (kh + gap + 1, 1) if side == 'top' else (1, h + gap + 1)
    a = np.full((hh, ww), background, np.int8)
    a[br:br + h, bc:bc + w] = board
    a[kr:kr + kh, kc:kc + kw] = key
    return a


def _blocking_cells(terrain, end, direction):
    r, c = end
    dr, dc = direction
    candidates = [(r + dr, c + dc)]
    if dr and dc:
        candidates += [(r + dr, c), (r, c + dc)]
    return [p for p in candidates if terrain[p]]


def _crosses(box, shape):
    r, s, c, d = box
    return (2 * r < shape[0] - 1 < 2 * (s - 1),
            2 * c < shape[1] - 1 < 2 * (d - 1))


def _choose_window(rng, terrain, lane, direction, seam, starts, endpoints, colors):
    h, w = terrain.shape
    blockers = _blocking_cells(terrain, lane[-1], direction)
    if not blockers or any(p[0] in (0, h - 1) or p[1] in (0, w - 1) for p in blockers):
        return None
    core = blockers + lane[-2:]
    if direction[0] and direction[1]:
        nr, nc = lane[-1][0] + direction[0], lane[-1][1] + direction[1]
        if terrain[nr, nc] == 0:
            core.append((nr, nc))
    r, s, c, d = _bbox(core)
    for _ in range(24):
        box = [max(1, r - int(rng.integers(3))), min(h - 1, s + int(rng.integers(3))),
               max(1, c - int(rng.integers(3))), min(w - 1, d + int(rng.integers(3)))]
        if seam:
            axis = int(rng.integers(2))
            low, high, n = (0, 1, h) if axis == 0 else (2, 3, w)
            box[low] = min(box[low], (n - 1) // 2 - 1)
            box[high] = max(box[high], n // 2 + 2)
        elif any(_crosses(box, terrain.shape)):
            continue
        vr, vs, vc, vd = box
        static = terrain.copy()
        static[vr:vs, vc:vd] = -1
        for p in starts:
            static[p] = 0
        try:
            restored = _restore(static)
        except AssertionError:
            continue
        if not np.array_equal(restored, terrain):
            continue
        info = dict(box=box, background=colors[0], wall=colors[1], bead=colors[2])
        target = _crop(terrain, endpoints, info)
        # Strong ablations: reconstruction changes *bead destinations*, not only walls;
        # ignoring queues, reversing the arrow, and omitting motion change the crop.
        opened = terrain.copy()
        opened[vr:vs, vc:vd] = 0
        wrong_ends, _ = _settle(opened, starts, direction)
        bead_mask = target == colors[2]
        if np.array_equal(bead_mask, _crop(opened, wrong_ends, info) == colors[2]):
            continue
        if np.array_equal(target, _crop(terrain, set(starts), info)):
            continue
        no_queue, _ = _settle(terrain, starts, direction, collide=False)
        if np.array_equal(target, _crop(terrain, no_queue, info)):
            continue
        reversed_ends, _ = _settle(terrain, starts, tuple(-v for v in direction))
        if np.array_equal(target, _crop(terrain, reversed_ends, info)):
            continue
        return tuple(box), target
    return None


def _make_pair(rng, direction, seam, convoy_size, corner_witness, style, seen, *, fallback_only=False):
    colors = tuple(map(int, rng.choice(10, 5, replace=False)))
    background, wall, bead, trail, veil = colors
    key = _make_key(rng, direction, wall, trail, bead)
    for attempt in range(100):
        fallback = fallback_only or attempt >= 75
        h, w = (17, 19) if fallback else (int(rng.integers(11, 20)), int(rng.integers(11, 20)))
        terrain = _terrain(rng, h, w, style, fallback)
        lanes = _lanes(terrain, direction)
        suitable = []
        for i, lane in enumerate(lanes):
            if len(lane) < max(5, convoy_size + 2):
                continue
            r, c = lane[-1]
            dr, dc = direction
            corner = bool(dr and dc and terrain[r + dr, c + dc] == 0)
            if corner_witness and not corner:
                continue
            suitable.append(i)
        for raw_i in rng.permutation(suitable)[:18]:
            i = int(raw_i)
            primary = lanes[i]
            # At least the final two sites start empty. Tokens remain ordered.
            indices = sorted(map(int, rng.choice(len(primary) - 2, convoy_size, replace=False)))
            starts = [primary[j] for j in indices]
            endpoints = set(primary[-convoy_size:])
            chosen = [(i, convoy_size)]
            others = [j for j, line in enumerate(lanes) if j != i and len(line) >= 2]
            for j in rng.permutation(others)[:int(rng.integers(0, 4))]:
                j = int(j)
                line = lanes[j]
                count = int(rng.integers(1, min(2, len(line) - 1) + 1))
                starts += [line[k] for k in sorted(map(int, rng.choice(len(line) - 1, count, replace=False)))]
                endpoints.update(line[-count:])
                chosen.append((j, count))
            # The construction's label is lane-wise count conservation and packing.
            selected = _choose_window(rng, terrain, primary, direction, seam, starts, endpoints, colors)
            if selected is None:
                continue
            box, output = selected
            vr, vs, vc, vd = box
            board = np.where(terrain, wall, background).astype(np.int8)
            board[vr:vs, vc:vd] = veil
            for p in starts:
                board[p] = bead
            # Markers on the veil may not trim its observable rectangular bounds.
            cells = list(map(tuple, np.argwhere(board == veil)))
            if not cells or _bbox(cells) != box:
                continue
            a = _scene(rng, board, key, background)
            if a is None or _fingerprint(a) in seen:
                continue
            if corner_witness:
                wrong, _ = _settle(terrain, starts, direction, corner=False)
                info = dict(box=box, background=background, wall=wall, bead=bead)
                if np.array_equal(output, _crop(terrain, wrong, info)):
                    continue
            return {'input': a, 'output': output}
    raise RuntimeError('bounded maze/window construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    style = int(rng.integers(3))
    plans = [(CARDINAL[int(rng.integers(4))], False, 2, False),
             (DIAGONAL[int(rng.integers(4))], True, 2, True)]
    for _ in range(nt - 2):
        plans.append((DIRECTIONS[int(rng.integers(8))], bool(rng.integers(2)),
                      int(rng.integers(2, 4)), False))
    rng.shuffle(plans)
    plans += [(DIRECTIONS[int(rng.integers(8))], True, int(rng.integers(3, 5)), False)
              for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, (direction, seam, size, corner) in enumerate(plans):
        pair = _make_pair(rng, direction, seam, size, corner,
                          style if rng.random() < .6 else int(rng.integers(3)), seen)
        seen.add(_fingerprint(pair['input']))
        puzzle['train' if index < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(pair):
    info = _read(pair['input'])
    terrain, starts, direction = info['terrain'], info['starts'], info['direction']
    ends, moves = _settle(terrain, starts, direction)
    target = _crop(terrain, ends, info)
    _require(np.array_equal(target, _grid(pair['output'])), 'incorrect settled-window label')
    _require(len(ends) == len(starts), 'bead conservation failed')
    lanes = _lanes(terrain, direction)
    occupancy = [sum(p in starts for p in lane) for lane in lanes]
    _require(max(occupancy) >= 2, 'no interacting queue')
    _require(set().union(*(set(line[-n:]) for line, n in zip(lanes, occupancy) if n)) == ends,
             'lanes are not packed at their downstream ends')
    _require(any(n > 0 for _, _, n in moves), 'no bead moved')
    opened = terrain.copy()
    r, s, c, d = info['box']
    opened[r:s, c:d] = 0
    wrong_a, _ = _settle(opened, starts, direction)
    _require(not np.array_equal(target == info['bead'], _crop(opened, wrong_a, info) == info['bead']),
             'reconstruction does not affect bead destinations in the answer')
    no_move = _crop(terrain, set(starts), info)
    wrong_queue, _ = _settle(terrain, starts, direction, collide=False)
    reversed_ends, _ = _settle(terrain, starts, tuple(-v for v in direction))
    _require(not np.array_equal(target, no_move), 'motion is irrelevant to crop')
    _require(not np.array_equal(target, _crop(terrain, wrong_queue, info)), 'queue interaction is irrelevant')
    _require(not np.array_equal(target, _crop(terrain, reversed_ends, info)), 'arrow polarity is irrelevant')
    no_corner, _ = _settle(terrain, starts, direction, corner=False)
    return dict(info, endpoints=ends, moves=moves, occupancy=occupancy,
                corner_witness=not np.array_equal(target, _crop(terrain, no_corner, info)),
                seam=any(_crosses(info['box'], terrain.shape)),
                crop_count_change=int(np.count_nonzero(target == info['bead'])) !=
                                  int(np.count_nonzero(info['board'][r:s, c:d] == info['bead'])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate inputs')
            seen.add(fp)
            info = _analysis(pair)
            _require(b.shape == (info['box'][1] - info['box'][0], info['box'][3] - info['box'][2]),
                     'answer must be target-aligned veil crop')
            _require(info['veil'] not in b and info['trail'] not in b, 'output contains key or veil')
            analyses[split].append(info)
    train = analyses['train']
    _require(len({x['direction'] for x in train}) >= 2, 'training needs different arrows')
    _require(any(0 in x['direction'] for x in train), 'missing cardinal demonstration')
    _require(any(x['corner_witness'] for x in train), 'missing diagonal corner demonstration')
    _require(any(x['seam'] for x in train), 'missing seam-overlap demonstration')


def self_test():
    """42 streamed seeds; no files, caches, traces, or retained puzzle collection."""
    signatures, shapes, windows, geometries, palettes, directions, convoy_sizes = (set() for _ in range(7))
    counts = dict(pairs=0, corner=0, seam=0, changing_crop_count=0, overlay=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analysis(pair)
                a, b = pair['input'], pair['output']
                digest.update(repr(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                shapes.add(a.shape)
                windows.add(b.shape)
                geometries.add(hashlib.sha256(info['terrain'].tobytes()).digest())
                palettes.add(tuple(info[k] for k in ('background', 'wall', 'bead', 'trail', 'veil')))
                directions.add(info['direction'])
                convoy_sizes.add(max(info['occupancy']))
                counts['pairs'] += 1
                counts['corner'] += info['corner_witness']
                counts['seam'] += info['seam']
                counts['changing_crop_count'] += info['crop_count_change']
                counts['overlay'] += any(info['static'][p] == 0 and
                    info['box'][0] <= p[0] < info['box'][1] and
                    info['box'][2] <= p[1] < info['box'][3] for p in info['starts'])
                original = a.copy()
                _require(np.array_equal(solve(a), b) and np.array_equal(a, original), 'solve changed input')
                # Family is invariant under whole-scene rotations and pair-local palette bijections.
                perm = np.roll(np.arange(10, dtype=np.int8), seed % 10)
                for turn in (1, 2, 3):
                    transformed = perm[np.rot90(a, turn)]
                    _require(np.array_equal(solve(transformed), perm[np.rot90(b, turn)]), 'equivariance failed')
        signatures.add(digest.digest())
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for split in puzzle
                         for p, q in zip(puzzle[split], again[split]) for k in p), 'nondeterministic')
    _require(len(signatures) == 42, 'seed diversity')
    _require(len(shapes) >= 30 and len(windows) >= 20 and len(geometries) >= 150, 'geometry diversity')
    _require(len(palettes) >= 150 and directions == set(DIRECTIONS), 'palette/direction diversity')
    _require({2, 3, 4} <= convoy_sizes, 'queue-size range')
    _require(counts['corner'] >= 42 and counts['seam'] >= 84, 'demonstration coverage')
    _require(counts['changing_crop_count'] >= 20 and counts['overlay'] >= 20, 'dynamic window coverage')
    # Check every explicit legal count. Invalid types and counts raise ValueError.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71 + nt * 5 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'count mismatch')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Exercise the bounded retry tail itself, including its longest queues.
    fallback_cases = 0
    for direction in DIRECTIONS:
        for size in (2, 3, 4):
            for seam in (False, True):
                pair = _make_pair(np.random.default_rng(201 + fallback_cases), direction,
                                  seam, size, False, 0, set(), fallback_only=True)
                _analysis(pair)
                fallback_cases += 1
        if 0 not in direction:
            pair = _make_pair(np.random.default_rng(301 + fallback_cases), direction,
                              True, 2, True, 0, set(), fallback_only=True)
            _require(_analysis(pair)['corner_witness'], 'fallback lost corner witness')
            fallback_cases += 1
    # Reject a wrong label and unsupported static evidence using small, in-memory mutations.
    p = generate_puzzle(9, 2, 1)
    bad = p['test'][0]['output'].copy()
    p['test'][0]['output'][0, 0] = (int(bad[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    static = np.zeros((11, 13), np.int8)
    static[1, 2] = 1
    for invalid in (static, np.full((11, 13), -1, np.int8)):
        try:
            _restore(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('conflicting or missing orbit accepted')
    return dict(valid=True, seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                output_shapes=len(windows), terrain_structures=len(geometries),
                palettes=len(palettes), directions=len(directions), convoy_sizes=sorted(convoy_sizes),
                fallback_cases=fallback_cases, **counts)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
