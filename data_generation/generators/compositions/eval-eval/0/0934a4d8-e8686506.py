"""Finite tile assembly through an erased, bilaterally symmetric scaffold.

The exported solver reads only input pixels. Reflection applies to wall/socket
classes, not to the colors or boundaries of the assembled tiles.
"""
from collections import Counter
from functools import lru_cache
import json

import numpy as np


class _SearchLimit(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _normalize(cells):
    cells = [(int(r), int(c)) for r, c in cells]
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _components(cells):
    remaining = {(int(r), int(c)) for r, c in cells}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(comp)))
    return result


def _read(a):
    """Recover roles, crop and inventory without labels or construction data."""
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "The outer border must be uniform background")
    frames = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        rr, cc = np.where(a == color)
        r, s, c, d = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        b = a[r:s, c:d]
        if (min(b.shape) >= 5 and bg in b and len(np.unique(b)) >= 3
                and all(np.all(edge == color) for edge in (b[0], b[-1], b[:, 0], b[:, -1]))):
            frames.append((color, (r, s, c, d), b.copy()))
    _require(len(frames) == 1, "Need one complete rectangular frame")
    wall, box, board = frames[0]
    r, s, c, d = box
    _require(r > 0 and c > 0 and s < a.shape[0] and d < a.shape[1], "Frame requires background margin")
    other = set(map(int, np.unique(board))) - {bg, wall}
    _require(len(other) == 1, "Board must contain one eraser color")
    mask = other.pop()
    er = np.argwhere(board == mask)
    mr, mc = er.min(axis=0)
    ms, md = er.max(axis=0) + 1
    _require(len(er) >= 2 and len(er) == (ms - mr) * (md - mc), "Eraser must be a solid rectangle")
    _require(mr > 0 and mc > 0 and ms < board.shape[0] and md < board.shape[1], "Eraser must avoid the frame")
    outside = a != bg
    outside[r:s, c:d] = False
    comps = _components(map(tuple, np.argwhere(outside)))
    _require(comps, "Missing tile inventory")
    pieces = []
    for comp in comps:
        colors = {int(a[p]) for p in comp}
        _require(len(colors) == 1, "Each loose tile must be monochrome")
        color = colors.pop()
        _require(color not in (wall, mask, bg), "Tile colors must have separate roles")
        _require(all(not (r - 1 <= x <= s and c - 1 <= y <= d) for x, y in comp),
                 "Loose tiles must be separated from the board")
        pieces.append((color, _normalize(comp)))
    return board, bg, wall, mask, pieces, box


def _domains(board, bg, wall, mask, axis):
    h, w = board.shape
    n = h * w
    if axis is None:
        mate = tuple(range(n))
    else:
        mate = tuple((h - 1 - r) * w + c if axis == 0 else r * w + w - 1 - c
                     for r in range(h) for c in range(w))
    flat = board.ravel()
    allowed = required = 0
    visible_socket_pairs = 0
    orphan_hole = 0
    for i, value in enumerate(flat):
        j = mate[i]
        peer = flat[j]
        if axis is not None and value != mask and peer != mask:
            if (value == bg) != (peer == bg):
                return None
            visible_socket_pairs += value == bg and i < j
        if value == bg:
            allowed |= 1 << i
            required |= 1 << i
        elif value == mask:
            if axis is None or peer in (bg, mask):
                allowed |= 1 << i
            if axis is not None and peer == bg:
                required |= 1 << i
            if peer == mask and i <= j:
                orphan_hole += 1
    if axis is not None and visible_socket_pairs < 1:
        return None
    return allowed, required, mate, orphan_hole


def _assembly(board, bg, wall, mask, pieces, axis, node_limit=24000):
    """Search distinct colored answers with exact inventory and orbit closure.

    Unknown orbits are decided during placement. They are never restored by an
    independent preliminary symmetry solver. At most two answers are retained.
    """
    domains = _domains(board, bg, wall, mask, axis)
    if domains is None:
        return (), 0
    allowed, initial_required, mate, _ = domains
    h, w = board.shape
    area = sum(len(shape) for color, shape in pieces)
    if not initial_required.bit_count() <= area <= allowed.bit_count():
        return (), 0
    inventory = Counter(pieces)
    types = sorted(inventory)
    moves, by_cell = [], [[] for _ in range(board.size)]
    for t, (color, shape) in enumerate(types):
        sh = max(r for r, c in shape) + 1
        sw = max(c for r, c in shape) + 1
        tmoves = []
        for r in range(1, h - sh):
            for c in range(1, w - sw):
                ids = [(r + dr) * w + c + dc for dr, dc in shape]
                bits = sum(1 << i for i in ids)
                if bits & allowed != bits:
                    continue
                closure = bits
                for i in ids:
                    closure |= 1 << mate[i]
                if closure & allowed != closure:
                    continue
                paint = sum((color + 1) << (4 * i) for i in ids)
                move = t, bits, closure, paint
                tmoves.append(move)
                for i in ids:
                    by_cell[i].append(move)
        if not tmoves:
            return (), 0
        moves.append(tmoves)
    sizes = tuple(len(shape) for color, shape in types)
    nodes = 0

    @lru_cache(maxsize=None)
    def search(occupied, required, counts):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("Assembly proof exceeded node bound")
        pending = required & ~occupied
        left_area = sum(k * size for k, size in zip(counts, sizes))
        if pending.bit_count() > left_area or (allowed & ~occupied).bit_count() < left_area:
            return ()
        if not any(counts):
            return (0,) if not pending else ()
        best = None
        bits = pending
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            choices = [m for m in by_cell[i] if counts[m[0]] and not m[1] & occupied]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        if best is None:
            # All exposed obligations are met; remaining tiles choose new
            # hidden socket orbits. Choose one type, but enumerate all its moves.
            choices_by_type = [[m for m in moves[t] if not m[1] & occupied]
                               for t in range(len(types)) if counts[t]]
            if any(not x for x in choices_by_type):
                return ()
            best = min(choices_by_type, key=len)
        answers = set()
        for t, bits, closure, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(occupied | bits, required | closure, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search(0, initial_required, tuple(inventory[t] for t in types))
    answers = []
    for code in codes:
        out = np.full(board.shape, wall, dtype=np.int8)
        for i in range(board.size):
            color = (code >> (4 * i)) & 15
            if color:
                out.flat[i] = color - 1
        answers.append(out)
    return tuple(answers), nodes


def _infer(input_grid, node_limit=None):
    a = _grid(input_grid)
    board, bg, wall, mask, pieces, box = _read(a)
    answers, axes, nodes = {}, [], 0
    for axis in (0, 1):
        solutions, used = _assembly(board, bg, wall, mask, pieces, axis, node_limit=node_limit)
        nodes += used
        if solutions:
            axes.append(axis)
        for answer in solutions:
            answers[_key(answer)] = answer
    _require(len(answers) == 1, "Need exactly one colored completion across supported axes")
    return next(iter(answers.values())).copy(), (board, bg, wall, mask, pieces, box, axes, nodes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _infer(input_grid)[0]


def _shape(rng, maxh, maxw):
    style = int(rng.integers(5))
    if style == 0:
        h = int(rng.integers(1, min(3, maxh) + 1))
        w = int(rng.integers(1, min(4, maxw) + 1))
        return tuple((r, c) for r in range(h) for c in range(w))
    if style == 1 and min(maxh, maxw) >= 2:
        h, w = int(rng.integers(2, min(4, maxh) + 1)), int(rng.integers(2, min(4, maxw) + 1))
        return tuple(sorted({(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}))
    if style == 2 and maxw >= 3 and maxh >= 2:
        h = int(rng.integers(2, min(4, maxh) + 1))
        return tuple(sorted({(0, c) for c in range(3)} | {(r, 1) for r in range(h)}))
    cells = {(0, 0)}
    for _ in range(int(rng.integers(2, 8))):
        choices = sorted({(r + dr, c + dc) for r, c in cells
                          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
        choices = [p for p in choices if max(r for r, c in cells | {p}) - min(r for r, c in cells | {p}) < maxh
                   and max(c for r, c in cells | {p}) - min(c for r, c in cells | {p}) < maxw]
        if not choices:
            break
        cells.add(choices[int(rng.integers(len(choices)))])
    return _normalize(cells)


def _layout(rng, h, w):
    used, regions = set(), []
    groups = int(rng.integers(2, 5))
    half = (w - 2) // 2
    for _ in range(90):
        if len(regions) >= 2 * groups:
            break
        shape = _shape(rng, min(4, h - 2), min(4, half))
        sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        top = int(rng.integers(1, h - sh))
        leftmax = (w // 2) - sw
        if leftmax < 1:
            continue
        left = leftmax if rng.random() < .55 else int(rng.integers(1, leftmax + 1))
        cells = {(top + r, left + c) for r, c in shape}
        mirror = {(r, w - 1 - c) for r, c in cells}
        if cells & mirror or (cells | mirror) & used:
            continue
        if len(used | cells | mirror) > (h - 2) * (w - 2) * .63:
            continue
        regions.extend([cells, mirror])
        used |= cells | mirror
    if len(regions) < 4:
        return None
    if w % 2 and rng.random() < .30:
        length = int(rng.integers(1, min(4, h - 2) + 1))
        top = int(rng.integers(1, h - length))
        centered = {(r, w // 2) for r in range(top, top + length)}
        if not centered & used:
            regions.append(centered)
    return regions


def _scatter(rng, board, bg, pieces):
    h, w = board.shape
    for trial in range(8):
        ch = int(rng.integers(max(16, h + 5), 31)) if trial < 6 else 30
        cw = int(rng.integers(max(16, w + 5), 31)) if trial < 6 else 30
        br, bc = int(rng.integers(1, ch - h)), int(rng.integers(1, cw - w))
        a = np.full((ch, cw), bg, dtype=np.int8)
        a[br:br + h, bc:bc + w] = board
        blocked = np.zeros(a.shape, dtype=bool)
        blocked[0] = blocked[-1] = True
        blocked[:, 0] = blocked[:, -1] = True
        blocked[br - 1:br + h + 1, bc - 1:bc + w + 1] = True
        for color, shape in sorted(pieces, key=lambda x: -len(x[1])):
            sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
            choices = [(r, c) for r in range(1, ch - sh) for c in range(1, cw - sw)
                       if all(not blocked[r + dr, c + dc] for dr, dc in shape)]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            for dr, dc in shape:
                rr, cc = r + dr, c + dc
                a[rr, cc] = color
                for x, y in ((rr, cc), (rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    blocked[x, y] = True
        else:
            return a
    return None


def _make_pair(rng, regime, palette, orientation, require_a):
    bg, wall, mask = map(int, palette[:3])
    for attempt in range(400):
        h = int(rng.integers(7 if regime != 'test' else 9, 14))
        w = int(rng.integers(7, 14))
        regions = _layout(rng, h, w)
        if regions is None:
            continue
        signatures = sorted({_normalize(cells) for cells in regions})
        if len(signatures) < 2:
            continue
        ncolors = int(rng.integers(2, min(5, len(signatures)) + 1))
        colors = [int(palette[3 + i % ncolors]) for i in range(len(signatures))]
        rng.shuffle(colors)
        paint = dict(zip(signatures, colors))
        full = np.full((h, w), wall, dtype=np.int8)
        scaffold = full.copy()
        pieces = []
        for cells in regions:
            sig = _normalize(cells)
            color = paint[sig]
            pieces.append((color, sig))
            for cell in cells:
                full[cell] = color
                scaffold[cell] = bg
        # A full-color mirror-copy shortcut must be visibly false.
        if np.array_equal(full, full[:, ::-1]):
            continue
        for _ in range(12):
            mh = int(rng.integers(4, min(6, h - 2) + 1)) if regime == 'test' else int(rng.integers(1, 4))
            mr = int(rng.integers(1, h - mh))
            if regime == 'direct':
                mw = int(rng.integers(1, min(3, (w - 2) // 2) + 1))
                mc = int(rng.integers(1, w // 2 - mw + 1))
                if rng.random() < .5:
                    mc = w - mc - mw
            else:
                mw = (2 if w % 2 == 0 else 3) + 2 * int(rng.integers(2))
                mc = (w - mw) // 2
                # Occasionally unbalance the overlap without losing full orbits.
                if mw >= 4 and rng.random() < .35:
                    mc += int(rng.choice([-1, 1]))
            if mc < 1 or mc + mw >= w:
                continue
            target = scaffold[mr:mr + mh, mc:mc + mw]
            if not (np.any(target == bg) and np.any(target == wall)):
                continue
            if regime != 'direct' and any(all(mr <= r < mr + mh and mc <= c < mc + mw for r, c in cells)
                                          for cells in regions):
                continue
            board = scaffold.copy()
            board[mr:mr + mh, mc:mc + mw] = mask
            domains = _domains(board, bg, wall, mask, 1)
            if domains is None:
                continue
            if regime != 'direct':
                orphan = (board == mask) & (board[:, ::-1] == mask)
                if not (np.any(orphan & (full == wall)) and np.any(orphan & (full != wall))):
                    continue
            try:
                answers, _ = _assembly(board, bg, wall, mask, pieces, 1)
                if len(answers) != 1 or not np.array_equal(answers[0], full):
                    continue
                other, _ = _assembly(board, bg, wall, mask, pieces, 0)
                if any(not np.array_equal(x, full) for x in other):
                    continue
                if require_a:
                    relaxed, _ = _assembly(board, bg, wall, mask, pieces, None)
                    if len(relaxed) < 2:
                        continue
            except _SearchLimit:
                continue
            inp = _scatter(rng, board, bg, pieces)
            if inp is None:
                continue
            pair = {'input': np.rot90(inp, orientation).copy(),
                    'output': np.rot90(full, orientation).copy()}
            try:
                if not np.array_equal(_infer(pair['input'], node_limit=24000)[0], pair['output']):
                    continue
            except ValueError:
                continue
            return pair
    raise RuntimeError('Bounded sampler could not find a unique joint assembly')


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('Illegal example count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    vary_palette = rng.random() < .70
    orientation = int(rng.integers(4))
    mix_orientation = rng.random() < .65
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            regime = 'test' if split == 'test' else ('direct' if index == 0 or (index > 1 and rng.random() < .45) else 'coupled')
            require_a = split == 'test' or index == 1
            for _ in range(8):
                colors = rng.permutation(10) if vary_palette else palette
                turn = int(rng.integers(4)) if mix_orientation else orientation
                pair = _make_pair(rng, regime, colors, turn, require_a)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not generate distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    a_witness = b_witness = direct_witness = asymmetric = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            predicted, record = _infer(inp, node_limit=24000)
            _require(np.array_equal(out, predicted), 'Incorrect solved board')
            board, bg, wall, mask, pieces, box, axes, nodes = record
            _require(not np.any((out == bg) | (out == mask)), 'Unfilled or erased output')
            fixed = board == wall
            _require(np.all(out[fixed] == wall), 'Exposed walls changed')
            _require(np.all(out[board == bg] != wall), 'Exposed socket became a wall')
            color_areas = Counter()
            for color, shape in pieces:
                color_areas[color] += len(shape)
            for color, amount in color_areas.items():
                _require(np.count_nonzero(out == color) == amount, 'Inventory color area changed')
            _require(any(np.array_equal(out == wall, np.flip(out == wall, axis)) for axis in axes),
                     'Completed scaffold lacks its reflection')
            hidden = board == mask
            _require(np.any(out[hidden] == wall) and np.any(out[hidden] != wall), 'Mask must hide walls and sockets')
            if split == 'train':
                asymmetric |= not any(np.array_equal(out, np.flip(out, axis)) for axis in axes)
                local_b = False
                for axis in axes:
                    orphan = hidden & np.flip(hidden, axis)
                    direct_witness |= not np.any(orphan)
                    local_b |= np.any(orphan & (out == wall)) and np.any(orphan & (out != wall))
                b_witness |= local_b
                if local_b and not a_witness:
                    relaxed, _ = _assembly(board, bg, wall, mask, pieces, None)
                    a_witness |= len(relaxed) == 2
            else:
                relaxed, _ = _assembly(board, bg, wall, mask, pieces, None)
                _require(len(relaxed) == 2, 'Tests must still require reflection')
    _require(a_witness, 'Training must defeat assembly without reflection')
    _require(b_witness, 'Training must include fully erased wall and socket orbits')
    _require(direct_witness and asymmetric, 'Training must expose direct recovery and color asymmetry')


def self_test():
    shapes, boards, canvases, counts, masks, backgrounds, structures = set(), set(), set(), set(), set(), set(), set()
    signatures, shared, repeated, several, axes_seen = set(), 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p[k]) for split in ('train', 'test') for p in puzzle[split] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, record = _infer(pair['input'])
                _require(np.array_equal(predicted, pair['output']), 'Solver disagreement')
                board, bg, wall, mask, pieces, box, axes, nodes = record
                boards.add(board.shape)
                canvases.add(pair['input'].shape)
                counts.add(len(pieces))
                backgrounds.add(bg)
                shapes.update(s for c, s in pieces)
                masks.add(tuple(np.ptp(np.argwhere(board == mask), axis=0) + 1))
                axes_seen.update(axes)
                structures.add((board.shape, (pair['output'] == wall).tobytes()))
                repeated += any(n > 1 for n in Counter(pieces).values())
                several += any(len({s for k, s in pieces if k == c}) > 1 for c, s in pieces)
                for comp in _components(map(tuple, np.argwhere(pair['output'] != wall))):
                    shared += len({int(pair['output'][p]) for p in comp}) > 1
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for sp in puzzle for p, q in zip(puzzle[sp], again[sp]) for k in p),
                     'Nondeterministic puzzle')
    _require(len(signatures) == 42 and len(structures) >= 150, 'Insufficient structural variation')
    _require(len(shapes) >= 35 and len(boards) >= 25 and len(canvases) >= 50 and len(masks) >= 15,
             'Insufficient geometry variation')
    _require(len(counts) >= 4 and len(backgrounds) >= 8 and axes_seen == {0, 1}, 'Counts/palette/axis variation missing')
    _require(shared >= 15 and repeated >= 15 and several >= 15, 'Missing shared gaps or repeated inventories')
    for a, b in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(8100 + a * 5 + b, a, b)
        _require(len(p['train']) == a and len(p['test']) == b, 'Counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    p = generate_puzzle(19, 2, 1)
    for turn in (1, 2):
        moved = {sp: [{k: np.rot90(v, turn).copy() for k, v in pair.items()} for pair in p[sp]] for sp in p}
        validate_generated_puzzle(moved)
    perm = np.array([6, 2, 9, 0, 7, 1, 8, 3, 5, 4], dtype=np.int8)
    validate_generated_puzzle({sp: [{k: perm[v] for k, v in pair.items()} for pair in p[sp]] for sp in p})
    for kind in ('label', 'inventory'):
        bad = {sp: [{k: v.copy() for k, v in pair.items()} for pair in p[sp]] for sp in p}
        pair = bad['test'][0]
        if kind == 'label':
            pair['output'][1, 1] = (int(pair['output'][1, 1]) + 1) % 10
        else:
            board, bg, wall, mask, pieces, box = _read(pair['input'])
            r, s, c, d = box
            exterior = pair['input'] != bg
            exterior[r:s, c:d] = False
            pair['input'][tuple(np.argwhere(exterior)[0])] = bg
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            continue
        raise AssertionError('Corrupted label/inventory accepted')
    # Translation preserves orientation: two right-facing Ls cannot replace an
    # L and its separately supplied opposite-facing partner.
    board = np.full((5, 9), 5, dtype=np.int8)
    left = ((1, 1), (2, 1), (2, 2))
    right = tuple((r, 8 - c) for r, c in left)
    for cell in left + right:
        board[cell] = 0
    board[1:3, 2:4] = 8
    ls, rs = _normalize(left), _normalize(right)
    good = [(1, ls), (2, rs)]
    _require(len(_assembly(board, 0, 5, 8, good, 1)[0]) == 1, 'Oriented L fixture failed')
    _require(not _assembly(board, 0, 5, 8, [(1, rs), (2, rs)], 1)[0], 'Illegal reflected tile accepted')
    _require(not _assembly(board, 0, 5, 8, good + [(1, ls)], 1)[0], 'Surplus tile accepted')
    # Equal-item exchanges are one colored answer; different-color singleton
    # exchanges remain a real ambiguity despite binary reflection symmetry.
    board = np.full((5, 9), 5, dtype=np.int8)
    board[2, 1] = board[2, 7] = 0
    board[1:3, 3:4] = 8
    dot = ((0, 0),)
    _require(len(_assembly(board, 0, 5, 8, [(1, dot), (1, dot)], 1)[0]) == 1,
             'Equal tiles falsely ambiguous')
    _require(len(_assembly(board, 0, 5, 8, [(1, dot), (2, dot)], 1)[0]) == 2,
             'Colored ambiguity missed')
    return {'seeds': 42, 'board_shapes': len(boards), 'canvas_shapes': len(canvases),
            'oriented_shapes': len(shapes), 'scaffolds': len(structures), 'mask_shapes': len(masks),
            'tile_counts': sorted(counts), 'shared_socket_components': shared,
            'repeated_inventories': repeated, 'several_shapes_one_color': several}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
