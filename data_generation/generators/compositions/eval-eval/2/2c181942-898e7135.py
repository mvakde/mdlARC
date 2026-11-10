"""Keyed docking blueprints: a lock opens only on color AND full shape.

All conventions are visible locally. No parent code, fixtures, files, random
state, or demonstrations are consulted by solve(). Generation builds its answer
directly; validation recovers the inventory, scale, ports and fits from pixels.
"""
import hashlib
from functools import reduce
from math import gcd

import numpy as np


_N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_BAR = (((0, 1), (0, 2)), ((1, 3), (2, 3)),
        ((3, 1), (3, 2)), ((1, 0), (2, 0)))
_MOUTH = (((-1, 1), (-1, 2)), ((1, 4), (2, 4)),
          ((4, 1), (4, 2)), ((1, -1), (2, -1)))


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    result = []
    while pending:
        p = min(pending)
        pending.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                q = r + dr, c + dc
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _mask(cells):
    r0, r1, c0, c1 = _bounds(cells)
    result = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in cells:
        result[r - r0, c - c0] = True
    return result


def _key(mask):
    return tuple(mask.shape), np.asarray(mask, dtype=np.uint8).tobytes()


def _turns(mask):
    return {_key(np.rot90(mask, q)) for q in range(4)}


def _class(mask):
    return min(_turns(mask) | _turns(np.fliplr(mask)))


def _run_gcd(mask):
    lengths = []
    for array in (mask, mask.T):
        for row in array:
            boundaries = np.flatnonzero(np.diff(np.pad(row.astype(int), (1, 1))))
            lengths.extend(int(b - a) for a, b in zip(boundaries[::2], boundaries[1::2]))
    return reduce(gcd, lengths)


def _enlarge(grid, scale):
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer grid required"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=True)


def _read_board(board, bg, structure):
    """Return the uniquely patterned hub and its radially attached chambers."""
    assert min(board.shape) >= 6
    assert all(np.all(edge == structure) for edge in
               (board[0], board[-1], board[:, 0], board[:, -1]))
    hubs = []
    for r in range(1, board.shape[0] - 4):
        for c in range(1, board.shape[1] - 4):
            patch = board[r:r + 4, c:c + 4]
            colors = [int(patch[cells[0]]) for cells in _BAR]
            if len(set(colors)) != 4 or bg in colors or structure in colors:
                continue
            wanted = np.full((4, 4), structure, dtype=np.int8)
            for color, cells in zip(colors, _BAR):
                for p in cells:
                    wanted[p] = color
            if np.array_equal(patch, wanted):
                hubs.append((r, c, colors))
    assert len(hubs) == 1, "exactly one four-port hub required"
    hr, hc, labels = hubs[0]
    special = (board != structure) & (board != bg)
    assert np.count_nonzero(special) == 8, "only the hub supplies colored board cells"
    ports = {color: {(r + hr, c + hc) for r, c in cells}
             for color, cells in zip(labels, _BAR)}
    rooms = []
    for cells in _components(board == bg):
        cellset = set(cells)
        neighbors = {(r + dr, c + dc) for r, c in cells for dr, dc in _N4}
        touched = [color for color, ps in ports.items() if neighbors & ps]
        assert len(touched) == 1, "a chamber must touch exactly one labeled port"
        color = touched[0]
        side = labels.index(color)
        mouths = {(r + hr, c + hc) for r, c in _MOUTH[side]}
        assert mouths <= cellset and ports[color] <= neighbors, "two-cell mouth required"
        assert (all(r < hr for r, c in cells) if side == 0 else
                all(c >= hc + 4 for r, c in cells) if side == 1 else
                all(r >= hr + 4 for r, c in cells) if side == 2 else
                all(c < hc for r, c in cells)), "chamber crosses behind its port"
        rooms.append(dict(cells=cells, color=color, side=side, mask=_mask(cells)))
    assert 2 <= len(rooms) <= 4
    assert len({room['side'] for room in rooms}) == len(rooms)
    return dict(hub=(hr, hc), labels=labels, rooms=rooms)


def _recover(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > grid.size // 2, "background must be a strict majority"
    candidates = []
    for structure in range(10):
        if structure == bg:
            continue
        for cells in _components(grid == structure):
            r0, r1, c0, c1 = _bounds(cells)
            board = grid[r0:r1, c0:c1]
            if min(board.shape) < 6:
                continue
            try:
                info = _read_board(board, bg, structure)
            except AssertionError:
                continue
            candidates.append((r0, r1, c0, c1, structure, board.copy(), info))
    assert len(candidates) == 1, "one unambiguous rectangular blueprint required"
    r0, r1, c0, c1, structure, board, info = candidates[0]
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    moat = outside.copy()
    moat[:] = False
    moat[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
    assert not np.any(moat & outside & (grid != bg)), "blueprint clearance violated"
    pieces = []
    for cells in _components((grid != bg) & outside):
        colors = {int(grid[p]) for p in cells}
        assert len(colors) == 1 and structure not in colors, "monochrome loose keys required"
        a, b, c, d = _bounds(cells)
        mask = grid[a:b, c:d] != bg
        assert np.count_nonzero(mask) == len(cells)
        pieces.append(dict(color=colors.pop(), raw_mask=mask, bounds=(a, b, c, d)))
    assert 2 <= len(pieces) <= 5
    # The largest common solid-square unit is the scale. Generated inventories
    # are primitive collectively: their compressed run-length gcd is one.
    scale = reduce(gcd, (_run_gcd(p['raw_mask']) for p in pieces))
    assert scale in (2, 3), "common block scale must be two or three"
    assert max(board.shape) * scale <= 30
    for p in pieces:
        mask = p['raw_mask']
        h, w = mask.shape
        assert h % scale == w % scale == 0
        blocks = mask.reshape(h // scale, scale, w // scale, scale)
        small = blocks.all(axis=(1, 3))
        assert np.array_equal(_enlarge(small, scale), mask), "partial macrocell"
        assert len(_components(small)) == 1
        p['mask'], p['class'] = small, _class(small)
    logical = board.copy()
    for room in info['rooms']:
        room['class'] = _class(room['mask'])
        fitting = [p for p in pieces if p['color'] == room['color'] and p['class'] == room['class']]
        assert len(fitting) <= 1, "duplicate accepted key"
        room['active'] = bool(fitting)
        room['reflection'] = bool(fitting and _key(room['mask']) not in _turns(fitting[0]['mask']))
        room['rotation'] = bool(fitting and not room['reflection'] and
                                _key(room['mask']) != _key(fitting[0]['mask']))
        if fitting:
            for p in room['cells']:
                logical[p] = room['color']
    info.update(background=bg, structural=structure, scale=scale, board=board,
                bounds=(r0, r1, c0, c1), pieces=pieces, logical=logical)
    return _enlarge(logical, scale), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive scale, activation and enlarged output solely from the given grid."""
    return _recover(input_grid)[0]


def _north_shape(rng, depth):
    # The terminal two-cell neck is preserved; unequal side growth can make
    # off-center, chiral, concave bodies. The mold resolves orientation ties.
    cells = {(-r, c) for r in range(1, depth + 1) for c in (0, 1)}
    extra = int(rng.integers(0, min(8, 12 - len(cells)) + 1))
    for _ in range(extra):
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4}
                         - cells)
        options = [p for p in options if -depth <= p[0] <= -2 and -2 <= p[1] <= 3]
        if not options:
            break
        cells.add(options[int(rng.integers(len(options)))])
    return cells


def _position_shape(north, side, reflect):
    coords = {(r, (1 - c if reflect else c) + 1) for r, c in north}
    for _ in range(side):
        coords = {(c, 3 - r) for r, c in coords}
    return coords


def _wrong_shape(rng, area, forbidden_class):
    # Equal area blocks area-only matching. Random connected growth is not a
    # second rule: these objects simply fail the full lock silhouette test.
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < area:
            options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4} - cells)
            options = [p for p in options if
                       max(r for r, c in cells | {p}) - min(r for r, c in cells | {p}) < 4 and
                       max(c for r, c in cells | {p}) - min(c for r, c in cells | {p}) < 5]
            cells.add(options[int(rng.integers(len(options)))])
        mask = _mask(cells)
        if _class(mask) != forbidden_class and _run_gcd(mask) == 1:
            return mask
    # A 1-by-area line is primitive and differs from every two-wide chamber.
    return np.ones((1, area), dtype=bool)


def _orientation(rng, mask, witness=None):
    candidates = [np.rot90(base, q).copy() for base in (mask, np.fliplr(mask)) for q in range(4)]
    if witness == 'reflection':
        candidates = [m for m in candidates if _key(mask) not in _turns(m)]
    elif witness == 'rotation':
        candidates = [m for m in candidates if _key(m) != _key(mask) and _key(mask) in _turns(m)]
    assert candidates
    return candidates[int(rng.integers(len(candidates)))]


def _free_origins(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), dtype=int)
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = s[h:, w:] - s[:-h, w:] - s[h:, :-w] + s[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, inventory, scale, bg):
    objects = [(board, np.ones(board.shape, dtype=bool))]
    for color, small in inventory:
        mask = _enlarge(small, scale)
        objects.append((np.full(mask.shape, color, dtype=np.int8), mask))
    for attempt in range(24):
        h, w = map(int, rng.integers(21, 31, size=2)) if attempt < 12 else (30, 30)
        grid = np.full((h, w), bg, dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(rng.permutation(len(objects)))
        order.sort(key=lambda i: -objects[i][1].size)
        for i in order:
            data, mask = objects[i]
            bh, bw = mask.shape
            origins = _free_origins(blocked, bh, bw)
            if not len(origins):
                break
            r, c = map(int, origins[int(rng.integers(len(origins)))])
            grid[r:r + bh, c:c + bw][mask] = data[mask]
            blocked[max(0, r - 1):r + bh + 1, max(0, c - 1):c + bw + 1] = True
        else:
            if np.count_nonzero(grid == bg) <= grid.size // 2:
                continue
            if rng.random() < .7:
                r0, r1, c0, c1 = _bounds(np.argwhere(grid != bg))
                margins = list(map(int, rng.integers(0, 4, size=4)))
                cropped = grid[max(0, r0 - margins[0]):r1 + margins[1],
                               max(0, c0 - margins[2]):c1 + margins[3]]
                if np.count_nonzero(cropped == bg) > cropped.size // 2:
                    grid = cropped.copy()
            return grid
    return None


def _make_pair(rng, scale, sides, witness, previous_bg, novel_classes, seen):
    for attempt in range(160):
        if novel_classes is not None and attempt and attempt % 20 == 0:
            # A dense scale-three board has only two radial cells per opposing
            # room. Resample the footprint when its small shape vocabulary is
            # already exhausted by training; adjacent ports permit deeper keys.
            sides = sorted(map(int, rng.choice(4, int(rng.integers(2, 5)), replace=False)))
        palette = list(map(int, rng.permutation(10)))
        if palette[0] == previous_bg:
            palette = palette[1:] + palette[:1]
        bg, structure = palette[:2]
        labels = palette[2:6]
        n_active = int(rng.integers(1, len(sides)))
        active = set(map(int, rng.choice(sides, n_active, replace=False)))
        chosen = int(rng.choice(sorted(active)))
        blocked_side = int(rng.choice(sorted(set(sides) - active)))
        norths = {}
        for side in sides:
            depth_max = 4 if scale == 2 else (2 if (side + 2) % 4 in sides else 3)
            norths[side] = _north_shape(rng, int(rng.integers(2, depth_max + 1)))
        if witness:
            # Both are chiral. They differ in area and shoulder offset.
            left = -1 if witness == 'reflection' else -2
            norths[chosen] = {(-1, 0), (-1, 1)} | {(-2, c) for c in range(left, 2)}
        repeated = witness == 'reflection' or rng.random() < .35
        if repeated:
            norths[blocked_side] = norths[chosen].copy()
        relative = {side: _position_shape(norths[side], side, rng.random() < .5) for side in sides}
        union = set().union(*(relative.values()))
        if sum(map(len, relative.values())) != len(union):
            continue
        if any(any((r + dr, c + dc) in relative[b] for r, c in relative[a] for dr, dc in _N4)
               for a in sides for b in sides if a < b):
            continue
        all_cells = union | {(r, c) for r in range(4) for c in range(4)}
        lo_r, hi_r, lo_c, hi_c = _bounds(all_cells)
        limit = 30 // scale
        need_h, need_w = hi_r - lo_r + 2, hi_c - lo_c + 2
        if max(need_h, need_w) > limit:
            continue
        extras = [int(rng.integers(0, min(3, limit - n) + 1)) for n in (need_h, need_w)]
        top, left = 1 + int(rng.integers(extras[0] + 1)), 1 + int(rng.integers(extras[1] + 1))
        shift = top - lo_r, left - lo_c
        board = np.full((need_h + extras[0], need_w + extras[1]), structure, dtype=np.int8)
        hr, hc = shift
        for color, cells in zip(labels, _BAR):
            for r, c in cells:
                board[r + hr, c + hc] = color
        for cells in relative.values():
            for r, c in cells:
                board[r + hr, c + hc] = bg
        try:
            parsed = _read_board(board, bg, structure)
        except AssertionError:
            continue
        rooms = {room['side']: room for room in parsed['rooms']}
        if novel_classes is not None and not any(_class(rooms[s]['mask']) not in novel_classes for s in active):
            continue
        logical = board.copy()
        inventory = []
        for side in sorted(active):
            room = rooms[side]
            mask = _orientation(rng, room['mask'], witness if side == chosen else None)
            inventory.append((labels[side], mask))
            for p in room['cells']:
                logical[p] = labels[side]
        blocked_room = rooms[blocked_side]
        decoy = _wrong_shape(rng, int(blocked_room['mask'].sum()), _class(blocked_room['mask']))
        inventory.append((labels[blocked_side], _orientation(rng, decoy)))
        if not repeated or rng.random() < .35:
            inventory.append((palette[6], _orientation(rng, blocked_room['mask'])))
        if len(inventory) > 5 or reduce(gcd, (_run_gcd(m) for color, m in inventory)) != 1:
            continue
        inp = _pack(rng, board, inventory, scale, bg)
        if inp is None or _key(inp) in seen:
            continue
        output = _enlarge(logical, scale)
        recovered, info = _recover(inp)
        assert np.array_equal(recovered, output), "construction/recovery disagreement"
        return dict(input=inp, output=output), info
    raise RuntimeError('bounded chamber/inventory packing exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _schedules(rng, ntrain, ntest):
    mode = int(rng.integers(5))
    if mode < 2:
        a, b = (2, 3) if mode == 0 else (3, 2)
        return [a] * ntrain, [b] * ntest
    if mode < 4:
        k = mode
        return [k] * ntrain, [k] * ntest
    return [2, 3] + list(map(int, rng.choice([2, 3], ntrain - 2))), list(map(int, rng.choice([2, 3], ntest)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    schedules = _schedules(rng, nt, nq)
    puzzle, seen, train_classes = dict(train=[], test=[]), set(), set()
    previous_bg = None
    missing = set(range(4))
    for split, scales in zip(('train', 'test'), schedules):
        for i, scale in enumerate(scales):
            count = int(rng.integers(2, 5))
            if split == 'train' and i == 1:
                count = max(count, len(missing))
                sides = sorted(missing | set(map(int, rng.choice(sorted(set(range(4)) - missing),
                                                                  count - len(missing), replace=False))))
            else:
                sides = sorted(map(int, rng.choice(4, count, replace=False)))
            witness = ('reflection' if i == 0 else 'rotation' if i == 1 else None) if split == 'train' else None
            pair, info = _make_pair(rng, scale, sides, witness, previous_bg,
                                    train_classes if split == 'test' else None, seen)
            assert _key(pair['input']) not in seen
            seen.add(_key(pair['input']))
            previous_bg = info['background']
            if split == 'train':
                missing -= set(sides)
                train_classes.update(room['class'] for room in info['rooms'])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    expected, info = _recover(inp)
    assert np.array_equal(out, expected), 'incorrect enlarged lock assembly'
    rooms, pieces = info['rooms'], info['pieces']
    assert any(r['active'] for r in rooms) and any(not r['active'] for r in rooms)
    closed = [r for r in rooms if not r['active']]
    assert any(any(p['class'] == r['class'] and p['color'] != r['color'] for p in pieces) for r in closed), 'missing color-necessity witness'
    assert any(any(p['color'] == r['color'] and p['class'] != r['class'] and
                   p['mask'].sum() == r['mask'].sum() for p in pieces) for r in closed), 'missing equal-area shape-necessity witness'
    assert all(4 <= r['mask'].sum() <= 12 for r in rooms)
    # A raw-sized board, unconditional port coloring, and shape-only opening
    # each give a different answer. These counterfactuals use actual pixels.
    assert out.shape != info['board'].shape
    all_open, shape_only = info['board'].copy(), info['logical'].copy()
    color_only = info['logical'].copy()
    for r in rooms:
        for p in r['cells']:
            all_open[p] = r['color']
            if any(key['class'] == r['class'] for key in pieces):
                shape_only[p] = r['color']
            if any(key['color'] == r['color'] for key in pieces):
                color_only[p] = r['color']
    assert not np.array_equal(all_open, info['logical'])
    assert not np.array_equal(shape_only, info['logical'])
    assert not np.array_equal(color_only, info['logical'])
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            k = _key(np.asarray(pair['input']))
            assert k not in seen, 'duplicate input'
            seen.add(k)
            infos[split].append(info)
    train = infos['train']
    assert any(r['reflection'] for i in train for r in i['rooms'])
    assert any(r['rotation'] for i in train for r in i['rooms'])
    assert set(range(4)) == {r['side'] for i in train for r in i['rooms']}
    assert any(len({r['class'] for r in i['rooms']}) < len(i['rooms']) for i in train)
    assert len({i['background'] for i in train}) >= 2
    classes = {r['class'] for i in train for r in i['rooms']}
    assert all(any(r['active'] and r['class'] not in classes for r in i['rooms']) for i in infos['test'])


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in puzzle[split])


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.digest()


def self_test():
    """Bounded streaming suite: retain metrics and hashes, not 42 puzzles."""
    signatures, shapes, rooms, schedules, active_counts, piece_counts = set(), set(), set(), set(), set(), set()
    backgrounds, board_shapes, sidesets, reflected = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        grouped = {s: [_recover(p['input'])[1] for p in puzzle[s]] for s in puzzle}
        schedules.add(tuple(tuple(sorted({i['scale'] for i in grouped[s]})) for s in ('train', 'test')))
        for split in ('train', 'test'):
            for pair, info in zip(puzzle[split], grouped[split]):
                shapes.add(pair['input'].shape)
                board_shapes.add(info['board'].shape)
                backgrounds.add(info['background'])
                sidesets.add(tuple(sorted(r['side'] for r in info['rooms'])))
                rooms.update(r['class'] for r in info['rooms'])
                active_counts.add(sum(r['active'] for r in info['rooms']))
                piece_counts.add(len(info['pieces']))
                reflected += sum(r['reflection'] for r in info['rooms'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        if seed == 0:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
            damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupted output accepted')
            raw = _recover(puzzle['train'][0]['input'])[1]['pieces'][0]
            broken = puzzle['train'][0]['input'].copy()
            r0, r1, c0, c1 = raw['bounds']
            r, c = np.argwhere(raw['raw_mask'])[0]
            broken[r0 + r, c0 + c] = grouped['train'][0]['background']
            try:
                solve(broken)
            except AssertionError:
                pass
            else:
                raise AssertionError('partial macrocell accepted')
    assert len(signatures) == 42 and len(shapes) >= 30 and len(board_shapes) >= 15
    assert len(rooms) >= 45 and len(backgrounds) >= 8 and len(sidesets) >= 9
    assert active_counts == {1, 2, 3} and piece_counts >= {2, 3, 4, 5}
    assert len(schedules) >= 5 and reflected >= 42
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            p = generate_puzzle(10000 + 10 * a + b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
