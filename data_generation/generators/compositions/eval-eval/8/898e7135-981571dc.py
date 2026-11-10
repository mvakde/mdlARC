"""Scattered mirror mosaic: fit, register and repair damaged scaled tiles.

Only the standard library and NumPy are used. No files or traces are written.
Zero is exterior/empty cavity, nine is erased tile material. Other colors are
local payloads, except the color of the small map's complete structural frame.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    # Python coordinates avoid NumPy scalar arithmetic in the graph walk.
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _crop(array):
    r0, r1, c0, c1 = _bounds(np.argwhere(array != 0))
    return array[r0:r1, c0:c1].copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _orientations(a):
    result = {}
    for base in (a, a[:, ::-1]):
        # These are rot90(base, 0..3) in the same order. Direct 2D views avoid
        # repeated generic axis normalization for the many tiny layout masks.
        for view in (base, base.T[::-1, :], base[::-1, ::-1], base.T[:, ::-1]):
            key = _key(view)
            if key not in result:
                result[key] = view.copy()
    return list(result.values())


def _class(mask):
    return min(_key(x.astype(np.int8)) for x in _orientations(mask))


def _orbit(p, shape):
    r, c = p
    h, w = shape
    return tuple(sorted({(r, c), (h - 1 - r, c),
                         (r, w - 1 - c), (h - 1 - r, w - 1 - c)}))


def _geometry(board, structural):
    mask = board != structural
    assert not mask[0].any() and not mask[-1].any()
    assert not mask[:, 0].any() and not mask[:, -1].any()
    assert np.array_equal(mask, mask[::-1]) and np.array_equal(mask, mask[:, ::-1])
    holes = _components(mask)
    orbit_ids = np.full(board.shape, -1, dtype=int)
    orbits = []
    for p in map(tuple, np.argwhere(mask)):
        if orbit_ids[p] < 0:
            orb = _orbit(p, board.shape)
            for q in orb:
                assert mask[q]
                orbit_ids[q] = len(orbits)
            orbits.append(orb)
    return holes, orbits, orbit_ids


def _parse(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
    assert all(1 <= n <= 30 for n in a.shape) and np.all((a >= 0) & (a <= 9))
    candidates = []
    for structural in range(1, 9):
        cells = np.argwhere(a == structural)
        if not len(cells):
            continue
        r0, r1, c0, c1 = _bounds(cells)
        board = a[r0:r1, c0:c1]
        if min(board.shape) < 5:
            continue
        if (np.all(board[0] == structural) and np.all(board[-1] == structural)
                and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)
                and np.any(board == 0) and not np.any(board == 9)):
            # A solid rectangular tile is not another map. A candidate must
            # also exhibit the map's symmetric, multi-cavity geometry.
            try:
                candidate_holes, _, _ = _geometry(board, structural)
            except AssertionError:
                continue
            if 2 <= len(candidate_holes) <= 6:
                candidates.append((structural, (r0, r1, c0, c1), board.copy()))
    assert len(candidates) == 1, "one complete, uniquely colored map frame is required"
    structural, bounds, board = candidates[0]
    assert not np.any(board == 9), "the small map has clues, never erasure blocks"
    holes, orbits, orbit_ids = _geometry(board, structural)
    assert 2 <= len(holes) <= 6
    r0, r1, c0, c1 = bounds
    external = a.copy()
    external[r0:r1, c0:c1] = 0
    moat = external[max(0, r0-1):r1+1, max(0, c0-1):c1+1]
    assert not moat.any(), "objects must have an empty moat"
    components = _components(external != 0)
    assert len(components) == len(holes)
    area = sum(map(len, holes))
    ratio, remainder = divmod(sum(map(len, components)), area)
    scale = math.isqrt(ratio)
    assert not remainder and scale >= 2 and scale * scale == ratio
    assert max(board.shape) * scale <= 30
    pieces = []
    for cells in components:
        u0, u1, v0, v1 = _bounds(cells)
        piece = external[u0:u1, v0:v1]
        h, w = piece.shape
        assert h % scale == w % scale == 0
        small = piece[::scale, ::scale].copy()
        assert np.array_equal(piece, np.repeat(np.repeat(small, scale, 0), scale, 1))
        assert len(_components(small != 0)) == 1
        pieces.append(small)
    return dict(board=board, structural=structural, bounds=bounds, scale=scale,
                holes=holes, pieces=pieces, orbits=orbits, ids=orbit_ids)


def _options(data, symmetry=True):
    """Enumerate actual raster placements, preserving every observed tile color."""
    board, ids = data['board'], data['ids']
    # Pieces do not change while clues are added. This cache belongs only to
    # this parsed input/construction, and contains no answers or latent state.
    if '_piece_orientations' not in data:
        data['_piece_orientations'] = [_orientations(piece) for piece in data['pieces']]
    options = []
    for cells in data['holes']:
        r0, r1, c0, c1 = _bounds(cells)
        mask = np.zeros((r1-r0, c1-c0), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        opts = []
        for pi, orientations in enumerate(data['_piece_orientations']):
            for oriented in orientations:
                if oriented.shape != mask.shape or not np.array_equal(oriented != 0, mask):
                    continue
                values = tuple(int(oriented[r-r0, c-c0]) for r, c in cells)
                if any(board[p] != 0 and v != 9 and board[p] != v for p, v in zip(cells, values)):
                    continue
                colors = {}
                valid = True
                for p, v in zip(cells, values):
                    if v == 9:
                        continue
                    oi = int(ids[p])
                    if symmetry and oi in colors and colors[oi] != v:
                        valid = False
                        break
                    colors[oi] = v
                if valid:
                    opts.append((pi, tuple(sorted(colors.items())), values))
        assert opts, "a cavity has no matching tile"
        options.append(opts)
    return options


def _reconstructions(data, limit=2):
    """Finite CSP: shape assignment, clue registration and reflection agreement.

    Keep at most two different completions, rather than all assignments. An
    unwitnessed orbit remains -1 and counts as a nonunique interpretation.
    Memoization and a node cap bound even malformed-input work and memory.
    """
    options = _options(data)
    known = [-1] * len(data['orbits'])
    for p in map(tuple, np.argwhere((data['board'] != 0) &
                                  (data['board'] != data['structural']))):
        oi, color = int(data['ids'][p]), int(data['board'][p])
        assert known[oi] in (-1, color), "map clues contradict a mirror orbit"
        known[oi] = color
    seen, answers = set(), {}
    node_count = 0

    def visit(remaining, used, colors, chosen):
        nonlocal node_count
        if len(answers) >= limit:
            return
        state = (remaining, used, colors)
        if state in seen:
            return
        seen.add(state)
        node_count += 1
        if node_count > 60000:
            raise ValueError("placement search exceeded the family work bound")
        if not remaining:
            answers.setdefault(colors, chosen)
            return
        best = None
        for hi in range(len(options)):
            if not remaining & (1 << hi):
                continue
            usable = [o for o in options[hi] if not used & (1 << o[0]) and
                      all(colors[oi] in (-1, v) for oi, v in o[1])]
            if not usable:
                return
            if best is None or len(usable) < len(best[1]):
                best = (hi, usable)
        hi, usable = best
        for option in usable:
            updated = list(colors)
            for oi, v in option[1]:
                updated[oi] = v
            visit(remaining ^ (1 << hi), used | (1 << option[0]), tuple(updated),
                  chosen + ((hi, option),))
            if len(answers) >= limit:
                return

    visit((1 << len(options)) - 1, 0, tuple(known), ())
    return answers, node_count


def _logical(data, colors):
    result = np.full(data['board'].shape, data['structural'], dtype=int)
    for color, orb in zip(colors, data['orbits']):
        for p in orb:
            result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive one uniquely registered mosaic entirely from this input."""
    data = _parse(input_grid)
    answers, _ = _reconstructions(data)
    assert len(answers) == 1, "multiple colored mosaics or no assembly"
    colors = next(iter(answers))
    assert -1 not in colors, "an entire reflection orbit is missing"
    logical = _logical(data, colors)
    return np.repeat(np.repeat(logical, data['scale'], 0), data['scale'], 1)


def _reflect_mask(mask):
    return mask | mask[::-1] | mask[:, ::-1] | mask[::-1, ::-1]


def _grow(rng, area):
    cells = {(0, 0)}
    for _ in range(area - 1):
        choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _N4} - cells)
        r0, r1, c0, c1 = _bounds(cells)
        choices = [q for q in choices if max(r1-1, q[0]) - min(r0, q[0]) < 4
                   and max(c1-1, q[1]) - min(c0, q[1]) < 4]
        if not choices:
            break
        cells.add(choices[int(rng.integers(len(choices)))])
    r0, r1, c0, c1 = _bounds(cells)
    mask = np.zeros((r1-r0, c1-c0), dtype=bool)
    for r, c in cells:
        mask[r-r0, c-c0] = True
    return mask


def _layout(rng, scale):
    """Place reflected shape groups, allowing them to meet their own folds.

    Groups may have one, two or four cavities; separate groups never touch.
    There is no fixed board position, cavity count, quadrant size or shape list.
    """
    for attempt in range(160):
        h, w = map(int, rng.integers(7, min(14, 30 // scale) + 1, size=2))
        occupied = np.zeros((h, w), dtype=bool)
        part_count = 0
        for index in range(int(rng.integers(2, 5))):
            # Occupancy changes only after a group is accepted. Its moat is
            # therefore constant throughout this group's placement attempts.
            grown = occupied.copy()
            grown[1:, :] |= occupied[:-1, :]
            grown[:-1, :] |= occupied[1:, :]
            grown[:, 1:] |= occupied[:, :-1]
            grown[:, :-1] |= occupied[:, 1:]
            for _ in range(24):
                if index == 0:
                    kind = int(rng.integers(4))
                    base = (np.ones((1, int(rng.integers(3, 5))), dtype=bool) if kind == 0 else
                            np.ones((2, 2), dtype=bool) if kind == 1 else
                            np.array([[1, 1, 1], [0, 1, 0]], dtype=bool) if kind == 2 else
                            np.array([[1, 0, 1], [1, 1, 1]], dtype=bool))
                else:
                    base = _grow(rng, int(rng.integers(1, 8)))
                orientations = _orientations(base)
                base = orientations[int(rng.integers(len(orientations)))]
                sh, sw = base.shape
                if h-sh-1 < 1 or w-sw-1 < 1:
                    continue
                r, c = int(rng.integers(1, h-sh)), int(rng.integers(1, w-sw))
                patch = np.zeros((h, w), dtype=bool)
                patch[r:r+sh, c:c+sw] = base
                patch = _reflect_mask(patch)
                if np.any(patch & grown):
                    continue
                parts = _components(patch)
                # The tested moat proves the two masks' components cannot join.
                if part_count + len(parts) > 6 or any(len(x) > 10 or max(b-a for a, b in
                       ((_bounds(x)[0], _bounds(x)[1]), (_bounds(x)[2], _bounds(x)[3]))) > 5 for x in parts):
                    continue
                occupied |= patch
                part_count += len(parts)
                break
        parts = _components(occupied)
        classes = set()
        ambiguous_shape = False
        for cells in parts:
            r0, r1, c0, c1 = _bounds(cells)
            shape = occupied[r0:r1, c0:c1].copy()
            # Other cavities cannot occur inside this small bounding box when
            # constructing the actual shape mask.
            shape[:] = False
            for r, c in cells:
                shape[r-r0, c-c0] = True
            classes.add(_class(shape))
            ambiguous_shape |= len(cells) >= 3 and len(_orientations(shape)) < 8
        if (2 <= len(parts) <= 6 and len(classes) >= 2 and occupied.sum() >= 16
                and ambiguous_shape):
            return occupied
    raise RuntimeError("bounded symmetric cavity construction exhausted")


def _pack(rng, board, pieces, scale):
    objects = [board] + [np.repeat(np.repeat(p, scale, 0), scale, 1) for p in pieces]
    for attempt in range(40):
        h, w = (map(int, rng.integers(22, 31, size=2)) if attempt < 20 else (30, 30))
        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        order = list(rng.permutation(len(objects)))
        # Larger objects first, random ties. The board has no special location.
        order.sort(key=lambda i: -objects[i].size)
        for i in order:
            obj = objects[i]
            sh, sw = obj.shape
            if sh > h or sw > w:
                break
            summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            totals = summed[sh:, sw:] - summed[:-sh, sw:] - summed[sh:, :-sw] + summed[:-sh, :-sw]
            positions = np.argwhere(totals == 0)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            grid[r:r+sh, c:c+sw] = obj
            blocked[max(0, r-1):r+sh+1, max(0, c-1):c+sw+1] = True
        else:
            if rng.random() < .7:
                r0, r1, c0, c1 = _bounds(np.argwhere(grid != 0))
                r0, c0 = max(0, r0-int(rng.integers(3))), max(0, c0-int(rng.integers(3)))
                r1, c1 = min(h, r1+int(rng.integers(3))), min(w, c1+int(rng.integers(3)))
                grid = grid[r0:r1, c0:c1].copy()
            return grid
    return None


def _shape_only_wrong(data, logical):
    """Can a complete shape/clue fit put a surviving color in a wrong place?"""
    options = _options(data, symmetry=False)
    edges = [{o[0] for o in opts} for opts in options]

    def matching(remaining, used):
        if not remaining:
            return True
        hi = min(remaining, key=lambda i: len(edges[i] - used))
        return any(matching(remaining - {hi}, used | {pi}) for pi in edges[hi] - used)

    for hi, opts in enumerate(options):
        cells = data['holes'][hi]
        for pi, _, values in opts:
            wrong = any(v != 9 and v != logical[p] for p, v in zip(cells, values))
            if wrong and matching(set(range(len(options))) - {hi}, {pi}):
                return True
    return False


def _make_pair(rng, scale, hard):
    for attempt in range(90):
        mask = _layout(rng, scale)
        structural = int(rng.integers(1, 9))
        pool = [c for c in range(1, 9) if c != structural]
        palette = list(map(int, rng.choice(pool, size=int(rng.integers(3, 7)), replace=False)))
        board = np.full(mask.shape, structural, dtype=int)
        board[mask] = 0
        holes, orbits, ids = _geometry(board, structural)
        values = list(map(int, rng.choice(palette, size=len(orbits))))
        if len(orbits) < 6:
            continue
        # Both random artwork and folded bands/outlines; no smoothing of speckles.
        style = int(rng.integers(3))
        if style:
            phase = int(rng.integers(len(palette)))
            for oi, orb in enumerate(orbits):
                r, c = orb[0]
                index = (r + c + phase) if style == 1 else max(r, c) + phase
                if rng.random() < .8:
                    values[oi] = palette[index % len(palette)]
        full = _logical(dict(board=board, structural=structural, orbits=orbits), values)
        damaged = full.copy()
        eligible = [i for i, orb in enumerate(orbits) if len(orb) == 4 and
                    len({next(j for j, cells in enumerate(holes) if p in cells) for p in orb}) > 1]
        if not eligible:
            continue
        protected = set(map(int, rng.choice(eligible, size=min(len(eligible), 2 if hard else 1), replace=False)))
        for oi, orb in enumerate(orbits):
            if oi in protected:
                keep = int(rng.integers(4))
                for j, p in enumerate(orb):
                    if j != keep:
                        damaged[p] = 9
            else:
                probability = float(rng.uniform(.25, .48) if hard else rng.uniform(.08, .25))
                erased = rng.random(len(orb)) < probability
                if erased.all():
                    erased[int(rng.integers(len(orb)))] = False
                for p, missing in zip(orb, erased):
                    if missing:
                        damaged[p] = 9
        pieces = []
        for cells in holes:
            r0, r1, c0, c1 = _bounds(cells)
            piece = np.zeros((r1-r0, c1-c0), dtype=int)
            for r, c in cells:
                piece[r-r0, c-c0] = damaged[r, c]
            choices = _orientations(piece)
            pieces.append(choices[int(rng.integers(len(choices)))])
        rng.shuffle(pieces)
        data = dict(board=board, structural=structural, holes=holes, pieces=pieces,
                    orbits=orbits, ids=ids, scale=scale)
        # Counterexample-guided registration. A clue is added only to rule out a
        # real alternative colored assembly, never a hidden positional tie-break.
        anchored = set()
        try:
            for _ in range(min(9, max(2, len(orbits)//2)) + 1):
                answers, _ = _reconstructions(data)
                if len(answers) == 1 and next(iter(answers)) == tuple(values):
                    break
                wrong = next((x for x in answers if x != tuple(values)), None)
                if wrong is None:
                    break
                disagreements = [i for i, v in enumerate(wrong) if v != values[i]
                                 and i not in protected and i not in anchored]
                if not disagreements:
                    break
                oi = disagreements[int(rng.integers(len(disagreements)))]
                p = orbits[oi][int(rng.integers(len(orbits[oi])))]
                board[p] = values[oi]
                anchored.add(oi)
            else:
                continue
            if not anchored or len(answers) != 1 or next(iter(answers)) != tuple(values):
                continue
            unanchored = set(range(len(orbits))) - anchored
            if len(unanchored) < 3 or len({values[i] for i in unanchored}) < 2:
                continue
            if not _shape_only_wrong(data, full):
                continue
        except (ValueError, AssertionError):
            continue
        grid = _pack(rng, board, pieces, scale)
        if grid is None:
            continue
        output = np.repeat(np.repeat(full, scale, 0), scale, 1)
        pair = {'input': grid, 'output': output}
        # Interchangeable tiles can give the same completed artwork but move the
        # missing blocks between orbits. Check the input-recovered assignment,
        # rather than relying on the construction's hidden damage arrangement.
        try:
            features = _pair_features(pair)
            if hard and features['diagonal'] < 2:
                continue
        except (AssertionError, ValueError):
            continue
        return pair
    raise RuntimeError("bounded mosaic registration/packing exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(5))
    if regime < 4:
        ka, kb = ((2, 2), (3, 3), (2, 3), (3, 2))[regime]
        scales = [ka] * nt + [kb] * ns
    else:
        scales = [2, 3] + list(map(int, rng.choice([2, 3], size=nt+ns-2)))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            for _ in range(8):
                scale = scales[i if split == 'train' else nt+i]
                pair = _make_pair(rng, scale, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
            seen.add(key)
            result[split].append(pair)
    return result


def _pair_features(pair):
    data = _parse(pair['input'])
    answers, nodes = _reconstructions(data)
    assert len(answers) == 1 and -1 not in next(iter(answers))
    colors, chosen = next(iter(answers.items()))
    logical = _logical(data, colors)
    k = data['scale']
    assert np.array_equal(np.repeat(np.repeat(logical, k, 0), k, 1), pair['output'])
    assert np.array_equal(logical, logical[::-1]) and np.array_equal(logical, logical[:, ::-1])
    board = data['board']
    clues = (board != 0) & (board != data['structural'])
    assert clues.any() and np.all(logical[clues] == board[clues])
    anchored = {int(data['ids'][p]) for p in map(tuple, np.argwhere(clues))}
    unanchored = set(range(len(colors))) - anchored
    assert len(unanchored) >= 3 and len({colors[i] for i in unanchored}) >= 2
    assembled = board.copy()
    moved = False
    for hi, (pi, _, observed) in chosen:
        for p, v in zip(data['holes'][hi], observed):
            assembled[p] = v
        cells = data['holes'][hi]
        r0, r1, c0, c1 = _bounds(cells)
        target = np.zeros((r1-r0, c1-c0), dtype=int)
        for p, v in zip(cells, observed):
            target[p[0]-r0, p[1]-c0] = v
        moved |= _key(target) != _key(data['pieces'][pi])
    erased = assembled == 9
    assert erased.any() and np.all(logical[~erased] == assembled[~erased])
    diagonal = 0
    owners = {p: hi for hi, cells in enumerate(data['holes']) for p in cells}
    for oi, orb in enumerate(data['orbits']):
        visible = {int(assembled[p]) for p in orb if assembled[p] != 9}
        assert len(visible) == 1 and next(iter(visible)) == colors[oi]
        if (oi not in anchored and len(orb) == 4 and sum(assembled[p] == 9 for p in orb) == 3
                and len({owners[p] for p in orb}) > 1):
            diagonal += 1
    assert diagonal, 'both folds need an unanchored three-erasure witness'
    assert _shape_only_wrong(data, logical), 'shape/clue fitting must admit an incorrect arrangement'
    classes = set()
    for cells in data['holes']:
        r0, r1, c0, c1 = _bounds(cells)
        mask = np.zeros((r1-r0, c1-c0), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        classes.add(_class(mask))
    assert len(classes) >= 2
    return dict(scale=k, board_shape=tuple(board.shape), input_shape=tuple(np.asarray(pair['input']).shape),
                pieces=len(data['holes']), classes=tuple(sorted(classes)), clues=int(clues.sum()),
                erased=int(erased.sum()), diagonal=diagonal, moved=moved, nodes=nodes,
                structural=data['structural'], colors=len(set(colors)),
                geometry=(board.shape, (board != data['structural']).tobytes()))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                a = np.asarray(grid)
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            key = _key(pair['input'])
            assert key not in seen
            seen.add(key)
            features = _pair_features(pair)
            if split == 'test':
                assert features['diagonal'] >= 2, 'tests need two cross-tile hard orbits'


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test() -> None:
    """42 streamed seeds, geometric variation and concrete negative diagnostics."""
    hashes, shapes, layouts, classes, scales, counts, frames = set(), set(), set(), set(), set(), set(), set()
    moved = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_signature(puzzle))
        if seed == 0:
            first = puzzle
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _pair_features(pair)
                shapes.add(f['input_shape'])
                layouts.add(f['geometry'])
                classes.update(f['classes'])
                scales.add(f['scale'])
                counts.add(f['pieces'])
                frames.add(f['structural'])
                moved += f['moved']
    assert len(hashes) == 42 and len(shapes) >= 20 and len(layouts) >= 40
    assert len(classes) >= 15 and scales == {2, 3} and len(counts) >= 3 and len(frames) >= 6
    assert moved >= 30
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1) + 1):
            p = generate_puzzle(10000+10*nt+ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
            validate_generated_puzzle(p)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect frame output accepted')

