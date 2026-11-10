"""Dock the defects: periodic discrepancy masks become color-labelled pieces.

No external files, parent imports, caches, or generation state enter solve().
All generation and diagnostics are bounded and stay in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np


_N8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid colors")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _N8:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _trim(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty fault")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _canonical(mask):
    """Recover a downward centered two-cell terminal edge, modulo equivalence."""
    mask = _trim(np.asarray(mask, dtype=bool))
    _require(len(_components(mask)) == 1, "fault must be eight-connected")
    candidates = {}
    for turn in range(4):
        m = np.rot90(mask, turn).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        edge = np.zeros(w, dtype=bool)
        edge[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], edge):
            candidates.setdefault(_key(m), (m, []))[1].append(turn)
    _require(len(candidates) == 1, "missing or competing attachment edges")
    return next(iter(candidates.values()))


def _hub_positions(r, c):
    return (((r, c + 1), (r, c + 2)),
            ((r + 1, c + 3), (r + 2, c + 3)),
            ((r + 3, c + 1), (r + 3, c + 2)),
            ((r + 1, c), (r + 2, c)))


def _draw_hub(grid, r, c, labels):
    for color, points in zip(labels, _hub_positions(r, c)):
        for y, x in points:
            grid[y, x] = color


def _destination(mask, side, r, c):
    """Rotate a downward canonical mask and align its neck just outside a bar."""
    marker = np.zeros(mask.shape, dtype=bool)
    marker[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    oriented = np.rot90(mask, -side)
    neck = np.argwhere(np.rot90(marker, -side))
    outward = np.asarray(((-1, 0), (0, 1), (1, 0), (0, -1))[side])
    target = np.asarray(_hub_positions(r, c)[side]) + outward
    shift = target.min(axis=0) - neck.min(axis=0)
    return np.argwhere(oriented) + shift


def _recover(motif):
    """Unique nearest periodic repair; all phases have at least four copies.

    A period is supported only when every phase-coordinate has a strict
    majority. Equal best scores may use redundant periods, never different
    repaired arrays. The selected period is the shortest representative.
    """
    cross, length = motif.shape
    candidates = []
    for period in range(1, length // 4 + 1):
        template = np.empty((cross, period), dtype=np.int8)
        supported = True
        for phase in range(period):
            observations = motif[:, phase::period]
            counts = observations.sum(axis=1)
            if np.any(2 * counts == observations.shape[1]):
                supported = False
                break
            template[:, phase] = 2 * counts > observations.shape[1]
        if not supported:
            continue
        clean = template[:, np.arange(length) % period]
        # At least one row must genuinely vary along the repetition axis.
        if not np.any(clean[:, 1:] != clean[:, :-1]):
            continue
        cost = int(np.count_nonzero(clean != motif))
        if 1 <= cost <= motif.size // 4:
            candidates.append((cost, period, clean))
    _require(bool(candidates), "no supported sparse periodic repair")
    candidates.sort(key=lambda item: (item[0], item[1]))
    cost, period, clean = candidates[0]
    _require(all(np.array_equal(other, clean) for n, _, other in candidates if n == cost),
             "tied incompatible periodic repairs")
    return clean, period, cost


def _inspect(input_grid):
    grid = _array(input_grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "input must have a background border")
    rectangles, hubs = [], []
    for coords in _components(grid != bg):
        lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
        r, c = map(int, lo)
        h, w = map(int, hi - lo)
        block = grid[r:r + h, c:c + w]
        if (h, w) == (4, 4) and len(coords) == 8:
            labels = []
            check = np.full((4, 4), bg, dtype=np.int8)
            for points in _hub_positions(0, 0):
                values = [int(block[y, x]) for y, x in points]
                _require(values[0] == values[1] != bg, "invalid hub bars")
                labels.append(values[0])
            _draw_hub(check, 0, 0, labels)
            _require(len(set(labels)) == 4 and np.array_equal(check, block), "invalid hub")
            hubs.append((r, c, tuple(labels)))
        else:
            _require(len(coords) == h * w and min(h, w) >= 4, "invalid solid strip")
            frame = int(block[0, 0])
            _require(all(np.all(edge == frame) for edge in
                         (block[0], block[-1], block[:, 0], block[:, -1])), "broken frame")
            colors = set(map(int, np.unique(block)))
            _require(len(colors) == 2 and frame in colors, "strip must be binary")
            accent = next(iter(colors - {frame}))
            vertical = h > w
            motif = (block[1:-1, 1:-1].T if vertical else block[1:-1, 1:-1])
            _require(2 <= motif.shape[0] <= 6 and 12 <= motif.shape[1] <= 26,
                     "strip geometry outside family")
            repaired, period, cost = _recover((motif == accent).astype(np.int8))
            fault = _trim((motif == accent) != repaired)
            _require(len(_components(fault)) == 1, "fault must be eight-connected")
            fault_cells = ((motif == accent) != repaired)
            polarities = set(map(int, repaired[fault_cells]))
            _require(polarities == {0, 1}, "both error polarities are required")
            rectangles.append(dict(r=r, c=c, h=h, w=w, frame=frame, accent=accent,
                                   vertical=vertical, clean=repaired, fault=fault,
                                   period=period, cost=cost,
                                   fault_cells=fault_cells))
    _require(len(hubs) == 1 and 1 <= len(rectangles) <= 4, "one hub and one to four strips required")
    r, c, labels = hubs[0]
    frames = {m['frame'] for m in rectangles}
    accents = [m['accent'] for m in rectangles]
    _require(len(frames) == 1 and not frames.intersection(labels), "frame must differ from labels")
    _require(len(set(accents)) == len(accents), "strip accents must be distinct")
    answer = grid.copy()
    for m in rectangles:
        restored = np.where(m['clean'], m['accent'], m['frame'])
        answer[m['r'] + 1:m['r'] + m['h'] - 1,
               m['c'] + 1:m['c'] + m['w'] - 1] = restored.T if m['vertical'] else restored
    for m in rectangles:
        m['side'] = labels.index(m['accent']) if m['accent'] in labels else None
        physical_fault = m['fault'].T if m['vertical'] else m['fault']
        if m['side'] is None:
            try:
                m['canonical'], m['turns'] = _canonical(physical_fault)
            except ValueError:
                m['canonical'], m['turns'] = None, []
            continue
        # Fault was normalized to cross-by-length: convert its physical
        # orientation back before applying the silhouette attachment rule.
        canonical, turns = _canonical(physical_fault)
        coords = _destination(canonical, m['side'], r, c)
        rr, cc = coords.T
        _require(bool(np.all((0 <= rr) & (rr < grid.shape[0]) &
                             (0 <= cc) & (cc < grid.shape[1]))), "assembly would clip")
        _require(bool(np.all(answer[rr, cc] == bg)), "assembly would overlap")
        answer[rr, cc] = m['accent']
        m['physical_canonical'] = canonical
        m['canonical'], m['turns'] = canonical, turns
        m['required_turns'] = {(t - m['side']) % 4 for t in turns}
        m['target'] = coords
    return answer, dict(bg=bg, hub=(r, c), labels=labels, strips=rectangles)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from the input, without seed or stored examples."""
    return _inspect(input_grid)[0]


def _make_shape(rng, max_cross, input_turn, witness=None):
    """Compositional silhouettes, including small forms and diagonal wings."""
    directional = witness is not None
    for attempt in range(80):
        width = int(rng.choice((2, 4, 6)))
        depth = int(rng.integers(2, 6))
        kind = int(rng.integers(4))
        if witness is not None:
            width, depth = ((4, 2), (4, 3), (4, 4), (6, 3))[witness]
            kind = 1 if witness < 3 else 2
        center = width // 2
        mask = np.zeros((depth, width), dtype=bool)
        mask[:, center - 1:center + 1] = True
        stem = int(rng.integers(1, depth))
        if kind == 0:
            width = 2
            mask = np.ones((depth, 2), dtype=bool)
        elif kind == 1:
            mask[:depth - stem] = True
        elif kind == 2 and width > 2:
            mask[:] = False
            reach = min(center - 1, depth - 1)
            mask[reach:, center - 1:center + 1] = True
            for row in range(reach):
                d = reach - row
                mask[row, center - 1 - d] = True
                mask[row, center + d] = True
            mask = _trim(mask)
        else:
            for row in range(depth - stem):
                left = int(rng.integers(0, center))
                right = int(rng.integers(center + 1, width + 1))
                mask[row, left:right] = True
            mask[0, :center] = True
            mask[min(1, depth - stem - 1), center:] = True
            mask = _trim(mask)
        try:
            canonical, turns = _canonical(mask)
        except ValueError:
            if witness is not None:
                witness = None
            continue
        oriented = np.rot90(canonical, input_turn)
        if (oriented.shape[0] <= max_cross and canonical.sum() <= 18
                and (not directional or len(turns) == 1)):
            return canonical, oriented
        if witness is not None:
            witness = None
    # A four-cell-wide T always fits a cross section of four, in every turn.
    mask = np.asarray(((1, 1, 1, 1), (0, 1, 1, 0)), dtype=bool)
    return mask, np.rot90(mask, input_turn)


def _cut_layout(rng, crosses, height, width):
    """Pack strips by cuts around a reserved core; parsing never uses this."""
    order = list(map(int, rng.permutation(len(crosses))))

    def search(box, index):
        if index == len(order):
            return [], box
        r, c, h, w = box
        i = order[index]
        thick = crosses[i] + 2
        for side in map(int, rng.permutation(4)):
            horizontal = side in (0, 2)
            if (h - thick - 1 < 14 if horizontal else w - thick - 1 < 14):
                continue
            span = w if horizontal else h
            maximum = min(26, span - 2)
            if maximum < 12:
                continue
            length = int(rng.integers(12, maximum + 1))
            offset = int(rng.integers(0, span - length - 1))
            if side == 0:
                rect, nxt = (r, c + offset, thick, length + 2), (r + thick + 1, c, h - thick - 1, w)
            elif side == 2:
                rect, nxt = (r + h - thick, c + offset, thick, length + 2), (r, c, h - thick - 1, w)
            elif side == 3:
                rect, nxt = (r + offset, c, length + 2, thick), (r, c + thick + 1, h, w - thick - 1)
            else:
                rect, nxt = (r + offset, c + w - thick, length + 2, thick), (r, c, h, w - thick - 1)
            rest = search(nxt, index + 1)
            if rest is not None:
                placements, core = rest
                return [(i, rect, not horizontal, length)] + placements, core
        return None

    return search((1, 1, height - 2, width - 2), 0)


def _texture(rng, cross, length, fault, forced_period=None, edge=False):
    """Sample and independently recover a noisy texture with a given fault."""
    periods = list(range(2, min(5, length // 4) + 1))
    if forced_period in periods:
        periods = [forced_period]
    for attempt in range(240):
        period = int(rng.choice(periods))
        word = rng.integers(0, 2, size=(cross, period), dtype=np.int8)
        clean = word[:, np.arange(length) % period]
        fh, fw = fault.shape
        y = int(rng.integers(cross - fh + 1))
        x = (int(rng.choice((0, length - fw))) if edge else
             int(rng.integers(length - fw + 1)))
        mask = np.zeros((cross, length), dtype=bool)
        mask[y:y + fh, x:x + fw] = fault
        if set(map(int, clean[mask])) != {0, 1}:
            continue
        noisy = clean.copy()
        noisy[mask] = 1 - noisy[mask]
        try:
            recovered, p, cost = _recover(noisy)
        except ValueError:
            continue
        if np.array_equal(recovered, clean) and p == period and cost == int(mask.sum()):
            return noisy, clean
    return None


def _build_pair(rng, palette, config, test):
    n = len(config['sides'])
    bg, frame = palette[:2]
    labels = tuple(map(int, rng.permutation(palette[2:6])))
    for attempt in range(100):
        absent = list(map(int, rng.permutation(palette[6:])))
        shapes, faults, crosses, accents = [], [], [], []
        for i, side in enumerate(config['sides']):
            # The strip's eventual orientation is selected later. Delta
            # witnesses are enforced after that orientation becomes visible.
            turn = int(rng.integers(4))
            canonical, fault = _make_shape(rng, 4 if n >= 3 else 6, turn,
                                           config.get('witness', {}).get(i))
            if side is None and i not in config.get('dockable', ()) and rng.random() < .4:
                # Connected but not necessarily dockable: filtering precedes
                # attachment recognition, just as color correspondence requires.
                size = int(rng.integers(2, 5))
                fault = np.zeros((size, size), dtype=bool)
                fault[0, 0] = True
                for _ in range(int(rng.integers(3, 10))):
                    options = sorted({(int(r + dr), int(c + dc))
                                      for r, c in np.argwhere(fault) for dr, dc in _N8
                                      if 0 <= r + dr < size and 0 <= c + dc < size})
                    point = options[int(rng.integers(len(options)))]
                    fault[point] = True
                fault = _trim(fault)
            cross = int(rng.integers(max(2, fault.shape[0]), (4 if n >= 3 else 6) + 1))
            shapes.append(canonical)
            faults.append(fault)
            crosses.append(cross)
            accents.append(labels[side] if side is not None else absent.pop() if len(absent) else palette[6])
        height, width = map(int, rng.integers(24, 31, size=2))
        layout = _cut_layout(rng, crosses, height, width)
        if layout is None:
            height = width = 30
            layout = _cut_layout(rng, crosses, height, width)
        if layout is None:
            continue
        placements, core = layout
        grid = np.full((height, width), bg, dtype=np.int8)
        expected = grid.copy()
        ok = True
        for i, (r, c, h, w), vertical, length in placements:
            side, delta = config['sides'][i], config['deltas'][i]
            if delta is not None:
                physical = np.rot90(shapes[i], (-side - delta) % 4)
                fault = physical.T if vertical else physical
                if fault.shape[0] > crosses[i]:
                    ok = False
                    break
                faults[i] = fault
            result = _texture(rng, crosses[i], length, faults[i],
                              config.get('periods', {}).get(i), edge=test and rng.random() < .8)
            if result is None:
                ok = False
                break
            noisy, clean = result
            for canvas, texture in ((grid, noisy), (expected, clean)):
                canvas[r:r + h, c:c + w] = frame
                patch = np.where(texture, accents[i], frame)
                canvas[r + 1:r + h - 1, c + 1:c + w - 1] = patch.T if vertical else patch
        if not ok:
            absent = list(map(int, rng.permutation(palette[6:])))
            continue
        # Assembly size and overlaps depend on the recovered physical masks.
        targets = []
        occupied = set()
        for i, rect, vertical, length in placements:
            side = config['sides'][i]
            if side is None:
                continue
            physical = faults[i].T if vertical else faults[i]
            canonical, _ = _canonical(physical)
            coords = _destination(canonical, side, 0, 0)
            cells = set(map(tuple, coords))
            if occupied.intersection(cells):
                ok = False
                break
            occupied.update(cells)
            targets.append((i, coords))
        if not ok:
            continue
        all_points = np.concatenate([np.asarray([p for bar in _hub_positions(0, 0) for p in bar])]
                                    + [coords for _, coords in targets])
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        cr, cc, ch, cw = core
        lower = np.asarray((cr, cc)) - lo
        upper = np.asarray((cr + ch - 1, cc + cw - 1)) - hi
        if np.any(lower > upper):
            continue
        hub = tuple(int(rng.integers(a, b + 1)) for a, b in zip(lower, upper))
        _draw_hub(grid, *hub, labels)
        _draw_hub(expected, *hub, labels)
        for i, coords in targets:
            rr, cols = (coords + np.asarray(hub)).T
            expected[rr, cols] = accents[i]
        try:
            answer, _ = _inspect(grid)
        except ValueError:
            continue
        if np.array_equal(answer, expected):
            return {'input': grid, 'output': expected}
    raise RuntimeError("bounded construction failed")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    order = list(map(int, rng.permutation(4)))
    deltas = list(map(int, rng.permutation(4)))
    configs = []
    # Complementary witness pairs, shuffled below, work even with two trains.
    configs.append(dict(sides=order[:2], deltas=deltas[:2], periods={0: 2}, witness={0: 0, 1: 1}))
    configs.append(dict(sides=order[2:] + [None], deltas=deltas[2:] + [None],
                        periods={0: 3}, witness={0: 2, 1: 3}, dockable={2}))
    for _ in range(nt - 2):
        n = int(rng.integers(1, 4))
        sides = list(map(int, rng.choice(4, n, replace=False)))
        if n >= 2 and rng.random() < .5:
            sides[-1] = None
        configs.append(dict(sides=sides, deltas=[None] * n))
    rng.shuffle(configs)
    for _ in range(nq):
        n = int(rng.choice((1, 2, 3, 4), p=(.3, .15, .15, .4)))
        sides = list(map(int, rng.choice(4, n, replace=False)))
        if n > 1 and rng.random() < .5:
            sides[-1] = None
        configs.append(dict(sides=sides, deltas=[None] * n))
    puzzle, seen = {'train': [], 'test': []}, set()
    for serial, config in enumerate(configs):
        split = 'train' if serial < nt else 'test'
        pair_palette = list(np.roll(palette, serial % 10))
        for _ in range(8):
            pair = _build_pair(rng, pair_palette, config, split == 'test')
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('distinct-input construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "invalid splits")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 2, 10, 'num_train')
    _count(nq, 1, 4, 'num_test')
    _require(nt > nq, 'illegal count relation')
    seen, directions, rotations, periods = set(), set(), set(), set()
    filtering = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _array(pair['input']), _array(pair['output'])
            _require(inp.shape == out.shape, 'shape changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            answer, info = _inspect(inp)
            _require(np.array_equal(out, answer), 'incorrect output')
            _require(not np.array_equal(inp, out), 'identity pair')
            added = int(np.count_nonzero((inp == info['bg']) & (out != info['bg'])))
            matched_faults = sum(m['cost'] for m in info['strips'] if m['side'] is not None)
            all_faults = sum(m['cost'] for m in info['strips'])
            _require(added == matched_faults > 0, 'A-only repair must omit real answer cells')
            _require(int(np.count_nonzero(inp != out)) == all_faults + matched_faults,
                     'only repairs and selected silhouettes may change')
            hub_cells = [p for bar in _hub_positions(*info['hub']) for p in bar]
            for r, c in hub_cells:
                _require(inp[r, c] == out[r, c], 'hub changed')
            for m in info['strips']:
                physical = m['fault'].T if m['vertical'] else m['fault']
                _require(int(physical.sum()) == m['cost'], 'fault cells lost')
                # Both polarities imply that visible accent pixels neither
                # contain all fault cells nor coincide with the hidden piece.
                _require(set(map(int, m['clean'][m['fault_cells']])) == {0, 1},
                         'ignoring periodic prediction must change object geometry')
                if m['side'] is not None:
                    _require(len(m['target']) == m['cost'], 'silhouette not preserved')
                if split == 'train':
                    periods.add(m['period'])
                    if m['side'] is None:
                        filtering |= m['canonical'] is not None
                    else:
                        directions.add(m['side'])
                        # Symmetric equivalences do not establish a turn witness.
                        if len(m['required_turns']) == 1:
                            rotations.update(m['required_turns'])
    _require(directions == set(range(4)), 'training must cover every side')
    _require(rotations == set(range(4)), 'training must cover every turn')
    _require(filtering, 'training must demonstrate filtering')
    _require({2, 3} <= periods, 'training must distinguish more than alternation')


def self_test():
    """42 streaming seeds; no disk writes or retained puzzle corpus."""
    assert len(_canonical(np.ones((2, 2), dtype=bool))[1]) == 4
    assert len(_canonical(np.ones((4, 2), dtype=bool))[1]) == 2
    competing = np.asarray(((0, 1, 1, 1), (1, 1, 1, 1),
                            (1, 0, 1, 1), (0, 1, 1, 0)), dtype=bool)
    try:
        _canonical(competing)
    except ValueError:
        pass
    else:
        raise AssertionError('competing attachments accepted')
    shapes, periods, crosses, counts, silhouettes, layouts, signatures = (set() for _ in range(7))
    diagonal_count = asymmetric_count = boundary_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                grid = pair['input']
                digest.update(bytes(grid.shape) + grid.tobytes() + pair['output'].tobytes())
                shapes.add(grid.shape)
                _, info = _inspect(grid)
                counts.add(len(info['strips']))
                layouts.add(tuple(sorted(m['vertical'] for m in info['strips'])))
                for m in info['strips']:
                    periods.add(m['period'])
                    crosses.add(m['clean'].shape[0])
                    canonical = m['canonical'] if m['canonical'] is not None else m['fault']
                    if m['canonical'] is not None:
                        silhouettes.add(_key(canonical))
                    asymmetric_count += not np.array_equal(canonical, np.fliplr(canonical))
                    # Four-connectivity witness, contrasted with eight-connectivity.
                    cells = set(map(tuple, np.argwhere(canonical)))
                    pending = [cells.pop()]
                    while pending:
                        r, c = pending.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            if (r + dr, c + dc) in cells:
                                cells.remove((r + dr, c + dc))
                                pending.append((r + dr, c + dc))
                    diagonal_count += bool(cells)
                    f = m['fault_cells']
                    boundary_count += bool(f[:, 0].any() or f[:, -1].any())
        signatures.add(digest.digest())
    assert len(signatures) == 42
    assert len(shapes) >= 12 and counts == {1, 2, 3, 4}
    assert {2, 3, 4, 5} <= periods and {2, 3, 4, 5, 6} <= crosses
    assert len(silhouettes) >= 20 and len(layouts) >= 6
    assert diagonal_count >= 10 and asymmetric_count >= 5 and boundary_count >= 40
    first = generate_puzzle(903, 2, 1)
    second = generate_puzzle(903, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for split in first for a, b in
               zip(first[split], second[split]) for k in ('input', 'output'))
    # Input-only solve is insensitive to generation history and does not mutate input.
    query = first['test'][0]['input'].copy()
    old = query.copy()
    assert np.array_equal(solve(query), first['test'][0]['output']) and np.array_equal(query, old)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(4100 + 10 * nt + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][0, 0] = (int(query[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), canvas_shapes=len(shapes),
                periods=sorted(periods), cross_sections=sorted(crosses), strip_counts=sorted(counts),
                canonical_silhouettes=len(silhouettes), orientation_layouts=len(layouts),
                diagonal_faults=diagonal_count, asymmetric_faults=asymmetric_count,
                boundary_faults=boundary_count)
