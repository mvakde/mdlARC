"""Conserved fragment repair constrained by a partly erased periodic weave.

All answer conventions are encoded in the input border. No parent modules,
stored puzzles, labels, seeds, or learned palette are used by solve().
"""
from collections import Counter
from functools import lru_cache
from itertools import product

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, component = [first], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in todo:
                    todo.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(component)))
    return result


def _fragment(grid, cells):
    top = min(r for r, c in cells)
    left = min(c for r, c in cells)
    return tuple(sorted((r - top, c - left, int(grid[r, c])) for r, c in cells))


def _ticks(values, frame):
    colors = set(map(int, values)) - {frame}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    indices = np.flatnonzero(values == color)
    if len(indices) < 4 or indices[0] != 0:
        return None
    period = int(indices[1])
    if not 2 <= period <= 4 or len(values) % period:
        return None
    if not np.array_equal(indices, np.arange(0, len(values), period)):
        return None
    return color, period


def _parse(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > inp.size // 2, "Background must be a strict majority")
    candidates = []
    for f in map(int, colors):
        if f == bg:
            continue
        rr, cc = np.where(inp == f)
        r, c, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
        h, w = bottom - r - 1, right - c - 1
        if min(h, w) < 8:
            continue
        box = inp[r:bottom + 1, c:right + 1]
        carrier, accent = int(box[0, 0]), int(box[-1, -1])
        if len({bg, f, carrier, accent}) != 4 or box[0, -1] != f or box[-1, 0] != f:
            continue
        if not (np.all(box[-1, :-1] == f) and np.all(box[1:, -1][:-1] == f)):
            continue
        row_ticks, col_ticks = _ticks(box[1:-1, 0], f), _ticks(box[0, 1:-1], f)
        if row_ticks is None or col_ticks is None or row_ticks[0] != col_ticks[0]:
            continue
        highlight, p, q = row_ticks[0], row_ticks[1], col_ticks[1]
        if len({bg, f, carrier, accent, highlight}) != 5:
            continue
        if h // p not in (4, 6) or w // q not in (4, 6):
            continue
        board = box[1:-1, 1:-1].copy()
        if not set(map(int, np.unique(board))) <= {bg, carrier, accent}:
            continue
        if np.any(board == f):
            continue
        outside = inp != bg
        outside[r:bottom + 1, c:right + 1] = False
        if set(map(int, inp[outside])) - {carrier, accent}:
            continue
        cells = {tuple(map(int, xy)) for xy in np.argwhere(outside)}
        if any((r <= y <= bottom and c - 1 <= x <= right + 1)
               or (r - 1 <= y <= bottom + 1 and c <= x <= right) for y, x in cells):
            continue
        components = _components(cells)
        pieces = [_fragment(inp, comp) for comp in components]
        if not pieces or any(len(piece) > 8 for piece in pieces):
            continue
        holes = board == bg
        if int(holes.sum()) != sum(map(len, pieces)):
            continue
        known = {}
        consistent = True
        for y, x in np.argwhere(~holes):
            phase, value = (int(y) % p, int(x) % q), int(board[y, x])
            if phase in known and known[phase] != value:
                consistent = False
                break
            known[phase] = value
        if not consistent:
            continue
        unseen = sorted(set(product(range(p), range(q))) - set(known))
        if len(unseen) != 2 or set(known.values()) != {carrier, accent}:
            continue
        candidates.append(dict(inp=inp, bg=bg, frame=f, carrier=carrier, accent=accent,
                               highlight=highlight, bbox=(r, c, h + 2, w + 2),
                               board=board, holes=holes, periods=(p, q), known=known,
                               unseen=unseen, pieces=pieces, components=components))
    _require(len(candidates) == 1, "Expected one measuring border with a valid inventory")
    return candidates[0]


class _SearchLimit(Exception):
    pass


def _cover(target, holes, pieces, node_limit=None):
    """Exact colored cover; identical fragments are counted, never permuted.

    Singleton fragments need no search: after placing every larger fragment,
    color conservation gives their exact destinations. Return one physical
    witness. Different witnesses for the same colored board are equivalent.
    """
    coordinates = [tuple(map(int, xy)) for xy in np.argwhere(holes)]
    ids = {cell: i for i, cell in enumerate(coordinates)}
    inventory = Counter(pieces)
    target_counts = Counter(int(target[cell]) for cell in coordinates)
    supplied = Counter(color for piece in pieces for _, _, color in piece)
    if supplied != target_counts:
        return None, 0
    singles = Counter({piece[0][2]: n for piece, n in inventory.items() if len(piece) == 1})
    types = sorted(piece for piece in inventory if len(piece) > 1)
    counts = tuple(inventory[t] for t in types)
    moves = []
    h, w = target.shape
    for piece in types:
        sh = max(dr for dr, dc, color in piece) + 1
        sw = max(dc for dr, dc, color in piece) + 1
        choices = []
        for r in range(h - sh + 1):
            for c in range(w - sw + 1):
                if all((r + dr, c + dc) in ids and target[r + dr, c + dc] == color
                       for dr, dc, color in piece):
                    mask = sum(1 << ids[r + dr, c + dc] for dr, dc, color in piece)
                    choices.append((mask, r, c))
        if len(choices) < inventory[piece]:
            return None, 0
        moves.append(choices)
    nodes = 0

    @lru_cache(maxsize=None)
    def search(used, remaining):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("Colored-cover proof exceeded its generation budget")
        if not any(remaining):
            return ()
        best = None
        for t, number in enumerate(remaining):
            if not number:
                continue
            choices = [move for move in moves[t] if not move[0] & used]
            if len(choices) < number:
                return None
            score = len(choices) / number
            if best is None or score < best[0]:
                best = (score, t, choices)
        _, t, choices = best
        rest = list(remaining)
        rest[t] -= 1
        for mask, r, c in choices:
            suffix = search(used | mask, tuple(rest))
            if suffix is not None:
                return ((types[t], r, c),) + suffix
        return None

    witness = search(0, counts)
    if witness is None:
        return None, nodes
    used_cells = {(r + dr, c + dc) for piece, r, c in witness for dr, dc, color in piece}
    remainder = [cell for cell in coordinates if cell not in used_cells]
    _require(Counter(int(target[cell]) for cell in remainder) == singles, "Singleton accounting failed")
    witness += tuple((((0, 0, int(target[r, c])),), r, c) for r, c in remainder)
    return witness, nodes


def _repairs(info, node_limit=None):
    p, q = info['periods']
    h, w = info['board'].shape
    answers, count_compatible = [], []
    supplied = Counter(color for piece in info['pieces'] for _, _, color in piece)
    for values in product((info['carrier'], info['accent']), repeat=2):
        motif = np.empty((p, q), dtype=np.int8)
        for phase, color in info['known'].items():
            motif[phase] = color
        for phase, color in zip(info['unseen'], values):
            motif[phase] = color
        target = np.tile(motif, (h // p, w // q))
        if Counter(map(int, target[info['holes']])) != supplied:
            continue
        count_compatible.append(target)
        witness, nodes = _cover(target, info['holes'], info['pieces'], node_limit)
        if witness is not None:
            answers.append((target, motif, witness, nodes))
    return answers, count_compatible


def _central(shape, periods):
    h, w = shape
    p, q = periods
    central = np.zeros(shape, dtype=bool)
    r, c = (h - 2 * p) // 2, (w - 2 * q) // 2
    central[r:r + 2 * p, c:c + 2 * q] = True
    return central


def _finish(base, periods, accent, highlight):
    out = base.copy()
    out[_central(base.shape, periods) & (base == accent)] = highlight
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the border, fixed cells, and exact visible fragment inventory."""
    info = _parse(input_grid)
    answers, _ = _repairs(info)
    _require(len(answers) == 1, "There must be exactly one periodic colored repair")
    return _finish(answers[0][0], info['periods'], info['accent'], info['highlight'])


def _motif(rng, periods, density, asymmetric):
    p, q = periods
    for _ in range(100):
        a = np.zeros((p, q), dtype=bool)
        a.flat[rng.choice(p * q, density, replace=False)] = True
        if asymmetric and (np.array_equal(a, a[::-1]) or np.array_equal(a, a[:, ::-1])
                           or (p == q and np.array_equal(a, a.T))):
            continue
        return a
    raise AssertionError("Could not sample an asymmetric motif")


def _cut(rng, base, motif, periods, bg, carrier, accent, extra):
    p, q = periods
    h, w = base.shape
    a = tuple(map(int, np.argwhere(motif)[int(rng.integers(int(motif.sum())))]))
    b = tuple(map(int, np.argwhere(~motif)[int(rng.integers(int((~motif).sum())))]))
    yy, xx = np.indices(base.shape)
    holes = ((yy % p == a[0]) & (xx % q == a[1])) | ((yy % p == b[0]) & (xx % q == b[1]))
    # Leave two actual, oppositely colored cells as singleton pieces. Their
    # exchange is a constructive inventory-only counterexample in every pair.
    reserved = set()
    for phase in (a, b):
        y, x = int(rng.integers(h // p)), int(rng.integers(w // q))
        reserved.add((y * p + phase[0], x * q + phase[1]))
    left = Counter((int(y) % p, int(x) % q) for y, x in np.argwhere(~holes))
    for _ in range(extra):
        eligible = [tuple(map(int, xy)) for xy in np.argwhere(~holes)
                    if left[int(xy[0]) % p, int(xy[1]) % q] > 2]
        neighbors = [cell for cell in eligible if any(0 <= cell[0] + dy < h and 0 <= cell[1] + dx < w
                     and holes[cell[0] + dy, cell[1] + dx] for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))]
        choices = neighbors if neighbors and rng.random() < 0.85 else eligible
        if not choices:
            break
        y, x = choices[int(rng.integers(len(choices)))]
        holes[y, x] = True
        left[y % p, x % q] -= 1
    remaining = {tuple(map(int, xy)) for xy in np.argwhere(holes)} - reserved
    groups = [(cell,) for cell in sorted(reserved)]
    while remaining:
        ordered = sorted(remaining)
        first = ordered[int(rng.integers(len(ordered)))]
        cells = {first}
        remaining.remove(first)
        target_size = int(rng.integers(2, 9))
        while len(cells) < target_size:
            frontier = sorted({(y + dy, x + dx) for y, x in cells
                               for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))} & remaining)
            if not frontier:
                break
            cell = frontier[int(rng.integers(len(frontier)))]
            remaining.remove(cell)
            cells.add(cell)
        groups.append(tuple(sorted(cells)))
    pieces = [_fragment(base, cells) for cells in groups]
    damaged = base.copy()
    damaged[holes] = bg
    return damaged, pieces


def _border(board, roles, periods):
    bg, frame, carrier, accent, highlight = roles
    p, q = periods
    h, w = board.shape
    box = np.full((h + 2, w + 2), frame, dtype=np.int8)
    box[1:-1, 1:-1] = board
    box[0, 0], box[-1, -1] = carrier, accent
    box[0, 1:-1:q] = highlight
    box[1:-1:p, 0] = highlight
    return box


def _scatter(rng, box, pieces, bg):
    bh, bw = box.shape
    occupied_area = box.size  # Cut material is present exactly once, outside.
    dimensions = [(h, w) for h in range(max(bh + 2, 18), 31)
                  for w in range(max(bw + 2, 18), 31) if h * w > 2 * occupied_area]
    if not dimensions:
        return None
    for attempt in range(12):
        h, w = dimensions[int(rng.integers(len(dimensions)))] if attempt < 8 else (30, 30)
        r, c = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        canvas = np.full((h, w), bg, dtype=np.int8)
        canvas[r:r + bh, c:c + bw] = box
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, r - 1):min(h, r + bh + 1), max(0, c - 1):min(w, c + bw + 1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), float(rng.random())))
        for i in order:
            piece = pieces[i]
            ph, pw = max(y for y, x, color in piece) + 1, max(x for y, x, color in piece) + 1
            positions = [(y, x) for y in range(h - ph + 1) for x in range(w - pw + 1)
                         if all(not blocked[y + dy, x + dx] for dy, dx, color in piece)]
            if not positions:
                break
            y, x = positions[int(rng.integers(len(positions)))]
            for dy, dx, color in piece:
                ny, nx = y + dy, x + dx
                canvas[ny, nx] = color
                for by, bx in ((ny, nx), (ny - 1, nx), (ny + 1, nx), (ny, nx - 1), (ny, nx + 1)):
                    if 0 <= by < h and 0 <= bx < w:
                        blocked[by, bx] = True
        else:
            return canvas
    return None


def _pair(rng, periods, repeats, roles, density, asymmetric):
    bg, frame, carrier, accent, highlight = roles
    for _ in range(100):
        mask = _motif(rng, periods, density, asymmetric)
        motif = np.where(mask, accent, carrier).astype(np.int8)
        base = np.tile(motif, repeats)
        extra = int(rng.integers(6, 17 if repeats == (4, 4) else 23))
        damaged, pieces = _cut(rng, base, mask, periods, bg, carrier, accent, extra)
        if not any(len({color for _, _, color in piece}) == 2 for piece in pieces):
            continue
        canvas = _scatter(rng, _border(damaged, roles, periods), pieces, bg)
        if canvas is None:
            continue
        try:
            info = _parse(canvas)
            answers, compatible = _repairs(info, node_limit=18000)
        except (AssertionError, _SearchLimit):
            continue
        if len(compatible) != 2 or len(answers) != 1 or not np.array_equal(answers[0][0], base):
            continue
        return {'input': canvas, 'output': _finish(base, periods, accent, highlight)}
    raise AssertionError("Bounded construction failed to find a uniquely coupled repair")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible puzzles; only sampling preferences are puzzle-level state."""
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    period_choices = [(p, q) for p in range(2, 5) for q in range(2, 5)
                      if p * q >= 6 and min(p, q) <= 3]
    periods = period_choices[int(rng.integers(len(period_choices)))]
    palette = tuple(map(int, rng.permutation(10)[:5]))
    exchange = bool(rng.random() < 0.6)
    densities = list(range(2, periods[0] * periods[1] - 1))
    first_two = list(map(int, rng.choice(densities, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            density = first_two[index] if split == 'train' and index < 2 else int(rng.choice(densities))
            roles = tuple(map(int, rng.permutation(10)[:5])) if exchange else palette
            if split == 'train':
                repeats = (4, 4)
            else:
                axes = [axis for axis in (0, 1) if periods[axis] <= 3]
                axis = axes[int(rng.integers(len(axes)))]
                repeats = (6, 4) if axis == 0 else (4, 6)
            for _ in range(8):
                pair = _pair(rng, periods, repeats, roles, density, split == 'train' and index == 0)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("Could not sample a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    info = _parse(inp)
    answers, compatible = _repairs(info)
    _require(len(answers) == 1 and len(compatible) == 2, "Both mechanisms must resolve an ambiguity")
    base, motif, witness, nodes = answers[0]
    _require(np.array_equal(out, _finish(base, info['periods'], info['accent'], info['highlight'])),
             "Wrong repaired or highlighted cloth")
    _require(Counter(piece for piece, r, c in witness) == Counter(info['pieces']), "Inventory was not used once")
    covered = [(r + dy, c + dx) for piece, r, c in witness for dy, dx, color in piece]
    _require(len(set(covered)) == len(covered) == int(info['holes'].sum()), "Pieces overlap or leave a gap")
    _require(np.array_equal(base[~info['holes']], info['board'][~info['holes']]), "Fixed cloth changed in repair")
    _require(set(map(int, motif.ravel())) == {info['carrier'], info['accent']}, "Both motif roles are needed")
    singletons = {piece[0][2]: (r, c) for piece, r, c in witness if len(piece) == 1}
    _require(set(singletons) == {info['carrier'], info['accent']}, "Missing inventory-only swap witness")
    swapped = base.copy()
    x, y = singletons[info['carrier']], singletons[info['accent']]
    swapped[x], swapped[y] = base[y], base[x]
    p, q = info['periods']
    _require(not np.array_equal(swapped, np.tile(swapped[:p, :q], (base.shape[0] // p, base.shape[1] // q))),
             "Singleton exchange did not break periodicity")
    _require(not np.array_equal(_finish(swapped, (p, q), info['accent'], info['highlight']), out),
             "Ignoring B did not change the answer")
    alternative = next(t for t in compatible if not np.array_equal(t, base))
    _require(np.array_equal(alternative[~info['holes']], base[~info['holes']]), "Bad periodic-only witness")
    _require(not np.array_equal(_finish(alternative, (p, q), info['accent'], info['highlight']), out),
             "Ignoring A did not change the answer")
    central = _central(base.shape, (p, q))
    _require(np.array_equal(out == info['carrier'], base == info['carrier']), "Carrier was overwritten")
    _require(np.array_equal(out == info['highlight'], central & (base == info['accent'])), "Highlight is a fixed stencil")
    _require(int((out == info['highlight']).sum()) == 4 * int((motif == info['accent']).sum()), "Wrong central copy count")
    mixed = sum(len({color for _, _, color in piece}) == 2 for piece in info['pieces'])
    _require(mixed > 0, "No fragment carries both roles")
    return dict(periods=(p, q), board_shape=base.shape, canvas_shape=inp.shape,
                motif=(motif == info['accent']).tobytes(), density=int((motif == info['accent']).sum()),
                holes=info['holes'].tobytes(), piece_count=len(info['pieces']), mixed=mixed,
                signatures={tuple((r, c) for r, c, color in piece) for piece in info['pieces']},
                palette=(info['bg'], info['frame'], info['carrier'], info['accent'], info['highlight']), nodes=nodes)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Invalid example counts")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            key = _key(_grid(pair['input']))
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            analyses[split].append(_analyze(pair))
    periods = {a['periods'] for s in analyses for a in analyses[s]}
    _require(len(periods) == 1, "Generated puzzle changes its sampled period preference")
    p, q = next(iter(periods))
    _require(len({a['density'] for a in analyses['train']}) >= 2, "Training lacks distinct highlight footprints")
    _require(all(a['board_shape'] == (4 * p, 4 * q) for a in analyses['train']), "Unexpected training scale")
    _require(all(a['board_shape'] in ((6 * p, 4 * q), (4 * p, 6 * q)) for a in analyses['test']),
             "Missing repetition-count train/test shift")


def _fingerprint(puzzle):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural, semantic, ablation and corruption checks."""
    periods, shapes, motifs, cuts, signatures, palettes, fingerprints, counts = (set() for _ in range(8))
    maximum_nodes = 0
    for seed in range(42):
        # Rotate legal counts, in addition to the host's full explicit-count checks.
        ne = 1 + seed % 4
        nt = ne + 1 + (seed // 4) % (10 - ne)
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        counts.add((nt, ne))
        local_motifs, local_cuts = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair)
                _require(np.array_equal(solve(pair['input']), pair['output']), "Input-only solve mismatch")
                periods.add(a['periods'])
                shapes.add(a['board_shape'])
                motifs.add((a['periods'], a['motif']))
                cuts.add((a['board_shape'], a['holes']))
                local_motifs.add(a['motif'])
                local_cuts.add(a['holes'])
                signatures.update(a['signatures'])
                palettes.add(a['palette'])
                maximum_nodes = max(maximum_nodes, a['nodes'])
        _require(len(local_motifs) >= 2 and len(local_cuts) >= 2, "Only palette/position varies")
    _require(len(fingerprints) == 42, "Seeds repeat whole puzzles")
    _require(len(periods) >= 5 and len(shapes) >= 10, "Periods and board dimensions lack variation")
    _require(len(motifs) >= 80 and len(cuts) >= 200, "Motifs and cut geometry lack variation")
    _require(len(signatures) >= 25 and any(len(s) >= 5 for s in signatures), "Fragments lack shape diversity")
    _require(len(palettes) >= 30 and len(counts) >= 15, "Palettes/counts lack variation")
    first = generate_puzzle(123456, 3, 2)
    generate_puzzle(654321, 2, 1)
    _require(_fingerprint(first) == _fingerprint(generate_puzzle(123456, 3, 2)), "Seed depends on intervening calls")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    # Two semantic negative checks: carrier overwrite and lost inventory.
    for kind in ('carrier', 'inventory'):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in first[s]] for s in first}
        pair = bad['test'][0]
        info = _parse(pair['input'])
        if kind == 'carrier':
            cell = tuple(map(int, np.argwhere(pair['output'] == info['carrier'])[0]))
            pair['output'][cell] = info['highlight']
        else:
            pair['input'][info['components'][0][0]] = info['bg']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Validator accepted " + kind + " corruption")
    return dict(seeds=42, puzzles=len(fingerprints), motifs=len(motifs), cut_layouts=len(cuts),
                oriented_shapes=len(signatures), periods=len(periods), board_shapes=len(shapes),
                palettes=len(palettes), maximum_successful_cover_nodes=maximum_nodes)


if __name__ == '__main__':
    print(self_test())
