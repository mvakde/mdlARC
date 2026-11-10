"""Complete marker-selected walls, then paint the union of shortest gate routes.

Only standard-library modules and NumPy are used. No imports perform I/O.
Background=0, symmetry marker=5, chamber shell=8. The two terminals visibly
supply the route color. All other foreground cells are solid obstacles.
Generation and tests retain no files, examples, seed-dependent solver state,
or execution traces. See task_spec.md for the supported input language.
"""
from collections import deque
import hashlib
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def comps(cells):
    todo = set(cells)
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, part = [root], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in todo:
                        todo.remove(q)
                        stack.append(q)
        result.append(sorted(part))
    return result


def dist(blocked, start):
    h, w = blocked.shape
    d = np.full((h, w), -1, np.int16)
    d[start] = 0
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1
                todo.append((rr, cc))
    return d


def routes(board, centers):
    blocked = board != 0
    for p in centers:
        blocked[p] = False
    a, b = (dist(blocked, p) for p in centers)
    length = int(a[centers[1]])
    return ((a >= 0) & (b >= 0) & (a + b == length) & (board == 0), length)


def proto(rng, kind, cap=4, witness=True):
    h, w = int(rng.integers(2, cap+1)), int(rng.integers(2, cap+2))
    style = int(rng.integers(5))
    cells = {(r, c) for r in range(h) for c in range(w)}
    if style == 1:
        cells = {(r, c) for r in range(h) for c in range(w) if r == 0 or c == 0}
    elif style == 2:
        # Bounce at the edge rather than jumping across it; remain eight-connected.
        cells = {(r, min(r % (2*w-2), 2*w-2-r % (2*w-2))) for r in range(h)} | {(0, c) for c in range(w)}
    elif style == 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
        if h > 2 and w > 2:
            cells.add((1, 1))
    elif style == 4:
        cells = {(0, 0)}
        target = int(rng.integers(max(3, h*w//3), h*w+1))
        for _ in range(target-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    # Randomly reverse the payload, retaining its local coordinate frame.
    if rng.integers(2):
        cells = {(h-1-r, c) for r, c in cells}
    if rng.integers(2):
        cells = {(r, w-1-c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    cells = {(r-r0, c-c0) for r, c in cells}
    if kind == 'P':
        # Connect the corner with an eight-neighbor bridge.
        target = min(cells)
        r = c = 0
        cells.add((r, c))
        while (r, c) != target:
            r += int(np.sign(target[0]-r))
            c += int(np.sign(target[1]-c))
            cells.add((r, c))
        a, b = [(1, 1), (0, 1), (1, 0)][int(rng.integers(3))]
        sr, sc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        src = sorted((sr*(r+a), sc*(c+b)) for r, c in cells)
        mark, dst = [(0, 0)], [(-r, -c) for r, c in src]
    else:
        sign = int(rng.choice([-1, 1]))
        src = sorted((sign*(r+1), c) for r, c in cells)
        contact = int(rng.choice([c for r, c in cells if r == 0]))
        lo = int(rng.integers(-1, contact+1))
        hi = int(rng.integers(max(lo+1, contact), max(lo+2, w+1)))
        mark = [(0, c) for c in range(lo, hi+1)]
        dst = [(-r, c) for r, c in src]
        if kind == 'V':
            src, mark, dst = [[(c, r) for r, c in part] for part in (src, mark, dst)]
    def translated(part):
        rr, cc = min(r for r, c in part), min(c for r, c in part)
        return {(r-rr, c-cc) for r, c in part}
    if witness and translated(src) == translated(dst):
        # A three-cell L is asymmetric under each requested operation.
        src = [(1, 1), (1, 2), (2, 1)] if kind == 'P' else [(1, 0), (1, 1), (2, 0)]
        mark = [(0, 0)] if kind == 'P' else [(0, 0), (0, 1)]
        dst = [(-r, -c) if kind == 'P' else (-r, c) for r, c in src]
        if kind == 'V':
            src, mark, dst = [[(c, r) for r, c in part] for part in (src, mark, dst)]
    footprint = src + mark + dst
    r0, c0 = min(r for r, c in footprint), min(c for r, c in footprint)
    return {'kind': kind, 'src': [(r-r0, c-c0) for r, c in src],
            'mark': [(r-r0, c-c0) for r, c in mark],
            'dst': [(r-r0, c-c0) for r, c in dst]}


def place_chambers(rng, h, w):
    regime = int(rng.integers(4))
    if regime <= 1:
        row = int(rng.integers(2, h-2))
        delta = 0 if regime == 0 else int(rng.choice([-2, -1, 1, 2]))
        row2 = min(h-3, max(2, row+delta))
        centers = [(row, int(rng.integers(1, 5))), (row2, int(rng.integers(w-5, w-1)))]
    else:
        r0, r1 = int(rng.integers(1, h//2-2)), int(rng.integers(h//2+2, h-1))
        c0, c1 = int(rng.integers(1, w//2-2)), int(rng.integers(w//2+2, w-1))
        centers = [(r0, c0), (r1, c1)] if regime == 2 else [(r0, c1), (r1, c0)]
    openings = [int(rng.integers(4)), int(rng.integers(4))]
    if regime <= 1 and rng.random() < .5:
        openings = [1, 3] if rng.random() < .75 else [3, 1]
    if any(not (0 <= r+2*DIRS[d][0] < h and 0 <= c+2*DIRS[d][1] < w)
           for (r, c), d in zip(centers, openings)):
        return None
    return centers, openings


def _sample_geometry(rng, kinds, witness=True, cap=4):
    rejects = {'fit': 0, 'disconnect': 0, 'noeffect': 0, 'notie': 0}
    for attempt in range(800):
        low = 12 if len(kinds) == 1 else 14 if len(kinds) == 2 else 16
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        ch = place_chambers(rng, h, w)
        if ch is None:
            continue
        centers, openings = ch
        inp = np.zeros((h, w), np.int8)
        blocked = np.zeros_like(inp, bool)
        for (r, c), d in zip(centers, openings):
            inp[r-1:r+2, c-1:c+2] = 8
            inp[r, c] = 3
            dr, dc = DIRS[d]
            inp[r+dr, c+dc] = 0
            blocked[max(0, r-2):min(h, r+3), max(0, c-2):min(w, c+3)] = True
        placed = []
        for kind in kinds:
            g = proto(rng, kind, cap, witness)
            fp = g['src'] + g['mark'] + g['dst']
            fh, fw = max(r for r, c in fp)+1, max(c for r, c in fp)+1
            coords = [(r, c) for r in range(h-fh+1) for c in range(w-fw+1)]
            rng.shuffle(coords)
            t = float(rng.uniform(.20, .80))
            focus = [centers[0][j]*(1-t)+centers[1][j]*t for j in range(2)]
            style = float(rng.random())
            if style < .8:
                coords.sort(key=lambda q: abs(q[0]+fh/2-focus[0])+abs(q[1]+fw/2-focus[1]))
            accepted = None
            for r, c in coords:
                if not any(blocked[r+rr, c+cc] for rr, cc in fp):
                    accepted = r, c
                    break
            if accepted is None:
                break
            r, c = accepted
            g = {'kind': kind, **{key: [(rr+r, cc+c) for rr, cc in g[key]]
                                  for key in ('src', 'mark', 'dst')}}
            for rr, cc in g['src']:
                inp[rr, cc] = 1
            for rr, cc in g['mark']:
                inp[rr, cc] = 5
            for rr, cc in fp:
                rr, cc = rr+r, cc+c
                blocked[max(0, rr-1):min(h, rr+2), max(0, cc-1):min(w, cc+2)] = True
            placed.append(g)
        if len(placed) != len(kinds):
            rejects['fit'] += 1
            continue
        if np.count_nonzero(inp == 0) <= inp.size//2:
            rejects['fit'] += 1
            continue
        full = inp.copy()
        for g in placed:
            for q in g['dst']:
                full[q] = 1
        mask, length = routes(full, centers)
        if length < 0:
            rejects['disconnect'] += 1
            continue
        bare, _ = routes(inp, centers)
        if not np.any((mask != bare) & (full == 0)):
            rejects['noeffect'] += 1
            continue
        fails = False
        for g in placed:
            less = full.copy()
            for q in g['dst']:
                less[q] = 0
            alt, _ = routes(less, centers)
            if not np.any((mask != alt) & (full == 0)):
                fails = True
                break
        if fails:
            rejects['noeffect'] += 1
            continue
        if mask.sum() <= length-1:
            rejects['notie'] += 1
            continue
        blocked = full != 0
        for q in centers:
            blocked[q] = False
        reachable = dist(blocked, centers[0]) >= 0
        if not np.any(reachable & (full == 0) & ~mask):
            continue
        return inp, full, centers, placed, mask, length
    raise RuntimeError(('failed', kinds, rejects))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'Invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid must contain integers')
    _require(np.all((0 <= a) & (a <= 9)), 'Invalid grid color')
    return a.astype(np.int8, copy=True)


def _normalized(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return frozenset((r-r0, c-c0) for r, c in cells)


def _kind(marker):
    if len(marker) == 1:
        return 'P'
    rows, cols = {r for r, c in marker}, {c for r, c in marker}
    if len(rows) == 1 and len(marker) == max(cols)-min(cols)+1:
        return 'H'
    if len(cols) == 1 and len(marker) == max(rows)-min(rows)+1:
        return 'V'
    raise AssertionError('A marker is neither a straight segment nor a singleton')


def _map_payload(marker, kind, payload, inp):
    if min(max(abs(r-a), abs(c-b)) for r, c in payload for a, b in marker) != 1:
        return None
    ar, ac = marker[0]
    r0, r1 = min(r for r, c in payload), max(r for r, c in payload)
    c0, c1 = min(c for r, c in payload), max(c for r, c in payload)
    if kind == 'H':
        if not (r1 < ar or r0 > ar):
            return None
        if c1 < min(c for r, c in marker) or c0 > max(c for r, c in marker):
            return None
        dest = [(2*ar-r, c) for r, c in payload]
    elif kind == 'V':
        if not (c1 < ac or c0 > ac):
            return None
        if r1 < min(r for r, c in marker) or r0 > max(r for r, c in marker):
            return None
        dest = [(r, 2*ac-c) for r, c in payload]
    else:
        if not ((r1 <= ar or r0 >= ar) and (c1 <= ac or c0 >= ac)):
            return None
        dest = [(2*ar-r, 2*ac-c) for r, c in payload]
    h, w = inp.shape
    if any(not (0 <= r < h and 0 <= c < w) or inp[r, c] != 0 for r, c in dest):
        return None
    return dest


def _read_input(inp):
    """Reconstruct ownership, completed walls and gates from actual input cells."""
    h, w = inp.shape
    _require(np.count_nonzero(inp == 0) > inp.size//2, 'Background must be a clear zero field')
    centers, openings, chamber_cells, walls = [], [], set(), set()
    # Shell geometry, rather than a frequency guess, identifies the terminals.
    for r, c in np.argwhere((inp != 0) & (inp != 5) & (inp != 8)).tolist():
        if not (1 <= r < h-1 and 1 <= c < w-1):
            continue
        surrounding = {(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if dr or dc}
        local_walls = {q for q in surrounding if inp[q] == 8}
        gate = [d for d, (dr, dc) in enumerate(DIRS) if inp[r+dr, c+dc] == 0]
        if len(local_walls) == 7 and len(gate) == 1:
            centers.append((r, c))
            openings.append(gate[0])
            chamber_cells.update(surrounding | {(r, c)})
            walls.update(local_walls)
    _require(len(centers) == 2 and len(chamber_cells) == 18, 'Need two disjoint gated chambers')
    terminal = int(inp[centers[0]])
    _require(terminal == int(inp[centers[1]]) and np.count_nonzero(inp == terminal) == 2,
             'The two terminals must share an otherwise unused color')
    _require(walls == set(map(tuple, np.argwhere(inp == 8).tolist())), 'Extraneous shell walls')
    for (r, c), d in zip(centers, openings):
        dr, dc = DIRS[d]
        _require(0 <= r+2*dr < h and 0 <= c+2*dc < w and inp[r+2*dr, c+2*dc] == 0,
                 'A terminal gate is blocked')
    markers = comps(map(tuple, np.argwhere(inp == 5).tolist()))
    payloads = comps(map(tuple, np.argwhere((inp != 0) & (inp != 5) &
                                          (inp != 8) & (inp != terminal)).tolist()))
    _require(1 <= len(markers) <= 4 and len(markers) == len(payloads), 'Invalid group count')
    _require(all(len({int(inp[q]) for q in cells}) == 1 for cells in payloads),
             'A payload must be monochrome and eight-connected')
    groups, used, destinations = [], set(), set()
    for marker in markers:
        kind = _kind(marker)
        candidates = []
        for index, payload in enumerate(payloads):
            dest = _map_payload(marker, kind, payload, inp)
            if dest is not None:
                candidates.append((index, dest))
        _require(len(candidates) == 1, 'Ambiguous or missing marker/payload association')
        index, dest = candidates[0]
        _require(index not in used and not destinations.intersection(dest), 'Ownership/copy collision')
        used.add(index)
        destinations.update(dest)
        src = payloads[index]
        groups.append({'kind': kind, 'mark': marker, 'src': src, 'dst': dest, 'color': int(inp[src[0]])})
    _require(len(used) == len(payloads), 'Unclaimed payload')
    footprints = [set(g['src'] + g['mark'] + g['dst']) for g in groups]
    for i, cells in enumerate(footprints):
        other = set().union(*footprints[:i], chamber_cells)
        _require(all(max(abs(r-a), abs(c-b)) > 1 for r, c in cells for a, b in other),
                 'Completed groups must have a one-cell separation from other groups and chambers')
    completed = inp.copy()
    for g in groups:
        for q in g['dst']:
            completed[q] = g['color']
    for (r, c), d in zip(centers, openings):
        dr, dc = DIRS[d]
        _require(completed[r+dr, c+dc] == 0 and completed[r+2*dr, c+2*dc] == 0,
                 'Completion closes a terminal gate')
    return completed, centers, openings, groups, terminal


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; never calls generation or consults stored examples."""
    inp = _grid(input_grid)
    completed, centers, _, _, terminal = _read_input(inp)
    union, length = routes(completed, centers)
    if length < 0:
        raise ValueError('Completed walls disconnect the terminals')
    out = completed.copy()
    out[union] = terminal
    return out


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _make_pair(rng, kinds, witness):
    inp, completed, centers, groups, union, _ = _sample_geometry(rng, kinds, witness)
    terminal = int(rng.choice([1, 2, 3, 4, 6, 7, 9]))
    pool = [c for c in (1, 2, 3, 4, 6, 7, 9) if c != terminal]
    pool = [int(c) for c in rng.choice(pool, size=int(rng.integers(1, min(4, len(groups))+1)), replace=False)]
    inp[inp == 3] = terminal
    # Repaint explicit cells; no palette frequency assumption is needed by solve.
    for g in groups:
        color = int(rng.choice(pool))
        for q in g['src']:
            inp[q] = color
        for q in g['src'] + g['dst']:
            completed[q] = color
    completed[tuple(np.array(centers).T)] = terminal
    out = completed.copy()
    out[union] = terminal
    turns = int(rng.integers(4))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    if rng.integers(2):
        inp, out = inp[:, ::-1], out[:, ::-1]
    # Explicit copies ensure no shared mutable arrays or negative-stride surprises.
    return {'input': inp.copy(), 'output': out.copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    # Even two demonstrations establish both axis orientations and the half-turn.
    # Training isolates these operations; testing recombines their wall effects.
    plans = [(['H', 'V'], True), (['P'], True)]
    for _ in range(num_train-2):
        count = int(rng.integers(1, 4))
        kinds = (['P']*count if rng.integers(2) else
                 [str(rng.choice(['H', 'V'])) for _ in range(count)])
        plans.append((kinds, False))
    order = rng.permutation(len(plans))
    plans = [plans[int(i)] for i in order]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for index in range(count):
            if split == 'train':
                kinds, witness = plans[index]
            else:
                kinds = ['P', str(rng.choice(['H', 'V']))]
                for _ in range(int(rng.choice(3, p=[.45, .45, .10]))):
                    kinds.append(str(rng.choice(['H', 'V', 'P'])))
                witness = False
            kinds = list(kinds)
            rng.shuffle(kinds)
            for _ in range(16):
                pair = _make_pair(rng, kinds, witness)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not obtain a distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, 'Canvas changed')
    completed, centers, openings, groups, terminal = _read_input(inp)
    union, length = routes(completed, centers)
    _require(length >= 0, 'Terminals disconnected')
    expected = completed.copy()
    expected[union] = terminal
    _require(np.array_equal(out, expected), 'Not symmetry completion followed by full shortest-route union')
    _require(np.array_equal(out[inp != 0], inp[inp != 0]), 'An existing cell changed')
    _require(int(union.sum()) > length-1, 'This generated scene must witness tied routes')
    bare, _ = routes(inp, centers)
    common = completed == 0
    _require(np.any((bare != union) & common), 'Symmetry did not change routing on surviving open cells')
    effects = []
    for g in groups:
        less = completed.copy()
        for q in g['dst']:
            less[q] = 0
        alt, _ = routes(less, centers)
        effect = int(np.count_nonzero((alt != union) & common))
        _require(effect > 0, 'An obstacle copy has no causal routing contribution')
        effects.append(effect)
    blocked = completed != 0
    for q in centers:
        blocked[q] = False
    reachable = dist(blocked, centers[0]) >= 0
    _require(np.any(reachable & common & ~union), 'Flood-fill shortcut is not refuted')
    for (r, c), d in zip(centers, openings):
        dr, dc = DIRS[d]
        _require(union[r+dr, c+dc], 'A mandatory gate was not highlighted')
    return {'shape': inp.shape, 'centers': centers, 'openings': openings,
            'groups': groups, 'terminal': terminal, 'length': length,
            'route_cells': int(union.sum()), 'copy_cells': sum(len(g['src']) for g in groups),
            'effects': effects, 'union': union}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'Splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'Invalid split counts')
    seen, trained, witnessed = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            inp = _grid(pair['input'])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, 'Repeated input')
            seen.add(key)
            kinds = {g['kind'] for g in info['groups']}
            if split == 'train':
                trained.update(kinds)
                witnessed.update(g['kind'] for g in info['groups']
                                  if _normalized(g['src']) != _normalized(g['dst']))
            else:
                _require(kinds <= trained, 'A test operation lacks training evidence')
                _require('P' in kinds and bool(kinds & {'H', 'V'}), 'Test lacks combined wall mechanisms')
    _require(trained == {'H', 'V', 'P'} and trained <= witnessed,
             'Training must distinguish mirrors and half-turns from untranslated copies')


HOST_SEEDS = (0, 485993720, 916919995, 843562703, 869733972, 103987638, 794559361,
              817164574, 636709488, 15190653, 447418950, 853616565, 496452181,
              691474397, 857661610, 896582615, 257996444, 761696734, 466720595,
              489517411, 39015024, 748328576, 980165246, 78201744, 80500243,
              267451203, 149270262, 901302251, 578560291, 846380665, 492240646,
              120940128, 16760254, 243629341, 446971623, 721768076, 545352575,
              750043052, 855523486, 488921855, 707098249, 974391979)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed measurements plus algebraic and mutation regressions.

    Keeps one puzzle at a time, small sets of signatures, and no disk artifacts.
    The returned measurements are descriptive, not independent semantic approval.
    """
    signatures, shapes, silhouettes, group_counts, colors, gates, layouts = (set() for _ in range(7))
    total_pairs = total_groups = route_cells = copy_cells = causal_cells = 0
    lengths, union_sizes = set(), set()
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if first is None:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                inp = _grid(pair['input'])
                completed, centers, _, groups, _ = _read_input(inp)
                # Cross-check coordinate mappings with cropped-array transforms.
                for g in groups:
                    r0, r1 = min(r for r, c in g['src']), max(r for r, c in g['src'])
                    c0, c1 = min(c for r, c in g['src']), max(c for r, c in g['src'])
                    local = np.zeros((r1-r0+1, c1-c0+1), bool)
                    for r, c in g['src']:
                        local[r-r0, c-c0] = True
                    ar, ac = g['mark'][0]
                    if g['kind'] == 'H':
                        local, top, left = np.flipud(local), 2*ar-r1, c0
                    elif g['kind'] == 'V':
                        local, top, left = np.fliplr(local), r0, 2*ac-c1
                    else:
                        local, top, left = np.rot90(local, 2), 2*ar-r1, 2*ac-c1
                    mapped = {(top+int(r), left+int(c)) for r, c in np.argwhere(local)}
                    _require(mapped == set(g['dst']), 'Array-transform symmetry regression')
                    silhouettes.add(tuple(sorted(_normalized(g['src']))))
                # Check the union against backward traversal of the shortest-path
                # DAG, an algebraic consistency check rather than a blind solver.
                blocked = completed != 0
                for q in centers:
                    blocked[q] = False
                d = dist(blocked, centers[0])
                stack, dag = [centers[1]], {centers[1]}
                while stack:
                    r, c = stack.pop()
                    for dr, dc in DIRS:
                        q = r+dr, c+dc
                        if (0 <= q[0] < inp.shape[0] and 0 <= q[1] < inp.shape[1]
                                and d[q] >= 0 and d[q] == d[r, c]-1 and q not in dag):
                            dag.add(q)
                            stack.append(q)
                _require({q for q in dag if completed[q] == 0} ==
                         set(map(tuple, np.argwhere(info['union']).tolist())), 'Shortest-path DAG regression')
                _require(np.array_equal(solve(inp), pair['output']), 'Input-only solve disagrees')
                shapes.add(info['shape'])
                layouts.add((info['shape'], tuple(info['centers']), tuple(info['openings'])))
                group_counts.add(len(groups))
                colors.add(info['terminal'])
                gates.add(tuple(info['openings']))
                lengths.add(info['length'])
                union_sizes.add(info['route_cells'])
                total_pairs += 1
                total_groups += len(groups)
                route_cells += info['route_cells']
                copy_cells += info['copy_cells']
                causal_cells += sum(info['effects'])
    _require(len(signatures) == 42, 'Whole-puzzle diversity')
    _require(len(shapes) >= 40 and len(layouts) >= 180, 'Geometry diversity')
    _require(len(silhouettes) >= 35 and len(group_counts) >= 3, 'Shape/count diversity')
    _require(len(colors) == 7 and len(gates) >= 12, 'Visible-role/gate diversity')
    _require(len(lengths) >= 20 and len(union_sizes) >= 35, 'Route diversity')
    _require(_fingerprint(first) == _fingerprint(generate_puzzle(0)), 'Seed reproducibility')
    for nt, ns in ((2, 1), (10, 4), (6, 3)):
        p = generate_puzzle(713, nt, ns)
        _require(len(p['train']) == nt and len(p['test']) == ns, 'Counts ignored')
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Missing one tied-route cell and one symmetric wall cell are both errors.
    info = _inspect_pair(first['train'][0])
    for missing in (tuple(np.argwhere(info['union'])[0]), info['groups'][0]['dst'][0]):
        bad = {s: [{k: _grid(v) for k, v in p.items()} for p in first[s]] for s in first}
        bad['train'][0]['output'][missing] = 0
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('A deliberately missing route/wall cell was accepted')
    # Solving support is broader than the active-copy/tied-route sampling filter.
    # This valid scene has an off-route point group and a unique straight route.
    simple = np.zeros((11, 17), np.int8)
    for r, c, gate in ((7, 2, 1), (7, 14, 3)):
        simple[r-1:r+2, c-1:c+2] = 8
        simple[r, c] = 3
        dr, dc = DIRS[gate]
        simple[r+dr, c+dc] = 0
    simple[2, 8] = 5
    simple[1, 9:11] = 1
    expected = simple.copy()
    expected[3, 6:8] = 1
    expected[7, 3:14] = 3
    _require(np.array_equal(solve(simple), expected), 'Unique-route input-only support regression')
    return {'seeds': 42, 'pairs': total_pairs, 'groups': total_groups,
            'distinct_puzzles': len(signatures), 'grid_shapes': len(shapes),
            'layouts': len(layouts), 'payload_silhouettes': len(silhouettes),
            'group_counts': sorted(group_counts), 'terminal_colors': sorted(colors),
            'ordered_gate_pairs': len(gates), 'shortest_lengths': len(lengths),
            'route_sizes': len(union_sizes), 'route_cells': route_cells,
            'copied_wall_cells': copy_cells, 'per_group_surviving_cell_effects': causal_cells,
            'independent_solver': False}
