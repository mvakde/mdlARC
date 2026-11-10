"""Edit-controlled printing: compare plates, count signed edit regions, print.

Only NumPy and the standard library are used. The solver is input-only; the
sampling profiles below teach the fixed rule and do not participate in solving.
"""

from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "color range")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    group.append(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty exceptional stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _decode(input_grid):
    """Recover every role, plate, stencil and quantity solely from input cells."""
    g = _grid(input_grid)
    lattice = int(g[0, 0])
    rs = np.flatnonzero(np.all(g == lattice, axis=1))
    cs = np.flatnonzero(np.all(g == lattice, axis=0))
    _require(len(rs) >= 2 and len(cs) >= 4, "outer frame and at least three copies")
    _require(rs[0] == cs[0] == 0 and rs[-1] == g.shape[0]-1
             and cs[-1] == g.shape[1]-1, "complete outer frame")
    _require(bool(np.all(np.diff(rs) == rs[1]) and np.all(np.diff(cs) == cs[1])),
             "regular one-cell lattice")
    ph, pw = int(rs[1]-1), int(cs[1]-1)
    _require(ph >= 3 and pw >= 3, "carrier perimeter and nonempty core")
    panels = [[g[r+1:s, c+1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _require(all(np.all(p != lattice) for row in panels for p in row),
             "lattice only in frame and separators")
    counts = Counter(_key(p) for row in panels for p in row)
    largest = max(counts.values())
    winners = [k for k, n in counts.items() if n == largest]
    _require(len(winners) == 1, "unique common plate")
    master = next(p for row in panels for p in row if _key(p) == winners[0])
    carrier = int(master[0, 0])
    colors = set(map(int, np.unique(g)))
    _require(len(colors) == 3 and carrier != lattice, "three distinct visible roles")
    ink = next(iter(colors - {lattice, carrier}))
    _require(set(map(int, np.unique(master))) == {carrier, ink}, "binary master")
    records = []
    for row in panels:
        for p in row:
            _require(np.all(p[[0, -1], :] == carrier)
                     and np.all(p[:, [0, -1]] == carrier), "carrier perimeter")
        exceptions = [(c, p) for c, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(exceptions) == 1, "one visible exception per band")
        column, plate = exceptions[0]
        added = (master == carrier) & (plate == ink)
        erased = (master == ink) & (plate == carrier)
        ac, rc = _components(added), _components(erased)
        _require(len(ac) == len(_components(added, True))
                 and len(rc) == len(_components(erased, True)), "diagonally touching edit regions")
        _require(len(ac) + len(rc) > 0, "visible edit required")
        _require(np.any((master == ink) & (plate == ink)), "retain some original ink")
        tile = _crop(plate == ink)
        records.append(dict(tile=tile, additions=len(ac), erasures=len(rc),
                            add_area=int(added.sum()), erase_area=int(erased.sum()),
                            add_sizes=tuple(map(len, ac)), erase_sizes=tuple(map(len, rc)),
                            column=column, plate=plate, added=added, erased=erased))
    return dict(lattice=lattice, carrier=carrier, ink=ink, master=master,
                ph=ph, pw=pw, copies=len(panels[0]), bands=len(panels), records=records)


def _pack(blocks, lattice):
    height = sum(b.shape[0] for b in blocks) + len(blocks) + 1
    width = max(b.shape[1] for b in blocks) + 2
    _require(height <= 30 and width <= 30, "print ledger exceeds the grid budget")
    out = np.full((height, width), lattice, dtype=np.int8)
    offset = 1
    for block in blocks:
        out[offset:offset+block.shape[0], 1:1+block.shape[1]] = block
        offset += block.shape[0] + 1
    return out


def _print_by_index(tile, additions, erasures, ink, carrier):
    h, w = tile.shape
    height, width = (erasures+1)*(h+1)-1, (additions+1)*(w+1)-1
    rr, cc = np.indices((height, width))
    valid = (rr % (h+1) < h) & (cc % (w+1) < w)
    stencil = tile[np.minimum(rr % (h+1), h-1), np.minimum(cc % (w+1), w-1)]
    return np.where(valid & stencil, ink, carrier).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    blocks = [_print_by_index(r['tile'], r['additions'], r['erasures'], d['ink'], d['carrier'])
              for r in d['records']]
    return _pack(blocks, d['lattice'])


def _print_by_concatenation(tile, additions, erasures, ink, carrier):
    """Construction route: quantities are from generated masks, not solve()."""
    colored = np.where(tile, ink, carrier).astype(np.int8)
    gap = np.full((tile.shape[0], 1), carrier, dtype=np.int8)
    row = np.concatenate([x for i in range(additions+1)
                          for x in ((colored, gap) if i < additions else (colored,))], axis=1)
    line = np.full((1, row.shape[1]), carrier, dtype=np.int8)
    return np.concatenate([x for i in range(erasures+1)
                           for x in ((row, line) if i < erasures else (row,))], axis=0)


def _transform(mask, index):
    return np.rot90(mask[:, ::-1] if index >= 4 else mask, index % 4)


def _orientation_witness(tiles):
    return all(any(not np.array_equal(t, _transform(t, k)) for t in tiles) for k in range(1, 8))


def _master(rng, h, w):
    rr, cc = np.indices((h, w))
    style = int(rng.integers(6))
    density = float(rng.uniform(.25, .75))
    if style == 0:
        mask = rng.random((h, w)) < density
    elif style == 1:
        mask = np.zeros((h, w), dtype=bool)
        frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
        for _ in range(max(1, int(h*w*density))):
            cells = sorted(frontier)
            if not cells:
                break
            r, c = cells[int(rng.integers(len(cells)))]
            frontier.remove((r, c))
            mask[r, c] = True
            for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                    frontier.add((x, y))
    elif style == 2:
        period = int(rng.integers(2, 4))
        coordinate = (rr, cc, rr+cc)[int(rng.integers(3))]
        mask = coordinate % period == int(rng.integers(period))
    elif style == 3:
        mask = np.zeros((h, w), dtype=bool)
        mask[[0, -1], :] = mask[:, [0, -1]] = True
        if rng.random() < .4:
            mask[h//2, :max(1, w//2)] = True
    elif style == 4:
        mask = np.zeros((h, w), dtype=bool)
        col = int(rng.integers(w))
        mask[:, col] = True
        for r in range(h):
            if rng.random() < .7:
                end = int(rng.integers(w))
                mask[r, min(col, end):max(col, end)+1] = True
    else:
        mask = (rr % 2 == int(rng.integers(2))) & (cc % 2 == int(rng.integers(2)))
        mask &= rng.random((h, w)) < .9
    if rng.random() < .2:
        mask = ~mask
    if rng.random() < .2:
        mask |= mask[:, ::-1]
    return mask


def _patches(rng, allowed, count, area=None):
    """Grow patches within one template color, separating same-sign patches."""
    h, w = allowed.shape
    if count == 0:
        return np.zeros_like(allowed)
    for _ in range(24):
        result = np.zeros_like(allowed)
        blocked = np.zeros_like(allowed)
        sizes = [1]*count
        if area is not None:
            for _ in range(area-count):
                sizes[int(rng.integers(count))] += 1
        else:
            sizes = list(map(int, rng.integers(1, 9, size=count)))
        good = True
        for target in sizes:
            free = allowed & ~blocked
            points = np.argwhere(free)
            if not len(points):
                good = False
                break
            start = tuple(map(int, points[int(rng.integers(len(points)))]))
            group, frontier = set(), {start}
            for _ in range(target):
                if not frontier:
                    break
                options = sorted(frontier)
                r, c = options[int(rng.integers(len(options)))]
                frontier.remove((r, c))
                group.add((r, c))
                for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if 0 <= x < h and 0 <= y < w and free[x, y] and (x, y) not in group:
                        frontier.add((x, y))
            if area is not None and len(group) != target:
                good = False
                break
            for r, c in group:
                result[r, c] = True
                blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        if good and len(_components(result)) == count == len(_components(result, True)):
            return result
    return None


def _fallback(kind, direction):
    """Explicit feasible geometry, used only after bounded random attempts."""
    if kind in ('add_witness', 'erase_witness'):
        mask = np.array([[1, 0, 1, 0, 1], [1, 1, 1, 0, 0],
                         [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]], dtype=bool)
        zero = np.zeros_like(mask)
        domino_blank, singles_blank = zero.copy(), zero.copy()
        domino_blank[1, 3:5] = True
        singles_blank[0, 1] = singles_blank[2, 2] = True
        domino_ink, singles_ink = zero.copy(), zero.copy()
        domino_ink[1, 0:2] = True
        singles_ink[0, 4] = singles_ink[3, 0] = True
        changes = ([(domino_blank, zero), (singles_blank, zero)] if kind == 'add_witness'
                   else [(zero, domino_ink), (domino_blank, singles_ink)])
    else:
        rr, cc = np.indices((4, 5))
        mask = (rr+cc) % 2 == 0
        added, erased = np.zeros_like(mask), np.zeros_like(mask)
        if kind == 'test' and direction == 'add':
            added[0, 1] = added[0, 3] = added[2, 1] = True
            erased[3, 1] = True
        elif kind == 'test':
            erased[0, 0] = erased[0, 2] = erased[0, 4] = True
            added[3, 0] = True
        else:
            added[0, 1] = erased[3, 1] = True
        changes = [(added, erased)]
    return mask, changes, 3


def _assemble_input(master_mask, changes, columns, copies, palette):
    lattice, carrier, ink = palette
    h, w = master_mask.shape
    ph, pw = h+2, w+2
    master = np.full((ph, pw), carrier, dtype=np.int8)
    master[1:-1, 1:-1][master_mask] = ink
    g = np.full((len(changes)*(ph+1)+1, copies*(pw+1)+1), lattice, dtype=np.int8)
    blocks = []
    for b, ((added, erased), column) in enumerate(zip(changes, columns)):
        variant_mask = (master_mask | added) & ~erased
        variant = master.copy()
        variant[1:-1, 1:-1] = np.where(variant_mask, ink, carrier)
        for c in range(copies):
            r0, c0 = 1+b*(ph+1), 1+c*(pw+1)
            g[r0:r0+ph, c0:c0+pw] = variant if c == column else master
        blocks.append(_print_by_concatenation(_crop(variant_mask), len(_components(added)),
                                              len(_components(erased)), ink, carrier))
    return {'input': g, 'output': _pack(blocks, lattice)}


def _make_pair(rng, kind, palette, direction=None):
    for attempt in range(128):
        if kind in ('add_witness', 'erase_witness'):
            h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
            profiles = ([(1, 0, 2, 0), (2, 0, 2, 0)] if kind == 'add_witness'
                        else [(0, 1, 0, 2), (1, 2, 2, 2)])
        else:
            h, w = int(rng.integers(1, 8)), int(rng.integers(1, 7))
            if h*w < 4:
                continue
            max_bands = min(4, 29//(h+3))
            bands = int(rng.integers(1, max_bands+1))
            profiles = []
            for b in range(bands):
                a, r = map(int, rng.integers(0, 3, size=2))
                if a+r == 0:
                    a = 1
                if kind == 'test' and b == 0:
                    a, r = (3, int(rng.integers(2))) if direction == 'add' else (int(rng.integers(2)), 3)
                profiles.append((a, r, None, None))
        master = _master(rng, h, w)
        if not master.any() or master.all():
            continue
        changes = []
        for a, r, aa, ra in profiles:
            added = _patches(rng, ~master, a, aa)
            erased = _patches(rng, master, r, ra)
            if added is None or erased is None or not np.any(master & ~erased):
                break
            # Compare a recognizable plate; do not invert every core cell.
            if (added.sum()+erased.sum()) > max(4, .65*h*w):
                break
            changes.append((added, erased))
        if len(changes) != len(profiles):
            continue
        if kind == 'add_witness' and not _orientation_witness(
                [_crop((master | a) & ~r) for a, r in changes]):
            continue
        if kind == 'add_witness' and not any(
                len(_components((master | a) & ~r)) != len(_components(a))+len(_components(r))
                for a, r in changes):
            continue
        if kind == 'add_witness' and not any(
                len(_components((master | a) & ~r)) > 1 for a, r in changes):
            continue
        copies = int(rng.integers(3, 29//(w+3)+1))
        columns = rng.integers(copies, size=len(changes))
        if kind == 'add_witness' and len(set(map(int, columns))) == 1:
            columns[1] = (columns[0]+int(rng.integers(1, copies))) % copies
        # The list order is not a count or sign convention.
        if kind in ('add_witness', 'erase_witness') and rng.random() < .5:
            changes.reverse()
        try:
            return _assemble_input(master, changes, columns, copies, palette)
        except AssertionError:
            # Oversized prints require a different geometry, never truncation.
            continue
    master, changes, copies = _fallback(kind, direction)
    columns = rng.integers(copies, size=len(changes))
    if kind == 'add_witness':
        columns[1] = (columns[0]+1) % copies
    return _assemble_input(master, changes, columns, copies, palette)


def _requested_count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= value <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # All 720 permutations preserve visible roles; at most 13 earlier inputs
    # could conflict. No source data or hidden palette dictionary is needed.
    import itertools
    colors = sorted(map(int, np.unique(pair['input'])))
    for palette in itertools.permutations(range(10), 3):
        lookup = np.arange(10, dtype=np.int8)
        lookup[colors] = palette
        candidate = {k: lookup[v] for k, v in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('palette exhaustion')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _requested_count(num_train, 'num_train', 2, 10)
    nq = _requested_count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shared_palette = rng.random() < .25
    base_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, amount in (('train', nt), ('test', nq)):
        for index in range(amount):
            kind = ('add_witness' if index == 0 else 'erase_witness' if index == 1 else 'ordinary')
            direction = None
            if split == 'test':
                kind, direction = 'test', str(rng.choice(['add', 'erase']))
            palette = base_palette if shared_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
            pair = _unique(_make_pair(rng, kind, palette, direction), seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, data = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(solve(inp), out), 'incorrect print ledger')
            _require(not np.array_equal(inp, out), 'identity transformation')
            data[split].append(d)
    train = [r for d in data['train'] for r in d['records']]
    _require(any(r['additions'] > 0 and r['erasures'] == 0 for r in train), 'pure addition witness')
    _require(any(r['erasures'] > 0 and r['additions'] == 0 for r in train), 'pure erasure witness')
    _require(any(r['add_area'] == r['erase_area'] > 0 for r in train), 'balanced mixed edit witness')
    for area, count in (('add_area', 'additions'), ('erase_area', 'erasures')):
        _require(any(a[area] == b[area] > 0 and a[count] != b[count] for a in train for b in train),
                 'equal edit area with different region counts: '+area)
    _require(any(len({r['column'] for r in d['records']}) > 1 for d in data['train']),
             'exception position witness')
    _require(_orientation_witness([r['tile'] for r in train]), 'stencil orientation witness')
    _require(any(len(_components(r['tile'])) > 1 for r in train),
             'demonstrate cropping disconnected ink as one stencil')
    _require(any(r['tile'].shape != (d['ph'], d['pw']) for d in data['train'] for r in d['records']),
             'tight cropping witness')
    # Parent ablations: A supplies the actual stencil and the signed counter;
    # B supplies topology and replication, rather than merely extra decoration.
    _require(any(not np.array_equal(r['tile'], _crop(d['master'] == d['ink']))
                 for d in data['train'] for r in d['records']), 'using master would give same stencil')
    _require(any(r['add_area'] > r['additions'] or r['erase_area'] > r['erasures'] for r in train),
             'pixel counting not distinguished')
    _require(any(len(_components(r['plate'] == d['ink'])) != r['additions'] + r['erasures']
                 for d in data['train'] for r in d['records']), 'counting plate ink not distinguished')
    _require(any(r['additions'] and r['erasures'] for r in train), 'two-axis interaction missing')
    _require(any(not np.array_equal(
        _pack([_print_by_index(r['tile'], 0, 0, d['ink'], d['carrier']) for r in d['records']],
              d['lattice']), np.asarray(pair['output']))
        for d, pair in zip(data['train'], puzzle['train'])), 'omitting repetition changes no answer')
    max_a = max(r['additions'] for r in train)
    max_r = max(r['erasures'] for r in train)
    for d in data['test']:
        _require(any(r['additions'] > max_a or r['erasures'] > max_r for r in d['records']),
                 'test must extrapolate an edit-region count')


def _puzzle_key(p):
    return tuple(tuple((_key(np.asarray(x['input'])), _key(np.asarray(x['output'])))
                       for x in p[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed structural, semantic, diversity and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, 'puzzle diversity')
    for seed in (0, 7, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12300+10*nt+nq, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    decodes = [_decode(pair['input']) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    records = [r for d in decodes for r in d['records']]
    geometries = {(d['ph'], d['pw'], d['bands'], d['copies']) for d in decodes}
    masters = {_key(d['master'] == d['ink']) for d in decodes}
    palettes = {(d['lattice'], d['carrier'], d['ink']) for d in decodes}
    tiles = {_key(r['tile']) for r in records}
    _require(len(geometries) >= 30 and len(masters) >= 70 and len(tiles) >= 100,
             'structural variation')
    _require(len(palettes) >= 60 and {d['lattice'] for d in decodes} == set(range(10)), 'palette variation')
    _require(any(d['bands'] == 1 for d in decodes) and any(d['bands'] >= 3 for d in decodes), 'band variation')
    _require(any(d['ph'] != d['pw'] for d in decodes), 'rectangular plates')
    _require(any(len(_components(d['master'] == d['ink'])) == 1 for d in decodes)
             and any(len(_components(d['master'] == d['ink'])) > 1 for d in decodes), 'motif topology')
    _require(any(r['additions'] >= 3 for r in records) and any(r['erasures'] >= 3 for r in records),
             'both count extrapolations')
    _require(any(max(r['add_sizes']+r['erase_sizes']) >= 3 for r in records), 'patch-size variation')
    _require(any(len({r['column'] for r in d['records']}) < d['bands'] for d in decodes), 'repeated columns')
    for defect in ('label', 'missing_exception', 'perimeter', 'band_order'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = next(p for p in bad['train'] if _decode(p['input'])['bands'] == 2)
        d = _decode(pair['input'])
        if defect == 'label':
            pair['output'][1, 1] = (int(pair['output'][1, 1])+1) % 10
        elif defect == 'missing_exception':
            r = d['records'][0]
            c0 = 1+r['column']*(d['pw']+1)
            pair['input'][1:1+d['ph'], c0:c0+d['pw']] = d['master']
        elif defect == 'perimeter':
            pair['input'][1, 1] = d['ink']
        else:
            blocks = [_print_by_index(r['tile'], r['additions'], r['erasures'], d['ink'], d['carrier'])
                      for r in d['records']][::-1]
            pair['output'] = _pack(blocks, d['lattice'])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('validator accepted '+defect)
    return dict(seeds=42, pairs=len(decodes), bands=len(records), geometries=len(geometries),
                master_bitmaps=len(masters), stencil_bitmaps=len(tiles), role_palettes=len(palettes),
                add_counts=sorted({r['additions'] for r in records}),
                erase_counts=sorted({r['erasures'] for r in records}))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
