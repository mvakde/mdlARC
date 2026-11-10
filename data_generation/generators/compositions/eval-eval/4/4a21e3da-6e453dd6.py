"""Cross-reading apertures: cut a hollow payload, then sort opposite row slices.

Only standard-library code and NumPy are used. No parent generator is imported.
The solver has no access to seeds, sampling configuration, or demonstrations.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    result = np.asarray(value)
    _require(result.ndim == 2 and all(1 <= d <= 30 for d in result.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(result.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= result) & (result <= 9))), "invalid color")
    return result.astype(np.int8, copy=False)


def _components(mask):
    """Four-connected components, in reproducible raster order."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, group = [(r, c)], []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(group)
    return groups


def _holes(mask):
    h, w = mask.shape
    return [g for g in _components(~mask)
            if not any(r in (0, h-1) or c in (0, w-1) for r, c in g)]


def _scene(value):
    """Rotate the unique edge marker to the top; infer all three visible roles."""
    grid = _grid(value)
    h, w = grid.shape
    _require(min(h, w) >= 5, "canvas too small for the family")
    rim = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors, counts = np.unique(rim, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    border = np.zeros(grid.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    points = np.argwhere(border & (grid != bg))
    _require(len(points) == 1, "exactly one isolated border marker required")
    r, c = map(int, points[0])
    _require(sum((r == 0, r == h-1, c == 0, c == w-1)) == 1,
             "marker must not be at a corner")
    marker = int(grid[r, c])
    _require(np.count_nonzero(grid == marker) == 1, "marker color is exclusive")
    payloads = set(map(int, np.unique(grid))) - {bg, marker}
    _require(len(payloads) == 1, "exactly one payload color required")
    payload = payloads.pop()
    turn = 0 if r == 0 else 1 if c == w-1 else 2 if r == h-1 else 3
    canonical = np.rot90(grid, turn).copy()
    _require(canonical.shape[1] >= 7, "not enough transverse space for two banks")
    ac = int(np.flatnonzero(canonical[0] == marker)[0])
    mask = canonical == payload
    _require(not np.any(mask[0]) and not np.any(mask[-1]) and
             not np.any(mask[:, 0]) and not np.any(mask[:, -1]),
             "payload must be interior")
    _require(len(_components(mask)) == 1, "input payload must be four-connected")
    _require(np.any(mask[:, ac]), "guide axis must intersect payload")
    _require(np.any(mask[:, :ac]) and np.any(mask[:, ac+1:]), "both banks required")
    return canonical, turn, bg, payload, marker, ac


def _bank_details(mask, ac, side):
    """Read chambers after cutting out the axis, separately for each component."""
    bank = mask.copy()
    if side == 0:
        bank[:, ac:] = False
    else:
        bank[:, :ac+1] = False
    result = []
    for group in _components(bank):
        rr, cc = np.asarray(group).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        local = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
        local[rr-r0, cc-c0] = True
        extreme = local.shape[1]-1 if side == 0 else 0
        behind = extreme-1 if side == 0 else 1
        active, decoy, contacts = [], [], []
        for hole in _holes(local):
            rows = {r for r, c in hole}
            touched = {r for r, c in hole
                       if c == behind and local[r, extreme]}
            if touched:
                active.append(hole)
                contacts.append(rows == touched)
            else:
                decoy.append(hole)
        enclosed = {p for hole in active + decoy for p in hole}
        # This tempting one-column detector deliberately ignores enclosure.
        naive = {r0+r for r in range(local.shape[0])
                 if 0 <= behind < local.shape[1] and not local[r, behind]
                 and local[r, extreme]}
        open_rows = {r0+r for r in range(local.shape[0])
                     if 0 <= behind < local.shape[1] and not local[r, behind]
                     and (r, behind) not in enclosed and local[r, extreme]}
        result.append(dict(cells=group, mask=local, origin=(r0, c0),
                           active=active, decoy=decoy, contacts=contacts,
                           active_rows={r0+r for hole in active for r, c in hole},
                           all_hole_rows={r0+r for hole in active + decoy for r, c in hole},
                           naive_rows=naive, open_rows=open_rows))
    return result


def _read(value):
    canonical, turn, bg, payload, marker, ac = _scene(value)
    mask = canonical == payload
    banks = [_bank_details(mask, ac, side) for side in (0, 1)]
    return canonical, turn, bg, payload, marker, ac, mask, banks


def _render(info, reading="active_rows", self_read=False, all_cells=False,
            stationary=False, separate=False, nearest=False):
    canonical, turn, bg, payload, marker, ac, mask, banks = info
    h, w = canonical.shape
    out = np.full_like(canonical, bg)
    hits = np.flatnonzero(mask[:, ac])
    stop = int(hits.min() if nearest else hits.max())
    out[:stop+1, ac] = marker
    out[mask[:, ac], ac] = payload
    rr, cc = np.indices(mask.shape)
    selections = []
    for side in (0, 1):
        senders = banks[side if self_read else 1-side]
        rows = {r for obj in senders for r in obj[reading]}
        selected = mask & (cc < ac if side == 0 else cc > ac)
        if not all_cells:
            selected &= np.isin(rr, sorted(rows))
        selections.append(selected)
        if stationary:
            out[selected] = payload
            continue
        groups = _components(selected) if separate else [list(map(tuple, np.argwhere(selected)))]
        for group in groups:
            if not group:
                continue
            ys, xs = np.asarray(group, dtype=int).T
            dy = -int(ys.min())
            dx = -int(xs.min()) if side == 0 else w-1-int(xs.max())
            ty, tx = ys+dy, xs+dx
            _require(bool(np.all(out[ty, tx] == bg)), "corner packing collision")
            out[ty, tx] = payload
    return np.rot90(out, -turn).copy(), selections


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the complete output from the input alone, without changing the input."""
    return _render(_read(input_grid))[0]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(value):
    info = _read(value)
    canonical, turn, bg, payload, marker, ac, mask, banks = info
    out, selections = _render(info)
    hits = np.flatnonzero(mask[:, ac])
    gap = any(not mask[r, ac] for r in range(int(hits.min()), int(hits.max())+1))
    active_counts = tuple(sum(len(o['active']) for o in bank) for bank in banks)
    productive = []
    multiline_productive = 0
    for side, bank in enumerate(banks):
        target = mask[:, ac+1:] if side == 0 else mask[:, :ac]
        count = 0
        for obj in bank:
            for hole in obj['active']:
                rows = {obj['origin'][0]+r for r, c in hole}
                occupied_rows = {r for r in rows if np.any(target[r])}
                count += bool(occupied_rows)
                multiline_productive += len(occupied_rows) >= 2
        productive.append(count)
    decoys = sum(len(o['decoy']) for bank in banks for o in bank)
    opens = sum(bool(o['open_rows']) for bank in banks for o in bank)
    pieces = tuple(len(_components(s)) for s in selections)
    # Holes enclosed in the unsplit shape but opened by removing the axis.
    ys, xs = np.nonzero(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    whole = mask[y0:ys.max()+1, x0:xs.max()+1]
    bank_holes = {tuple((r+o['origin'][0], c+o['origin'][1]))
                  for bank in banks for o in bank for hole in o['active']+o['decoy']
                  for r, c in hole}
    axis_sealed = sum(all(x0+c != ac for r, c in hole) and
                      any((y0+r, x0+c) not in bank_holes for r, c in hole)
                      for hole in _holes(whole))
    mutants = {
        'ignore_B': not np.array_equal(_render(info, all_cells=True)[0], out),
        'ignore_A': not np.array_equal(_render(info, stationary=True)[0], out),
        'self_read': not np.array_equal(_render(info, self_read=True)[0], out),
        'all_holes': not np.array_equal(_render(info, reading='all_hole_rows')[0], out),
        'no_enclosure': not np.array_equal(_render(info, reading='naive_rows')[0], out),
        'nearest': not np.array_equal(_render(info, nearest=True)[0], out),
    }
    # Separate packing can overlap; that is already a failure of that alternative.
    try:
        mutants['separate'] = not np.array_equal(_render(info, separate=True)[0], out)
    except AssertionError:
        mutants['separate'] = True
    return dict(shape=tuple(canonical.shape), orientation=turn,
                roles=(bg, payload, marker), axis=ac, gap=gap,
                short_endpoint=int(hits.max()) < int(ys.max()),
                active_counts=active_counts, decoys=decoys, opens=opens,
                productive_active_counts=tuple(productive),
                multiline_productive=multiline_productive,
                axis_sealed=axis_sealed, selected_counts=tuple(int(s.sum()) for s in selections),
                fragment_counts=pieces, bank_components=tuple(len(b) for b in banks),
                mutants=mutants)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _minimum(kind):
    return 4 if kind in ('active', 'axial_open') else 2 if kind == 'solid' else 3


def _make_bank(rng, kinds, budget, width, rich, detached):
    """Build connected chamber modules, optionally linked only via the input axis."""
    base = sum(_minimum(k) for k in kinds) + len(kinds)-1
    spare = min(budget-base, int(rng.integers(1, 10 if rich else 5)))
    extras = rng.multinomial(spare, np.full(2*len(kinds)-1, 1/(2*len(kinds)-1)))
    heights = [_minimum(k)+int(extras[2*i]) for i, k in enumerate(kinds)]
    gaps = [1+int(extras[2*i+1]) for i in range(len(kinds)-1)]
    length = sum(heights)+sum(gaps)
    mask = np.zeros((length, width), dtype=bool)
    if not detached:
        mask[:, 0] = True
    protected, row = set(), 0
    for i, (kind, height) in enumerate(zip(kinds, heights)):
        offset = int(rng.integers(1, min(3, width-2))) if kind == 'recessed' else 0
        room_width = int(rng.integers(offset+3, width+1)) if kind != 'solid' else int(rng.integers(1, width+1))
        view = mask[row:row+height]
        view[:, offset:room_width] = True
        # Top and bottom arms connect a recessed body to its leading protrusion.
        view[0, :room_width] = view[-1, :room_width] = True
        view[:, 0] = True
        # Add outline material before carving doorways, so a tail cannot reseal one.
        if rich and rng.random() < .45:
            y = int(rng.choice([0, height-1]))
            view[y, :width] = True
        if kind != 'solid':
            top = 2 if rich and height >= 5 and rng.random() < .35 else 1
            bottom = height-2 if rich and height-top >= 4 and rng.random() < .35 else height-1
            for y in range(top, bottom):
                back = int(rng.integers(offset+2, room_width)) if rich and rng.random() < .55 else room_width-1
                view[y, offset+1:back] = False
            if kind == 'open':
                # Open from the back or through a rim, retaining the leading wall.
                opening = int(rng.integers(3))
                if opening == 0:
                    y = int(rng.integers(top, bottom))
                    view[y, offset+1:room_width] = False
                else:
                    if opening == 1:
                        view[:top, offset+1] = False
                    else:
                        view[bottom:, offset+1] = False
            elif kind == 'axial_open':
                # A doorway sealed by the axis in the original connected shape.
                y = int(rng.integers(top, bottom))
                view[y, 0] = False
                protected.add(row+y)
        row += height
        if i < len(gaps):
            row += gaps[i]
    return mask, protected


def _make_pair(rng, left_kinds, right_kinds, roles, turn, rich, witness, cover_apertures=False):
    bases = [sum(_minimum(k) for k in ks)+len(ks)-1 for ks in (left_kinds, right_kinds)]
    h = int(rng.integers(max(bases)+3, 31))
    widths = [int(rng.integers(5 if 'recessed' in ks else 3, 11 if rich else 8))
              for ks in (left_kinds, right_kinds)]
    detached = [False, False] if witness else [bool(rng.random() < .35) for _ in range(2)]
    banks = [_make_bank(rng, ks, h-2, width, rich, detach)
             for ks, width, detach in zip((left_kinds, right_kinds), widths, detached)]
    if witness or cover_apertures:
        # Give every gate a visible receiver. Extra rows are solid, varying-width
        # branches, not extra cavities or an extra solving convention.
        length = max(bank.shape[0] for bank, protected in banks)
        padded = []
        for (bank, guarded), width in zip(banks, widths):
            extra = length-bank.shape[0]
            top = int(rng.integers(extra+1))
            expanded = np.zeros((length, width), dtype=bool)
            expanded[top:top+bank.shape[0]] = bank
            for y in list(range(top)) + list(range(top+bank.shape[0], length)):
                expanded[y, :int(rng.integers(1, width+1))] = True
            # A narrow connector leaves top/bottom doorways exterior-connected.
            for y in (top-1, top+bank.shape[0]):
                if 0 <= y < length:
                    expanded[y] = False
                    expanded[y, 0] = True
            padded.append((expanded, {r+top for r in guarded}))
        banks = padded
        start = int(rng.integers(1, h-length))
        starts = [start, start]
    else:
        starts = [int(rng.integers(1, h-bank.shape[0])) for bank, protected in banks]
    w = int(rng.integers(sum(widths)+3, 31))
    ac = int(rng.integers(widths[0]+1, w-widths[1]-1))
    mask = np.zeros((h, w), dtype=bool)
    protected = set()
    for side, ((bank, guarded), start) in enumerate(zip(banks, starts)):
        rr, xx = np.nonzero(bank)
        cc = ac-1-xx if side == 0 else ac+1+xx
        mask[rr+start, cc] = True
        protected.update(start+r for r in guarded)
    first = min(starts)
    last = max(start+bank.shape[0]-1 for (bank, guarded), start in zip(banks, starts))
    mask[first:last+1, ac] = True
    # Delete spine cells only when the visible source stays connected. In particular,
    # a bank wall can detour around an internal axial gap.
    candidates = list(range(first, last+1))
    rng.shuffle(candidates)
    target = int(rng.integers(1, 6)) if witness or rich else int(rng.integers(0, 4))
    removed = 0
    for r in candidates:
        if removed >= target:
            break
        if r in protected or mask[:, ac].sum() <= 2:
            continue
        trial = mask.copy()
        trial[r, ac] = False
        if len(_components(trial)) == 1:
            mask = trial
            removed += 1
    # Sometimes the inward guide ends before the payload's overall bounding box.
    if witness or rng.random() < .45:
        for r in range(last, max(first, last-2)-1, -1):
            if r in protected or not mask[r, ac] or mask[:, ac].sum() <= 2:
                continue
            trial = mask.copy()
            trial[r, ac] = False
            if len(_components(trial)) == 1:
                mask = trial
    bg, payload, marker = roles
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[mask] = payload
    grid[0, ac] = marker
    grid = np.rot90(grid, -turn).copy()
    return {'input': grid, 'output': solve(grid)}


def _random_kinds(rng, count, max_active=2):
    kinds = list(map(str, rng.choice(['active', 'open', 'recessed', 'solid', 'axial_open'], count)))
    while kinds.count('active') > max_active:
        kinds[kinds.index('active')] = 'open'
    return kinds


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create one coherent puzzle; all conventions are fixed or visible in its input."""
    a = _integer(num_train, 'num_train', 2, 10)
    b = _integer(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    vary_palette = bool(rng.random() < .65)
    rich = bool(rng.random() < .65)
    orientation_mode = int(rng.integers(3))
    base_turn = int(rng.integers(4))
    train_turns = [base_turn]*a
    test_turns = [base_turn]*b
    if orientation_mode == 1:
        train_turns[1] = (base_turn+1) % 4
        for i in range(2, a):
            train_turns[i] = int(rng.integers(4))
        test_turns = [int(rng.integers(4)) for _ in range(b)]
    elif orientation_mode == 2:
        test_turns = [(base_turn+int(rng.integers(1, 4))) % 4 for _ in range(b)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count, turns in (('train', a, train_turns), ('test', b, test_turns)):
        for i in range(count):
            witness = split == 'train' and i < 2
            if vary_palette:
                roles = tuple(map(int, rng.choice(10, 3, replace=False)))
            for attempt in range(80):
                if split == 'train' and i == 0:
                    left, right = ['active', 'open'], ['active', 'recessed']
                elif witness:
                    left, right = ['active', 'axial_open', 'active'], ['active', 'open', 'recessed']
                elif split == 'train':
                    left = _random_kinds(rng, int(rng.integers(1, 4)))
                    right = _random_kinds(rng, int(rng.integers(1, 4)))
                else:
                    # The test combines three or four separated transmitting apertures.
                    left = ['active']*int(rng.integers(3, 5))
                    if rng.random() < .6:
                        left.append(str(rng.choice(['open', 'recessed', 'axial_open', 'solid'])))
                    right = _random_kinds(rng, int(rng.integers(2, 6)))
                rng.shuffle(left)
                rng.shuffle(right)
                if rng.random() < .5:
                    left, right = right, left
                pair = _make_pair(rng, left, right, roles, turns[i], rich, witness, split == 'test')
                f = _features(pair['input'])
                if _key(pair['input']) in seen:
                    continue
                if split == 'train' and max(f['active_counts']) > 2:
                    continue
                if witness:
                    needs = ['ignore_B', 'ignore_A', 'self_read', 'all_holes', 'no_enclosure', 'nearest']
                    if not all(f['mutants'][name] for name in needs):
                        continue
                    if i == 1 and not (f['mutants']['separate'] and f['axis_sealed']):
                        continue
                    if f['productive_active_counts'] != f['active_counts'] or not f['multiline_productive']:
                        continue
                if split == 'test':
                    if not 3 <= max(f['active_counts']) <= 4:
                        continue
                    if f['productive_active_counts'] != f['active_counts']:
                        continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction could not obtain informative distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Validate pixel-derived topology, selection, conservation, and demonstrations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid pair counts')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions must be preserved')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _read(inp)
            canonical, turn, bg, payload, marker, ac, mask, banks = info
            for bank in banks:
                bands = sorted((min(r for r, c in o['cells']), max(r for r, c in o['cells']))
                               for o in bank)
                _require(all(after[0] > before[1] for before, after in zip(bands, bands[1:])),
                         'bank components must have disjoint longitudinal bands')
                for obj in bank:
                    _require(all(obj['contacts']), 'partially leading cavities are outside the family')
            expected, selections = _render(info)
            _require(np.array_equal(out, expected), 'output violates cross-reading rule')
            _require(set(map(int, np.unique(out))) <= {bg, payload, marker}, 'unexpected output color')
            expected_count = int(mask[:, ac].sum()) + sum(int(s.sum()) for s in selections)
            _require(np.count_nonzero(out == payload) == expected_count, 'selected payload conservation failed')
            canonical_out = np.rot90(out, turn)
            hits = np.flatnonzero(mask[:, ac])
            _require(np.all(canonical_out[hits, ac] == payload), 'axial payload moved')
            _require(np.all(canonical_out[hits.max()+1:, ac] == bg), 'guide extends past farthest axial hit')
            for side, selected in enumerate(selections):
                if not np.any(selected):
                    continue
                ys, xs = np.nonzero(selected)
                dy = -int(ys.min())
                dx = -int(xs.min()) if side == 0 else canonical.shape[1]-1-int(xs.max())
                _require(np.all(canonical_out[ys+dy, xs+dx] == payload), 'collective translation failed')
            records[split].append(_features(inp))
    train = records['train']
    for name in ('ignore_B', 'ignore_A', 'self_read', 'all_holes', 'no_enclosure', 'nearest', 'separate'):
        _require(any(f['mutants'][name] for f in train), f'missing demonstration against {name}')
    _require(any(f['axis_sealed'] for f in train), 'no cut-before-enclosure witness')
    _require(any(max(f['productive_active_counts']) >= 2 for f in train), 'no multi-aperture demonstration')
    _require(any(f['multiline_productive'] for f in train), 'no multirow transmission demonstration')
    _require(any(f['short_endpoint'] for f in train), 'no axial endpoint demonstration')
    _require(all(max(f['active_counts']) <= 2 for f in train), 'training aperture count exceeds schedule')
    _require(all(3 <= max(f['productive_active_counts']) <= 4 and
                 f['productive_active_counts'] == f['active_counts']
                 for f in records['test']), 'productive test aperture-count shift missing')


def _diagnostic():
    """Hand-specified example with an independently enumerated expected output."""
    source = ('0000002000000/0000007000000/0007777000000/0007070000000/'
              '0007077777000/0007777707000/0000007707000/0000007777000/'
              '0000007000000/0000000000000')
    target = ('7770002000777/0000007000000/0000007000000/0000002000000/'
              '0000007000000/0000007000000/0000007000000/0000007000000/'
              '0000007000000/0000000000000')
    decode = lambda text: np.array([[int(c) for c in row] for row in text.split('/')], dtype=np.int8)
    return decode(source), decode(target)


def self_test():
    """Bounded 42-seed structural audit, counterexamples, and metamorphic checks."""
    inp, expected = _diagnostic()
    _require(np.array_equal(solve(inp), expected), 'hand-specified cross-reading diagnostic failed')
    palettes, shapes, orientations, structures = set(), set(), set(), set()
    active_counts, bank_counts, selected_counts = set(), set(), set()
    mutations = {k: 0 for k in _features(inp)['mutants']}
    signatures, palette_modes, direction_modes = set(), set(), set()
    total, open_count, decoy_count, sealed_count, gap_count, fragmented = 0, 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        own_palettes, own_directions = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                total += 1
                f = _features(pair['input'])
                shapes.add(f['shape'])
                palettes.add(f['roles'])
                orientations.add(f['orientation'])
                own_palettes.add(f['roles'])
                own_directions.add(f['orientation'])
                canonical = _read(pair['input'])[0]
                structures.add((canonical.shape, (canonical == f['roles'][1]).tobytes(), f['axis']))
                active_counts.update(f['active_counts'])
                bank_counts.update(f['bank_components'])
                selected_counts.update(f['selected_counts'])
                open_count += f['opens']
                decoy_count += f['decoys']
                sealed_count += f['axis_sealed']
                gap_count += f['gap']
                fragmented += max(f['fragment_counts']) > 1
                for name, fails in f['mutants'].items():
                    mutations[name] += bool(fails)
                original = pair['input'].copy()
                _require(np.array_equal(solve(original), pair['output']), 'solver mismatch')
                _require(np.array_equal(original, pair['input']), 'solver changed input')
        palette_modes.add(len(own_palettes) > 1)
        direction_modes.add(len(own_directions) > 1)
        _require(max(max(f['productive_active_counts']) for f in map(lambda p: _features(p['input']), puzzle['test'])) >
                 max(max(f['active_counts']) for f in map(lambda p: _features(p['input']), puzzle['train'])),
                 'test aperture-count shift missing')
    _require(len(signatures) == 42, 'cross-seed duplication')
    _require(len(structures) >= 180 and len(shapes) >= 45, 'structural diversity collapsed')
    _require(len(palettes) >= 40 and orientations == set(range(4)), 'roles/directions collapsed')
    _require({False, True} == palette_modes == direction_modes, 'sampling schedules collapsed')
    _require({0, 1, 2, 3, 4} <= active_counts, 'aperture counts collapsed')
    _require(max(bank_counts) >= 3 and len(selected_counts) >= 20, 'bank/selection variation collapsed')
    _require(min(mutations.values()) >= 42, 'shortcut counterexamples missing')
    _require(min(open_count, decoy_count, sealed_count, gap_count, fragmented) >= 42, 'topology coverage collapsed')
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn)), 'rotation equivariance failed')
        lookup = np.array([8, 4, 1, 5, 9, 6, 2, 0, 7, 3], dtype=np.int8)
        _require(np.array_equal(solve(lookup[np.rot90(inp, turn)]), lookup[np.rot90(expected, turn)]),
                 'visible-role recoloring failed')
    p = generate_puzzle(271828, 6, 2)
    q = generate_puzzle(271828, 6, 2)
    _require(all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
                 for x, y in zip(p[s], q[s]) for k in ('input', 'output')), 'nondeterministic puzzle')
    for a, b in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(314159, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad label accepted')
    return dict(seeds=42, pairs=total, canonical_shapes=len(shapes),
                color_normalized_structures=len(structures), palettes=len(palettes),
                orientations=sorted(orientations), active_counts=sorted(active_counts),
                bank_component_counts=sorted(bank_counts), selected_counts=len(selected_counts),
                shortcut_failures=mutations, open_components=open_count,
                decoy_holes=decoy_count, axis_sealed_holes=sealed_count,
                gapped_guides=gap_count, fragmented_selections=fragmented,
                independent_semantic_verification=False)
