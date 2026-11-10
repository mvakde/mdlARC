"""Wallpaper folded along a crease: input-visible motifs, ownership and phase.

Standard library + NumPy only. No parent fixtures, file reads, or generation
state are used by solve. Canonical coordinates point away from the swatch bank.
"""
from itertools import product
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _full_box(mask):
    return (2 <= mask.shape[0] <= 4 and 2 <= mask.shape[1] <= 4
            and 2 <= int(mask.sum()) < mask.size
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            found = remaining & {(r-1, c), (r+1, c), (r, c-1), (r, c+1)}
            remaining -= found
            stack.extend(found)
    return count


def _marker_sites(profile, width, left):
    h = len(profile)
    r, c = np.indices((h, width))
    side = c < profile[:, None] if left else c > profile[:, None]
    near = np.zeros((h, width), dtype=bool)
    for y, x in enumerate(profile):
        near[max(0, y-1):min(h, y+2), max(0, x-1):min(width, x+2)] = True
    return [tuple(map(int, p)) for p in np.argwhere(side & ~near)]


def _decode_canonical(a, rotation):
    """Only called after the unique divider and bank direction are identified."""
    def need(ok, message):
        if not bool(ok):
            raise ValueError(message)

    frame = int(a[0, 0])
    need(np.all(a[[0, -1], :] == frame) and np.all(a[:, [0, -1]] == frame),
         "broken outer frame")
    dividers = [i for i in range(1, a.shape[0]-1) if np.all(a[i] == frame)]
    need(len(dividers) == 1, "need exactly one divider")
    divider = dividers[0]
    bank, fabric = a[1:divider, 1:-1], a[divider+1:-1, 1:-1]
    h, w = fabric.shape
    need(2 <= bank.shape[0] < h and h >= 9 and w >= 11, "ambiguous bank or small fabric")
    colors, counts = np.unique(fabric, return_counts=True)
    need(len(colors) == 3, "fabric needs background, cut and one dot")
    bg = int(colors[np.argmax(counts)])
    need(int(counts.max()) > fabric.size//2, "background must be a strict majority")
    fg = [int(x) for x in colors if x != bg]
    dots = [x for x in fg if np.count_nonzero(fabric == x) == 1]
    need(len(dots) == 1, "dot must be unique")
    favored = dots[0]
    cut = next(x for x in fg if x != favored)
    cut_mask = fabric == cut
    need(np.all(cut_mask.sum(axis=1) == 1), "one cut cell per row is required")
    profile = np.argmax(cut_mask, axis=1)
    need(np.all((2 <= profile) & (profile <= w-3))
         and np.all(np.abs(np.diff(profile)) <= 1), "cut is outside the digital profile family")
    marker = tuple(map(int, np.argwhere(fabric == favored)[0]))
    left = bool(marker[1] < profile[marker[0]])
    need(marker in _marker_sites(profile, w, left), "dot touches the crease")
    accents = sorted(set(map(int, np.unique(bank))) - {bg})
    need(len(accents) == 2 and favored in accents, "bank needs two swatches including dot color")
    other = next(x for x in accents if x != favored)
    need(len({frame, bg, cut, favored, other}) == 5, "five color roles must be distinct")
    tiles, boxes = {}, {}
    for color in accents:
        coords = np.argwhere(bank == color)
        y0, x0 = map(int, coords.min(axis=0))
        y1, x1 = map(int, coords.max(axis=0)+1)
        patch = bank[y0:y1, x0:x1]
        need(np.all((patch == bg) | (patch == color)), "mixed-color swatch")
        mask = patch == color
        need(_full_box(mask), "swatch must be a non-solid full bounding box of size 2..4")
        tiles[color] = mask.copy()
        boxes[color] = (y0, y1, x0, x1)
    need(tiles[favored].shape == tiles[other].shape, "swatch periods must agree")
    ordered = sorted(accents, key=lambda x: boxes[x][2])
    need(boxes[ordered[1]][2] - boxes[ordered[0]][3] >= 2, "swatches need a two-column gap")
    return dict(canonical=a, rotation=rotation, divider=divider, bank=bank,
                fabric=fabric, frame=frame, background=bg, cut_color=cut,
                profile=profile.copy(), marker=marker, favored=favored, other=other,
                left=left, tiles=tiles, boxes=boxes, bank_left=ordered[0])


def _decode(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    candidates = []
    for rotation in range(4):
        b = np.rot90(a, rotation)
        dividers = [i for i in range(1, b.shape[0]-1) if np.all(b[i] == frame)]
        if len(dividers) == 1:
            divider = dividers[0]
            if 2 <= divider-1 < b.shape[0]-divider-2:
                candidates.append((b, rotation))
    if len(candidates) != 1:
        raise ValueError("no unique bank-above-fabric orientation")
    return _decode_canonical(*candidates[0])


def _render_fabric(d, mode="correct", phase=None, signs=(1, 1)):
    """Vectorized rendering; alternate modes are concrete teaching checks."""
    h, w = d['fabric'].shape
    p, q = d['tiles'][d['favored']].shape
    phase = (p//2, q//2) if phase is None else phase
    r, c = np.indices((h, w))
    spine = d['profile'][:, None]
    distance = abs(c-spine)
    if mode == 'fixed_origin':
        distance = abs(c-int(d['profile'][0]))
    elif mode == 'signed':
        distance = c-spine
    if mode == 'phase_none':
        phase = (0, 0)
    u, v = (signs[0]*r+phase[0]) % p, (signs[1]*distance+phase[1]) % q
    favored, other = d['favored'], d['other']
    if mode == 'bank_left':
        favored = d['bank_left']
        other = next(x for x in d['tiles'] if x != favored)
    ma, mb = d['tiles'][favored], d['tiles'][other]
    if mode == 'transpose':
        ma, mb = ma.T, mb.T
    am, bm = ma[u, v], mb[u, v]
    if mode == 'solid':
        am, bm = np.ones((h, w), bool), np.ones((h, w), bool)
    choose_a = (c < spine) if d['left'] else (c > spine)
    if mode == 'always_left':
        choose_a = c < spine
    out = np.full((h, w), d['background'], dtype=np.int8)
    out[choose_a & am] = favored
    out[~choose_a & bm] = other
    if mode == 'union':
        out[:] = d['background']
        out[bm] = other
        out[am] = favored
    elif mode == 'favored_all':
        out[:] = d['background']
        out[am] = favored
    elif mode == 'other_all':
        out[:] = d['background']
        out[bm] = other
    seam = c == spine
    out[seam] = d['background']
    out[seam & bm] = other
    out[seam & am] = favored
    if mode == 'reverse_priority':
        out[seam & bm] = other
    elif mode == 'drop_other_seam':
        out[seam & ~am] = d['background']
    elif mode == 'blank_seam':
        out[seam] = d['background']
    elif mode == 'solid_seam':
        out[seam] = d['cut_color']
    return out


def _finish(d):
    a = d['canonical'].copy()
    a[d['divider']+1:-1, 1:-1] = _render_fabric(d)
    return np.rot90(a, -d['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, periods, orientation, ownership and phase from input."""
    return _finish(_decode(input_grid))


def _phase_candidates(p, q):
    # Deduplicate algebraically identical maps, including +/- for period two.
    intended = (tuple((r+p//2) % p for r in range(p)),
                tuple((c+q//2) % q for c in range(q)))
    seen = {intended}
    for sr, sc, a, b in product((1, -1), (1, -1), range(p), range(q)):
        signature = (tuple((sr*r+a) % p for r in range(p)),
                     tuple((sc*c+b) % q for c in range(q)))
        if signature not in seen:
            seen.add(signature)
            yield (a, b), (sr, sc)


def _teaching_failures(pairs):
    data = [_decode(pair['input']) for pair in pairs]
    targets = [np.rot90(np.asarray(pair['output']), d['rotation'])[d['divider']+1:-1, 1:-1]
               for pair, d in zip(pairs, data)]
    p, q = data[0]['tiles'][data[0]['favored']].shape
    failures = []
    modes = ['union', 'favored_all', 'other_all', 'always_left', 'bank_left',
             'reverse_priority', 'drop_other_seam', 'blank_seam', 'solid_seam',
             'solid', 'fixed_origin', 'phase_none']
    if q > 2:
        modes.append('signed')
    if p == q:
        modes.append('transpose')
    for mode in modes:
        if all(np.array_equal(target, _render_fabric(d, mode)) for d, target in zip(data, targets)):
            failures.append(mode)
    for phase, signs in _phase_candidates(p, q):
        if all(np.array_equal(target, _render_fabric(d, phase=phase, signs=signs))
               for d, target in zip(data, targets)):
            failures.append(('phase', phase, signs))
    return failures


def _classic_masks(shape):
    p, q = shape
    r, c = np.indices(shape)
    masks = []
    corners = np.zeros(shape, bool)
    corners[np.ix_([0, p-1], [0, q-1])] = True
    masks.append(corners)
    masks.append((r == 0) | (r == p-1) | (c == 0) | (c == q-1))
    masks.append((r == p//2) | (c == q//2))
    masks.append((r == p//2) | (c == 0) | (c == q-1))
    if p == q:
        masks.extend([r == c, (r == c) | (r+c == p-1)])
    return [m for m in masks if _full_box(m)]


def _sample_mask(rng, shape, center_column=None):
    p, q = shape
    classics = [m for m in _classic_masks(shape)
                if center_column is None or np.array_equal(m[:, q//2], center_column)]
    if classics and rng.random() < .25:
        return classics[int(rng.integers(len(classics)))].copy()
    for _ in range(128):
        mask = rng.random(shape) < float(rng.uniform(.15, .85))
        if center_column is not None:
            mask[:, q//2] = center_column
        if _full_box(mask):
            return mask
    # A mixed constrained center column guarantees a hole. Fill all other
    # columns to meet the bounding extrema without changing that evidence.
    mask = np.ones(shape, bool)
    if center_column is None:
        mask[int(rng.integers(p)), int(rng.integers(q))] = False
    else:
        mask[:, q//2] = center_column
    _require(_full_box(mask), 'mask construction failed')
    return mask


def _mixed_column(rng, p):
    # Every nonconstant bit vector is eligible, not one fixed evidence stamp.
    code = int(rng.integers(1, 2**p-1))
    return np.array([(code >> k) & 1 for k in range(p)], dtype=bool)


def _profile(rng, h, w, kind, direction=1):
    if kind == 'straight':
        return np.full(h, int(rng.integers(2, w-2)), dtype=int)
    if kind == 'ramp':
        amplitude = int(rng.integers(1, min(5, w-4)))
        low = int(rng.integers(2, w-2-amplitude))
        changes = np.zeros(h, dtype=int)
        indices = rng.choice(np.arange(1, h), amplitude, replace=False)
        changes[indices] = direction
        start = low if direction > 0 else low+amplitude
        return start + np.cumsum(changes)
    legs = 2 if kind == 'bump' else int(rng.choice([3, 4, 5]))
    amplitude = int(rng.integers(1, min(2, (h-1)//legs)+1))
    start = int(rng.integers(2+amplitude, w-2-amplitude))
    positions = np.sort(rng.choice(np.arange(1, h), legs*amplitude, replace=False))
    changes = np.zeros(h, dtype=int)
    direction = int(rng.choice([-1, 1]))
    for i, position in enumerate(positions):
        changes[position] = direction * (1 if (i//amplitude) % 2 == 0 else -1)
    return start + np.cumsum(changes)


def _palette(rng, plan):
    if plan['palette_policy'] == 'fixed' or rng.random() < .3:
        return plan['palette']
    return tuple(map(int, rng.choice(10, 5, replace=False)))


def _make_pair(rng, plan, kind, left, evidence=None, test=False):
    p, q = plan['period']
    bank_h = p + int(rng.integers(0, 4))
    maximum_h = min(20, 30-bank_h-3)
    if plan['regime'] == 'deeper':
        h = int(rng.integers(17, maximum_h+1)) if test else int(rng.integers(9, 14))
    else:
        h = int(rng.integers(9, maximum_h+1))
    w = int(rng.integers(max(11, 2*q+2), 29))
    direction = -plan['direction'] if test and plan['regime'] == 'reverse' else plan['direction']
    profile = _profile(rng, h, w, kind, direction)
    frame, bg, cut, ca, cb = _palette(rng, plan)
    favored, other = (ca, cb) if rng.random() < .5 else (cb, ca)
    if evidence is None:
        am, bm = _sample_mask(rng, (p, q)), _sample_mask(rng, (p, q))
    else:
        column = _mixed_column(rng, p)
        am = _sample_mask(rng, (p, q), column)
        bm = _sample_mask(rng, (p, q), column if evidence == 'equal' else ~column)
    tiles = {favored: am, other: bm}
    bank = np.full((bank_h, w), bg, dtype=np.int8)
    gap = int(rng.integers(2, w-2*q+1))
    first_x = int(rng.integers(0, w-2*q-gap+1))
    second_x = first_x+q+gap
    ordering = (ca, cb) if rng.random() < .5 else (cb, ca)
    boxes = {}
    for color, x in zip(ordering, (first_x, second_x)):
        y = int(rng.integers(0, bank_h-p+1))
        bank[y:y+p, x:x+q][tiles[color]] = color
        boxes[color] = (y, y+p, x, x+q)
    sites = _marker_sites(profile, w, left)
    marker = sites[int(rng.integers(len(sites)))]
    fabric = np.full((h, w), bg, dtype=np.int8)
    fabric[np.arange(h), profile] = cut
    fabric[marker] = favored
    divider = bank_h+1
    canonical = np.full((bank_h+h+3, w+2), frame, dtype=np.int8)
    canonical[1:divider, 1:-1] = bank
    canonical[divider+1:-1, 1:-1] = fabric
    # Metadata here is local construction state. solve reads none of it.
    d = dict(canonical=canonical, rotation=0, divider=divider, bank=bank, fabric=fabric,
             frame=frame, background=bg, cut_color=cut, profile=profile, marker=marker,
             favored=favored, other=other, left=left, tiles=tiles, boxes=boxes,
             bank_left=ordering[0])
    out = _finish(d)
    rotation = plan['rotation'] if rng.random() < .65 else int(rng.integers(4))
    return {'input': np.rot90(canonical, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}


def _opposite_dot(pair, rng):
    d = _decode(pair['input'])
    a = d['canonical'].copy()
    fabric = a[d['divider']+1:-1, 1:-1]
    fabric[d['marker']] = d['background']
    sites = _marker_sites(d['profile'], fabric.shape[1], not d['left'])
    fabric[sites[int(rng.integers(len(sites)))]] = d['favored']
    inp = np.rot90(a, -d['rotation']).copy()
    return {'input': inp, 'output': solve(inp)}


def _motif_pair_key(pair):
    d = _decode(pair['input'])
    return tuple(sorted(_key(mask) for mask in d['tiles'].values()))


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'),
                                   (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(f'{name} must be in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    plan = dict(period=tuple(map(int, rng.integers(2, 5, size=2))),
                palette=tuple(map(int, rng.choice(10, 5, replace=False))),
                palette_policy=str(rng.choice(['fixed', 'variable'], p=[.3, .7])),
                regime=str(rng.choice(['more_turns', 'reverse', 'deeper'])),
                direction=int(rng.choice([-1, 1])), rotation=int(rng.integers(4)))
    first_left = bool(rng.integers(2))
    # Sample whole readable objects until the minimal two examples teach the
    # family. No seam pixels or motifs are edited after a candidate is drawn.
    for _ in range(192):
        first = _make_pair(rng, plan, 'straight', first_left, 'equal')
        second = _make_pair(rng, plan, 'ramp', not first_left, 'complement')
        if (_key(first['input']) != _key(second['input'])
                and first['input'].shape != second['input'].shape
                and not _teaching_failures([first, second])):
            break
    else:
        raise RuntimeError('could not construct informative demonstrations')
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(first['input']), _key(second['input'])}
    if nt >= 3:
        twin = _opposite_dot(puzzle['train'][int(rng.integers(2))], rng)
        puzzle['train'].append(twin)
        seen.add(_key(twin['input']))
    for i in range(len(puzzle['train']), nt):
        kinds = ['straight', 'ramp'] if plan['regime'] == 'reverse' else ['straight', 'ramp', 'bump']
        for _ in range(96):
            pair = _make_pair(rng, plan, str(rng.choice(kinds)), bool(rng.integers(2)))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('unique training input construction exhausted')
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    train_motifs = {_motif_pair_key(pair) for pair in puzzle['train']}
    test_motifs = set()
    for i in range(nq):
        kind = 'wave' if plan['regime'] == 'more_turns' else 'ramp'
        for _ in range(192):
            pair = _make_pair(rng, plan, kind, bool(rng.integers(2)), test=True)
            mk = _motif_pair_key(pair)
            if _key(pair['input']) not in seen and mk not in train_motifs | test_motifs:
                break
        else:
            raise RuntimeError('unique test motif construction exhausted')
        puzzle['test'].append(pair)
        seen.add(_key(pair['input']))
        test_motifs.add(mk)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, periods, states, selected_sides = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            d = _decode(inp)
            p, q = d['tiles'][d['favored']].shape
            periods.add((p, q))
            canonical_out = np.rot90(out, d['rotation'])
            answer = canonical_out[d['divider']+1:-1, 1:-1]
            _require(np.array_equal(canonical_out[:d['divider']+1], d['canonical'][:d['divider']+1]),
                     'bank, divider or upper frame changed')
            _require(np.array_equal(canonical_out[:, [0, -1]], d['canonical'][:, [0, -1]])
                     and np.array_equal(canonical_out[-1], d['canonical'][-1]), 'frame changed')
            _require(np.array_equal(solve(inp), out), 'solver/answer mismatch')
            h, w = answer.shape
            r, c = np.indices((h, w))
            cut = c == d['profile'][:, None]
            _require(_components(~cut) == 2, 'crease must separate two four-connected regions')
            choose_a = c < d['profile'][:, None] if d['left'] else c > d['profile'][:, None]
            _require(set(map(int, np.unique(answer))) == {d['background'], d['favored'], d['other']},
                     'both motifs and background must visibly contribute')
            # Check occupancy/foreground/background cell semantics directly.
            # This is authored invariant checking, not an independent solver.
            for y in range(h):
                for x in range(w):
                    u = (y+p//2) % p
                    v = (abs(x-int(d['profile'][y]))+q//2) % q
                    av = bool(d['tiles'][d['favored']][u, v])
                    bv = bool(d['tiles'][d['other']][u, v])
                    if cut[y, x]:
                        expected = d['favored'] if av else d['other'] if bv else d['background']
                        if split == 'train':
                            states.add((av, bv))
                    elif choose_a[y, x]:
                        expected = d['favored'] if av else d['background']
                    else:
                        expected = d['other'] if bv else d['background']
                    _require(int(answer[y, x]) == expected, 'registered occupancy invariant')
            if split == 'train':
                selected_sides.add(d['left'])
    _require(len(periods) == 1, 'generated puzzle should use a common visible swatch period')
    _require(states == set(product((False, True), repeat=2)), 'training seam truth table incomplete')
    _require(selected_sides == {False, True}, 'training does not expose both selected sides')
    _require(not _teaching_failures(puzzle['train']), 'a documented shortcut survives training')


def _turns(profile):
    increments = [int(x) for x in np.diff(profile) if x != 0]
    return sum(a != b for a, b in zip(increments, increments[1:]))


def _relocate_swatches(pair):
    """Shift the bank's two boxes together, preserving every extracted motif."""
    d = _decode(pair['input'])
    a = d['canonical'].copy()
    bank = a[1:d['divider'], 1:-1]
    bank[:] = d['background']
    leftmost = min(box[2] for box in d['boxes'].values())
    rightmost = max(box[3] for box in d['boxes'].values())
    shift = 1 if rightmost < bank.shape[1] else -1 if leftmost > 0 else 0
    for color, box in d['boxes'].items():
        y0, y1, x0, x1 = box
        vertical = 1 if y1 < bank.shape[0] else -1 if y0 > 0 else 0
        bank[y0+vertical:y1+vertical, x0+shift:x1+shift][d['tiles'][color]] = color
    return np.rot90(a, -d['rotation']).copy()


def _fingerprint(puzzle):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed diversity, structural and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, 'seed diversity')
    periods, shapes, profiles, motifs, densities, palettes = set(), set(), set(), set(), set(), set()
    orientations, turns, marker_sides, components = set(), set(), set(), set()
    shifts = set()
    for seed, puzzle in enumerate(puzzles):
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'whole-puzzle determinism')
        local_profiles, local_motifs = set(), set()
        td, qd = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                (td if split == 'train' else qd).append(d)
                periods.add(d['tiles'][d['favored']].shape)
                shapes.add(d['fabric'].shape)
                pk = (d['fabric'].shape, tuple(map(int, d['profile'])))
                profiles.add(pk)
                local_profiles.add(pk)
                palettes.add((d['frame'], d['background'], d['cut_color'], d['favored'], d['other']))
                orientations.add(d['rotation'])
                turns.add(_turns(d['profile']))
                marker_sides.add(d['left'])
                for mask in d['tiles'].values():
                    motifs.add(_key(mask))
                    local_motifs.add(_key(mask))
                    densities.add((mask.shape, int(mask.sum())))
                    components.add(_components(mask))
        _require(len(local_profiles) >= 3 and len(local_motifs) >= 3, 'within-puzzle structural variation')
        if min(d['fabric'].shape[0] for d in qd) > max(d['fabric'].shape[0] for d in td):
            shifts.add('deeper')
        if min(_turns(d['profile']) for d in qd) > max(_turns(d['profile']) for d in td):
            shifts.add('more_turns')
        train_signs = {int(x) for d in td for x in np.sign(np.diff(d['profile'])) if x}
        test_signs = {int(x) for d in qd for x in np.sign(np.diff(d['profile'])) if x}
        if train_signs and test_signs and train_signs.isdisjoint(test_signs):
            shifts.add('reverse')
        pair = puzzle['train'][1]
        d = _decode(pair['input'])
        a = _render_fabric(d)
        opposite = _opposite_dot(pair, np.random.default_rng(seed+500))
        e = _decode(opposite['input'])
        b = _render_fabric(e)
        cut = np.indices(a.shape)[1] == d['profile'][:, None]
        _require(np.array_equal(a[cut], b[cut]), 'dot side changed crease priority')
        _require(not np.array_equal(a, b), 'marker swap ineffective')
        _require(np.count_nonzero(pair['input'] != opposite['input']) == 2, 'counterfactual changes more than dot')
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)),
                     'quarter-turn equivariance')
        relocated = _relocate_swatches(pair)
        rd = _decode(relocated)
        _require(np.array_equal(_render_fabric(d), _render_fabric(rd)), 'swatch-position invariance')
    _require(periods == set(product(range(2, 5), repeat=2)), 'period variation')
    _require(len(shapes) >= 70 and len(profiles) >= 100, 'geometry variation')
    _require(len(motifs) >= 100 and len(densities) >= 25, 'motif variation')
    _require(len(palettes) >= 35 and orientations == set(range(4)), 'palette/orientation variation')
    _require(marker_sides == {False, True} and max(turns) >= 4 and max(components) >= 3,
             'topology/side variation')
    _require(shifts == {'deeper', 'more_turns', 'reverse'}, 'train/test shift diversity')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1000+10*nt+nq, nt, nq)
            _require(len(puzzle['train']) == nt and len(puzzle['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(123, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for role in ('frame', 'fabric'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in puzzles[0]}
        pair = bad['test'][0]
        d = _decode(pair['input'])
        canonical_out = np.rot90(pair['output'], d['rotation'])
        cell = (0, 0) if role == 'frame' else (d['divider']+1, 1)
        canonical_out[cell] = (int(canonical_out[cell])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'distinct_puzzles': 42, 'periods': len(periods),
            'fabric_shapes': len(shapes), 'crease_structures': len(profiles),
            'motifs': len(motifs), 'density_shape_combinations': len(densities),
            'palettes': len(palettes), 'shifts': sorted(shifts),
            'semantic_checks': 'authored invariants and metamorphic checks only'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
