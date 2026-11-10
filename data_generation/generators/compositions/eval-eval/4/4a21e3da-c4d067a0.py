"""Matching torn edges: register a cut jointly, then magnify its two pieces.

Only standard-library modules and NumPy are used. No files are read or written.
The two enlarged strips are movable rulers, not already anchored output pixels.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _matches(plan, seq):
    """Vertical occurrences, including repetitions in a single column."""
    h, w = plan.shape
    return [(r, c) for c in range(w) for r in range(h-len(seq)+1)
            if tuple(int(v) for v in plan[r:r+len(seq), c]) == tuple(seq)]


def _cuts(plan, sequences):
    a, b = (_matches(plan, seq) for seq in sequences)
    cuts = sorted({c+1 for r, c in a if any(d == c+1 for s, d in b)})
    return cuts, (a, b)


def _trim(plan, bg):
    p = np.argwhere(plan != bg)
    if not len(p):
        raise ValueError("empty torn piece")
    lo, hi = p.min(axis=0), p.max(axis=0)+1
    return plan[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _recover(input_grid):
    """Normalize only the visible marker direction, then parse input pixels."""
    raw = _grid(input_grid)
    counts = np.bincount(raw.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("ambiguous background")
    h, w = raw.shape
    rim = np.zeros(raw.shape, dtype=bool)
    rim[0] = rim[-1] = True
    rim[:, 0] = rim[:, -1] = True
    markers = np.argwhere(rim & (raw != bg))
    if len(markers) != 1:
        raise ValueError("input must have exactly one nonbackground rim cell")
    r, c = map(int, markers[0])
    if sum((r == 0, r == h-1, c == 0, c == w-1)) != 1:
        raise ValueError("marker must be on a side, not a corner")
    mark = int(raw[r, c])
    if counts[mark] != 1:
        raise ValueError("marker color is reserved")
    # Rotate the ENTIRE scene until the marker is at the top. Relative order
    # between the blueprint and the rulers is preserved by this normalization.
    turn = 0 if r == 0 else 1 if c == w-1 else 2 if r == h-1 else 3
    a = np.rot90(raw, turn).copy()
    axis = int(np.argwhere(a == mark)[0, 1])
    mask = (a != bg) & (a != mark)
    comps = _components(mask)
    swatches = [p[0] for p in comps if len(p) == 1]
    if len(swatches) < 6:
        raise ValueError("missing miniature blueprint")
    points = np.array(swatches, dtype=int)
    mr, mc = map(int, points.min(axis=0))
    er, ec = map(int, points.max(axis=0)+1)
    if np.any((points-(mr, mc)) % 2):
        raise ValueError("miniature swatches must have pitch two")
    if any(not np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1)):
        raise ValueError("miniature pitch is not witnessed")
    plan = a[mr:er:2, mc:ec:2].copy()
    mini_mask = np.zeros((er-mr, ec-mc), dtype=bool)
    mini_mask[::2, ::2] = True
    if np.any(a[mr:er, mc:ec][~mini_mask] != bg):
        raise ValueError("non-swatch pixel inside miniature")
    if np.count_nonzero(plan != bg) != len(swatches):
        raise ValueError("unaccounted miniature cells")
    rulers = [[], []]
    for cells in comps:
        if len(cells) == 1:
            continue
        p = np.array(cells, dtype=int)
        y, x = map(int, p.min(axis=0))
        bottom, right = map(int, p.max(axis=0)+1)
        side = bottom-y
        if (side < 2 or right-x != side or len(cells) != side*side
                or not np.all(a[y:bottom, x:right] == a[y, x])):
            raise ValueError("ruler components must be solid squares")
        if right <= axis:
            group = 0
        elif x > axis:
            group = 1
        else:
            raise ValueError("ruler crosses marker axis")
        rulers[group].append((y, x, side, int(a[y, x])))
    sequences, geometry = [], []
    for group in rulers:
        group.sort()
        if len(group) < 2 or len({p[1] for p in group}) != 1 or len({p[2] for p in group}) != 1:
            raise ValueError("each side needs a vertical ruler of equal squares")
        side = group[0][2]
        differences = np.diff([p[0] for p in group])
        stride = int(differences[0])
        if stride <= side or not np.all(differences == stride):
            raise ValueError("ruler needs consecutive equally spaced sites")
        sequences.append(tuple(p[3] for p in group))
        geometry.append((side, stride))
    candidates, matches = _cuts(plan, sequences)
    if len(candidates) != 1:
        raise ValueError("the ordered neighboring bank signatures must select one cut")
    cut = candidates[0]
    pieces = [_trim(plan[:, :cut], bg), _trim(plan[:, cut:], bg)]
    boxes = []
    for i, (piece, (side, stride)) in enumerate(zip(pieces, geometry)):
        ph, pw = (side+(n-1)*stride for n in piece.shape)
        y, x = 0, 0 if i == 0 else a.shape[1]-pw
        if ph > a.shape[0] or pw > a.shape[1] or x < 0:
            raise ValueError("corner mosaic does not fit")
        if (i == 0 and pw >= axis) or (i == 1 and x <= axis):
            raise ValueError("corner mosaic crosses marker axis")
        if not (ph <= mr or x+pw <= mc or ec <= x):
            raise ValueError("corner mosaic would overwrite miniature")
        boxes.append((y, x, ph, pw))
    return dict(raw=raw, grid=a, turn=turn, background=bg, marker=mark,
                axis=axis, plan=plan, mini=(mr, mc, er, ec), rulers=rulers,
                sequences=sequences, geometry=geometry, cut=cut, matches=matches,
                pieces=pieces, boxes=boxes)


def _render(info, *, unit=False, uniform=False, keep_rulers=False):
    a, bg = info['grid'], info['background']
    out = a.copy()
    if not keep_rulers:
        for group in info['rulers']:
            for y, x, side, color in group:
                out[y:y+side, x:x+side] = bg
    for i, piece in enumerate(info['pieces']):
        side, stride = (1, 2) if unit else info['geometry'][0 if uniform else i]
        ph, pw = (side+(n-1)*stride for n in piece.shape)
        y, x = 0, 0 if i == 0 else a.shape[1]-pw
        if ph > a.shape[0] or x < 0:
            return None
        for (r, c), color in np.ndenumerate(piece):
            if color != bg:
                out[y+r*stride:y+r*stride+side, x+c*stride:x+c*stride+side] = color
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone: no seed, examples, or generation metadata."""
    return _render(_recover(input_grid))


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= value <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def _options(height, left_width, right_width, unequal=False):
    result = []
    for sl in range(2, 5):
        for gl in range(1, 4):
            for sr in range(2, 5):
                for gr in range(1, 4):
                    if unequal and sl == sr:
                        continue
                    ph = max(sl+(height-1)*(sl+gl), sr+(height-1)*(sr+gr))
                    wl = sl+(left_width-1)*(sl+gl)
                    wr = sr+(right_width-1)*(sr+gr)
                    # Miniature is wholly below both final mosaics. Keep one
                    # blank column on either side of the marker's axis.
                    hmin = ph+(2*height-1)+3
                    wmin = wl+wr+3
                    wmin = max(wmin, 2*(left_width+right_width)-1+4)
                    if hmin <= 30 and wmin <= 30:
                        result.append((sl, gl, sr, gr, hmin, wmin, ph, wl, wr))
    return result


def _sample_plan(rng, h, lw, rw, bg, palette, lengths, starts, witness, holes):
    """Bounded proposals; the fallback has two rare, different bank colors."""
    cut, w = lw, lw+rw
    for attempt in range(65):
        plan = rng.choice(palette, size=(h, w)).astype(np.int8)
        if attempt < 64:
            density = float(rng.uniform(.52, .95)) if holes else 1.0
            plan[rng.random((h, w)) > density] = bg
            seqs = [tuple(int(v) for v in rng.choice(palette, size=n)) for n in lengths]
        else:
            # One unique color per bank prevents any competing neighboring cut.
            plan.fill(palette[2])
            seqs = [tuple([int(palette[i])]*n) for i, n in enumerate(lengths)]
        for i, col in enumerate((cut-1, cut)):
            r = starts[i]
            plan[r:r+lengths[i], col] = seqs[i]
        if witness:
            # Each signature occurs elsewhere too; only their adjacency resolves
            # the cut. The extra left signature is at the far right, and vice versa.
            plan[starts[0]:starts[0]+lengths[0], -1] = seqs[0]
            plan[starts[1]:starts[1]+lengths[1], 0] = seqs[1]
        if attempt == 64 and holes:
            protected = {(r, c) for i, c in enumerate((cut-1, cut))
                         for r in range(starts[i], starts[i]+lengths[i])}
            if witness:
                protected |= {(r, w-1 if i == 0 else 0) for i in range(2)
                              for r in range(starts[i], starts[i]+lengths[i])}
            choices = [(r, c) for r in range(h) for c in range(w)
                       if (r, c) not in protected]
            if choices:
                plan[choices[0]] = bg
        candidates, matches = _cuts(plan, seqs)
        if candidates != [cut] or (witness and min(map(len, matches)) < 2):
            continue
        pieces = [plan[:, :cut], plan[:, cut:]]
        # Preserve the full sampled extent, while allowing empty edge sites.
        if any(any(np.all(edge == bg) for edge in (p[0], p[-1], p[:, 0], p[:, -1])) for p in pieces):
            continue
        if holes and not np.any(plan == bg):
            continue
        return plan, seqs
    raise AssertionError('constructive plan fallback failed')


def _make_pair(rng, kind, turn, palette_roles):
    bg, marker, palette = palette_roles
    witness = kind == 'joint'
    h = (4 if witness else int(rng.integers(4, 6)) if kind == 'partial'
         else int(rng.integers(3, 6)))
    lw = 2 if witness else 3 if kind == 'partial' else int(rng.integers(2, 4))
    rw = 2 if witness else int(rng.integers(2, 4))
    if kind == 'test':
        if rng.integers(2):
            lw = 4
        else:
            rw = 4
    options = _options(h, lw, rw, unequal=witness)
    sl, gl, sr, gr, hmin, wmin, ph, wl, wr = options[int(rng.integers(len(options)))]
    if kind == 'joint':
        lengths = [2, 2]
        starts = [int(rng.choice([0, h-2])), int(rng.choice([0, h-2]))]
    elif kind == 'partial':
        lengths = [2, 2]
        starts = [int(rng.integers(h-1)), int(rng.integers(h-1))]
        starts[0] = 1 if h >= 4 else starts[0]
    else:
        lengths = [int(rng.integers(2, min(3, h)+1)) for _ in range(2)]
        starts = [int(rng.integers(h-n+1)) for n in lengths]
    # On tests both strips are proper interior row segments of the bank when
    # height allows; at height three, one familiar prefix/suffix is retained.
    if kind == 'test' and h >= 4:
        lengths = [2, 2]
        starts = [1, int(rng.integers(1, h-2))]
    holes = witness or kind in ('partial', 'test') or rng.random() < .65
    plan, seqs = _sample_plan(rng, h, lw, rw, bg, palette, lengths, starts, witness, holes)
    height = int(rng.integers(hmin, 31))
    width = int(rng.integers(wmin, 31))
    # Capacity of each side depends on its mosaic, not on a fixed middle line.
    axis = int(rng.integers(wl+1, width-wr-1))
    mini_h, mini_w = 2*h-1, 2*plan.shape[1]-1
    mr = int(rng.integers(ph+2, height-mini_h))
    mini_columns = list(range(1, width-mini_w))
    if witness:
        # Teach that the physical axis separates the rulers, not the blueprint.
        mini_columns = [c for c in mini_columns
                        if sum(c+2*k < axis for k in range(plan.shape[1])) != lw]
    mc = int(rng.choice(mini_columns))
    a = np.full((height, width), bg, dtype=np.int8)
    a[0, axis] = marker
    a[mr:mr+mini_h:2, mc:mc+mini_w:2] = plan
    for i, (side, gutter) in enumerate(((sl, gl), (sr, gr))):
        stride = side+gutter
        run_h = side+(lengths[i]-1)*stride
        y = int(rng.integers(1, mr-run_h))
        x = (int(rng.integers(1, axis-side)) if i == 0
             else int(rng.integers(axis+1, width-side)))
        for k, color in enumerate(seqs[i]):
            a[y+k*stride:y+k*stride+side, x:x+side] = color
    # Draw labels directly from the sampled plan, independently of input parsing.
    out = a.copy()
    for i, (side, gutter) in enumerate(((sl, gl), (sr, gr))):
        stride = side+gutter
        # Find only the visible ruler pixels, without clearing any miniature.
        for cells in _components((a != bg) & (a != marker)):
            if len(cells) > 1:
                rr, cc = zip(*cells)
                belongs = max(cc) < axis if i == 0 else min(cc) > axis
                if belongs:
                    for p in cells:
                        out[p] = bg
        piece = plan[:, :lw] if i == 0 else plan[:, lw:]
        pw = side+(piece.shape[1]-1)*stride
        x0 = 0 if i == 0 else width-pw
        for (r, c), color in np.ndenumerate(piece):
            if color != bg:
                out[r*stride:r*stride+side, x0+c*stride:x0+c*stride+side] = color
    return {k: np.rot90(v, -turn).copy() for k, v in (('input', a), ('output', out))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    pool_size = int(rng.integers(1, min(4, nt)+1))
    turns = [int(v) for v in rng.choice(4, pool_size, replace=False)]
    schedule = turns+[int(rng.choice(turns)) for _ in range(nt-pool_size)]
    rng.shuffle(schedule)
    fixed_palette = rng.random() < .3
    colors = rng.permutation(10)
    roles = (int(colors[0]), int(colors[1]), colors[2:7])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            kind = ('joint' if index == 0 else 'partial' if index == 1 else 'ordinary') if split == 'train' else 'test'
            turn = schedule[index] if split == 'train' else int(rng.choice(turns))
            for attempt in range(24):
                if not fixed_palette:
                    colors = rng.permutation(10)
                    roles = (int(colors[0]), int(colors[1]), colors[2:int(rng.integers(5, 8))])
                pair = _make_pair(rng, kind, turn, roles)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input retry budget exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    info = _recover(input_grid)
    interior = any(0 < r and r+len(info['sequences'][i]) < info['plan'].shape[0]
                   for i, matches in enumerate(info['matches'])
                   for r, c in matches if c == info['cut']-1+i)
    physical = sum(info['mini'][1]+2*c < info['axis'] for c in range(info['plan'].shape[1]))
    physical = max(1, min(info['plan'].shape[1]-1, physical))
    return dict(shape=tuple(info['raw'].shape), turn=info['turn'],
                background=info['background'], marker=info['marker'],
                plan_shape=tuple(info['plan'].shape), cut=info['cut'],
                piece_widths=tuple(p.shape[1] for p in info['pieces']),
                geometry=tuple(info['geometry']),
                lengths=tuple(map(len, info['sequences'])),
                match_counts=tuple(map(len, info['matches'])),
                holes=int(np.count_nonzero(info['plan'] == info['background'])),
                repeated_seed=any(len(set(s)) == 1 for s in info['sequences']),
                interior_signature=interior, physical_axis_cut_wrong=physical != info['cut'],
                moved=not np.array_equal(_render(info), _render(info, keep_rulers=True)),
                source_position=tuple(info['mini'][:2]))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError('puzzle must have exactly train/test')
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError('splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _count(nt, 'num_train', 2, 10)
    _count(ne, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError('pair must have exactly input/output')
            inp, out = (_grid(pair[k]) for k in ('input', 'output'))
            if inp.shape != out.shape or _key(inp) in seen:
                raise ValueError('shape mismatch or duplicate input')
            seen.add(_key(inp))
            info = _recover(inp)
            expected = _render(info)
            if not np.array_equal(out, expected) or np.array_equal(inp, out):
                raise ValueError('wrong or degenerate output')
            canonical_out = np.rot90(out, info['turn'])
            r, c, bottom, right = info['mini']
            if not np.array_equal(canonical_out[r:bottom, c:right], info['grid'][r:bottom, c:right]):
                raise ValueError('miniature must be preserved')
            # Counts include one preserved miniature and both literal copies;
            # no invented palette, erased blank, or independently packed swatch.
            for color in range(10):
                if color in (info['background'], info['marker']):
                    continue
                count = int(np.count_nonzero(info['plan'] == color))
                count += sum(int(np.count_nonzero(p == color))*s*s
                             for p, (s, stride) in zip(info['pieces'], info['geometry']))
                if np.count_nonzero(out == color) != count:
                    raise ValueError('incorrect literal-color area conservation')
            if np.array_equal(out, _render(info, unit=True)):
                raise ValueError('magnification is vacuous')
            features[split].append(pair_features(inp))
    tr, te = features['train'], features['test']
    if not any(min(f['match_counts']) >= 2 for f in tr):
        raise ValueError('missing jointly disambiguated seam demonstration')
    if not any(f['geometry'][0][0] != f['geometry'][1][0] for f in tr):
        raise ValueError('missing independent-scale demonstration')
    if not any(f['holes'] for f in tr) or not any(f['moved'] for f in tr):
        raise ValueError('missing blank-site or ruler-relocation demonstration')
    if not any(f['interior_signature'] for f in tr) or not any(f['physical_axis_cut_wrong'] for f in tr):
        raise ValueError('missing interior-match or displaced-axis demonstration')
    if not {f['turn'] for f in te} <= {f['turn'] for f in tr}:
        raise ValueError('test marker direction has no demonstration')


def self_test():
    """42 seeds, all legal counts, mutation checks; no retained sample payloads."""
    structures, dimensions, plans, geometries, backgrounds = set(), set(), set(), set(), set()
    turns = {'train': set(), 'test': set()}
    repeated = moved = joint = holey = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = pair_features(pair['input'])
                dimensions.add(f['shape'])
                plans.add(f['plan_shape'])
                geometries.update(f['geometry'])
                backgrounds.add(f['background'])
                turns[split].add(f['turn'])
                repeated += f['repeated_seed']
                moved += f['moved']
                joint += min(f['match_counts']) >= 2
                holey += f['holes'] > 0
                # Normalize colors to measure structure rather than recoloring.
                roles, norm = {}, []
                for color in pair['input'].ravel():
                    norm.append(roles.setdefault(int(color), len(roles)))
                structures.add(hashlib.sha256(bytes(norm)+bytes(f['shape'])).digest())
                info = _recover(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert not np.array_equal(_render(info, unit=True), pair['output'])
                if info['geometry'][0] != info['geometry'][1]:
                    alternate = _render(info, uniform=True)
                    assert alternate is None or not np.array_equal(alternate, pair['output'])
        # The larger test is a structural extrapolation, not a new convention.
        assert all(max(pair_features(p['input'])['piece_widths']) <= 3 for p in puzzle['train'])
        assert all(max(pair_features(p['input'])['piece_widths']) == 4 for p in puzzle['test'])
    assert len(structures) >= 190 and len(dimensions) >= 25 and len(plans) >= 10
    assert len(geometries) >= 6 and backgrounds == set(range(10))
    assert turns['train'] == turns['test'] == set(range(4))
    assert min(repeated, moved, joint, holey) >= 20
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(10000+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    a, b = generate_puzzle(777, 5, 2), generate_puzzle(777, 5, 2)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
