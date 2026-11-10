"""Regional ordinal paint: island area addresses a capped-bar palette.

Every command island independently chooses a pigment by descending bar length,
then expands all its macro cells using its cap-linked binary stamp.  There is
no puzzle-level color dictionary, generated-data cache, or seed-aware solver.
"""
from __future__ import annotations

import copy
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(board, background):
    """Four-neighbor, equal-identity components, with no diagonal joining."""
    h, w = board.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or board[r, c] == background:
                continue
            identity = int(board[r, c])
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and int(board[yy, xx]) == identity):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append((identity, tuple(sorted(cells))))
    return result


def _read_canonical(core, sep, background):
    """Read the three framed compartments, never a sampled parameter record."""
    h, w = core.shape
    _require(8 <= h <= 30 and 6 <= w <= 30, "framed dimensions")
    _require(np.all(core[[0, -1], :] == sep)
             and np.all(core[:, [0, -1]] == sep), "outer frame")
    dividers = np.flatnonzero(np.all(core == sep, axis=1)).tolist()
    _require(len(dividers) == 4 and dividers[0] == 0 and dividers[-1] == h-1,
             "three compartments required")
    a, b = dividers[1:3]
    n, mh = a-1, b-a-1
    _require(2 <= n <= 4 and 3 <= mh <= 7, "stamp and macro height")
    top = core[1:a]
    full_columns = np.flatnonzero(np.all(top == sep, axis=0)).tolist()
    # Library boundaries occur at n+1 intervals. A blank tail may separate
    # the final card boundary from the outer right frame.
    edges = [0]
    while edges[-1] + n + 1 in full_columns:
        if np.all(top[:, edges[-1]+1:edges[-1]+n+1] == background):
            break
        edges.append(edges[-1] + n + 1)
    k = len(edges)-1
    _require(2 <= k <= 4, "two to four stamps")
    _require(set(full_columns) == set(edges + [w-1]), "library dividers")
    _require(np.all(top[:, edges[-1]+1:w-1] == background), "library tail")
    templates, library_order = {}, []
    for left in edges[:-1]:
        card = top[:, left+1:left+n+1]
        labels = np.unique(card[card != background])
        _require(len(labels) == 1, "one nonempty binary stamp per card")
        cap = int(labels[0])
        _require(cap != sep and cap not in templates, "unique stamp identities")
        templates[cap] = card == cap
        library_order.append(cap)
    middle = core[a+1:b]
    map_edges = np.flatnonzero(np.all(middle == sep, axis=0)).tolist()
    _require(len(map_edges) in (2, 3) and map_edges[0] == 0
             and map_edges[-1] == w-1, "macro frame")
    right = map_edges[1]
    mw = right-1
    _require(3 <= mw <= 7 and mh*n <= 30 and mw*n <= 30, "macro dimensions")
    _require(np.all(middle[:, right+1:w-1] == background), "macro tail")
    board = middle[:, 1:right].copy()
    _require(set(map(int, np.unique(board))) <= {background, *templates},
             "macro commands must be stamp identities")
    _require(set(map(int, np.unique(board))) - {background} == set(templates),
             "every identity must occur in map")
    rack = core[b+1:h-1, 1:w-1]
    _require(not np.any(rack == sep), "frame color inside rack")
    bars = {cap: [] for cap in templates}
    spatial = []
    for x in range(rack.shape[1]):
        occupied = np.flatnonzero(rack[:, x] != background)
        if not len(occupied):
            continue
        y, end = int(occupied[0]), int(occupied[-1])
        values = rack[y:end+1, x]
        _require(end == rack.shape[0]-1 and len(values) >= 3
                 and len(occupied) == len(values), "contiguous baseline bar")
        cap, pigment = int(values[0]), int(values[1])
        _require(cap in templates and int(values[-1]) == cap, "matching cap identity")
        _require(pigment not in {background, sep, *templates}
                 and np.all(values[1:-1] == pigment), "uniform distinct payload")
        record = dict(cap=cap, pigment=pigment, length=len(values), column=x)
        bars[cap].append(record)
        spatial.append(record)
    islands = _components(board, background)
    for cap, family in bars.items():
        _require(2 <= len(family) <= 6, "competitor population")
        _require(len({bar['length'] for bar in family}) == len(family), "length tie")
        _require(all(3 <= bar['length'] <= 12 for bar in family), "bar length bounds")
        family.sort(key=lambda bar: -bar['length'])
    for cap, cells in islands:
        _require(1 <= len(cells) <= 4 and len(cells) <= len(bars[cap]),
                 "island ordinal must exist")
    _require(np.any(board == background), "visible empty macro space")
    return dict(n=n, board=board, background=background, separator=sep,
                templates=templates, library_order=library_order, bars=bars,
                spatial=spatial, islands=islands)


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for sep in map(int, np.unique(grid)):
        rr, cc = np.nonzero(grid == sep)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        crop = grid[r0:r1, c0:c1]
        if not (np.all(crop[[0, -1], :] == sep) and np.all(crop[:, [0, -1]] == sep)):
            continue
        outside = np.ones(grid.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        for turns in range(4):
            core = np.rot90(crop, turns)
            for bg in map(int, np.unique(core)):
                if bg == sep or not np.all(grid[outside] == bg):
                    continue
                for reflected in (False, True):
                    try:
                        parsed = _read_canonical(np.fliplr(core) if reflected else core, sep, bg)
                    except AssertionError:
                        continue
                    parsed['turns'] = turns
                    parsed['reflected'] = reflected
                    candidates.append(parsed)
                    # Without blank tails both readings can be equivalent:
                    # reversing macro and micro columns commutes with tiling.
                    break
    _require(len(candidates) == 1, f"expected one frame interpretation, got {len(candidates)}")
    return candidates[0]


def _render(parsed, alternative=None):
    """Tile reconstruction plus explicit teaching counterfactuals."""
    n, bg = parsed['n'], parsed['background']
    board = parsed['board']
    result = np.full((board.shape[0]*n, board.shape[1]*n), bg, dtype=np.int8)
    cap_order = list(dict.fromkeys(bar['cap'] for bar in parsed['spatial']))
    for cap, cells in parsed['islands']:
        family = parsed['bars'][cap]
        rank = len(cells)
        if alternative == 'total_color':
            rank = int(np.sum(board == cap))
        elif alternative == 'max_span':
            rank = max(max(y for y, _ in cells)-min(y for y, _ in cells)+1,
                       max(x for _, x in cells)-min(x for _, x in cells)+1)
        elif alternative == 'bounding_box':
            rank = ((max(y for y, _ in cells)-min(y for y, _ in cells)+1)
                    * (max(x for _, x in cells)-min(x for _, x in cells)+1))
        if alternative == 'ascending':
            family = list(reversed(family))
        elif alternative == 'left':
            family = sorted(family, key=lambda bar: bar['column'])
        elif alternative == 'right':
            family = sorted(family, key=lambda bar: -bar['column'])
        elif alternative == 'longest':
            rank = 1
        elif alternative == 'shortest':
            rank = len(family)
        if rank > len(family):
            continue  # The invalid global-count guess cannot answer this island.
        pigment = family[rank-1]['pigment']
        template = parsed['templates'][cap]
        if alternative == 'position_match':
            template = parsed['templates'][parsed['library_order'][cap_order.index(cap)]]
        elif alternative == 'solid_tiles':
            template = np.ones((n, n), dtype=bool)
        elif alternative == 'source_color':
            pigment = cap
        elif alternative == 'rotate_stamp':
            template = np.rot90(template)
        tile = np.where(template, pigment, bg)
        for y, x in cells:
            result[y*n:(y+1)*n, x*n:(x+1)*n] = tile
    if parsed.get('reflected', False):
        result = np.fliplr(result)
    return np.rot90(result, -parsed['turns']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the regional ranks, identity links, pigments and tiles from input."""
    return _render(_decode(input_grid))


def _motif(rng, n):
    style = int(rng.integers(8))
    rr, cc = np.indices((n, n))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (rr == 0) | (rr == n-1) | (cc == 0) | (cc == n-1)
    elif style == 2:
        mask = rr == cc
    elif style == 3:
        mask = (rr+cc) % 2 == int(rng.integers(2))
    elif style == 4:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif style == 5:
        mask = rr % 2 == int(rng.integers(2))
    else:
        mask = rng.random((n, n)) < rng.uniform(.2, .8)
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _shape(rng, size, bent=False):
    if bent:
        cells = {(0, 0), (0, 1), (1, 0)}
    else:
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = sorted({q for y, x in cells
                               for q in ((y-1,x), (y+1,x), (y,x-1), (y,x+1))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
    return [(y-y0, x-x0) for y, x in sorted(cells)]


def _place_islands(rng, regions, bg, bent_cap=None):
    # Rebuild the command map if a tightly packed collection will not fit.
    for _ in range(80):
        mh, mw = map(int, rng.integers(3, 8, size=2))
        if sum(size for cap, size in regions) >= mh*mw:
            continue
        board = np.full((mh, mw), bg, dtype=np.int8)
        memberships = []
        order = rng.permutation(len(regions))
        for index in order:
            cap, size = regions[int(index)]
            for attempt in range(150):
                shape = _shape(rng, size, bent=cap == bent_cap and size == 3)
                sh, sw = max(y for y, x in shape)+1, max(x for y, x in shape)+1
                if sh > mh or sw > mw:
                    continue
                y0, x0 = int(rng.integers(mh-sh+1)), int(rng.integers(mw-sw+1))
                cells = [(y+y0, x+x0) for y, x in shape]
                if any(board[y, x] != bg for y, x in cells):
                    continue
                if any(0 <= yy < mh and 0 <= xx < mw and board[yy, xx] == cap
                       for y, x in cells for yy, xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1))):
                    continue
                for y, x in cells:
                    board[y, x] = cap
                memberships.append((cap, cells))
                break
            else:
                break
        else:
            return board, memberships
    raise RuntimeError("bounded island placement exhausted")


def _make_pair(rng, params, n, k, probe=0, target_four=False, extra_regions=False):
    anchor = params['palette'][2]
    if params['mixed_palette']:
        rest = [int(c) for c in rng.permutation([c for c in range(10) if c != anchor])]
        bg, sep = rest[:2]
        caps = [anchor] + rest[2:k+1]
        pigments = rest[k+1:]
    else:
        bg, sep = params['palette'][:2]
        caps = params['palette'][2:2+k]
        pigments = params['palette'][2+k:]
    regions = []
    for cap in caps:
        sizes = [int(rng.integers(1, 4))]
        if cap == anchor and probe:
            sizes = [1, 2] if probe == 1 else [1, 3]
        elif cap == anchor and target_four:
            sizes = [int(rng.integers(1, 4)), 4]
        elif cap == anchor and extra_regions:
            sizes = [1, 2, 3]
        elif rng.random() < (.7 if extra_regions else .35):
            sizes.append(int(rng.integers(1, 4)))
        regions.extend((cap, size) for size in sizes)
    board, memberships = _place_islands(rng, regions, bg, anchor if probe == 2 else None)
    masks = {cap: _motif(rng, n) for cap in caps}
    if probe:
        # A visible hole and orientation difference make copying the complete
        # bitmap necessary, without reserving one fixed library silhouette.
        for _ in range(80):
            mask = rng.random((n, n)) < rng.uniform(.3, .7)
            if mask.any() and not mask.all() and not np.array_equal(mask, np.rot90(mask)):
                masks[anchor] = mask
                break
        else:
            masks[anchor] = np.tri(n, dtype=bool)
    bars = []
    ranked = {}
    for cap in caps:
        max_rank = max(size for identity, size in regions if identity == cap)
        minimum = max(2, max_rank, 3 if cap == anchor and probe == 1 else 2)
        count = int(rng.integers(minimum, 7))
        upper = int(rng.integers(count+2, 13))
        style = int(rng.integers(3))
        if style == 0:
            start = int(rng.integers(3, upper-count+2))
            lengths = list(range(start, start+count))
        elif style == 1:
            lengths = [int(x) for x in np.rint(np.linspace(3, upper, count))]
        else:
            lengths = [int(x) for x in rng.choice(np.arange(3, upper+1), count, replace=False)]
        lengths.sort(reverse=True)
        payloads = [int(rng.choice(pigments)) for _ in lengths]
        if cap == anchor and probe:
            # Distinct pigments at the demonstrated ordinals prevent a
            # coincidental shared payload from hiding the ranking mechanism.
            payloads[:max_rank] = [int(x) for x in rng.choice(pigments, max_rank, replace=False)]
        ranked[cap] = [dict(cap=cap, length=length, pigment=pigment)
                       for length, pigment in zip(lengths, payloads)]
        bars.extend(ranked[cap])
    if rng.random() < .3:
        order = [int(cap) for cap in rng.permutation(caps)]
        bars = [bar for cap in order for bar in rng.permutation(ranked[cap])]
    else:
        bars = [bars[int(i)] for i in rng.permutation(len(bars))]
    gaps = [0] * (len(bars)-1)
    budget = 28-len(bars)
    for _ in range(int(rng.integers(min(budget, 2*len(gaps))+1))):
        available = [i for i, gap in enumerate(gaps) if gap < 2]
        gaps[int(rng.choice(available))] += 1
    rack_width = len(bars)+sum(gaps)+2
    library_width = k*(n+1)+1
    width = max(rack_width, library_width, board.shape[1]+2)
    width += int(rng.integers(min(3, 30-width)+1))
    length = max(bar['length'] for bar in bars)
    d1, d2 = n+1, n+board.shape[0]+2
    height = d2+length+2
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[[0,d1,d2,height-1], :] = sep
    grid[:, [0,width-1]] = sep
    library_order = [int(c) for c in rng.permutation(caps)]
    for i, cap in enumerate(library_order):
        left = i*(n+1)
        grid[1:d1, left] = grid[1:d1, left+n+1] = sep
        grid[1:d1, left+1:left+n+1] = np.where(masks[cap], cap, bg)
    grid[d1+1:d2, 1:board.shape[1]+1] = board
    grid[d1+1:d2, board.shape[1]+1] = sep
    x = 1
    for i, bar in enumerate(bars):
        top = height-1-bar['length']
        grid[top, x] = grid[height-2, x] = bar['cap']
        grid[top+1:height-2, x] = bar['pigment']
        if i < len(gaps):
            x += 1+gaps[i]
    # Generation's answer uses construction membership and Kronecker masks;
    # the input-only solver independently rediscovers components and tiles.
    output = np.full((board.shape[0]*n, board.shape[1]*n), bg, dtype=np.int8)
    for cap, cells in memberships:
        layer = np.zeros(board.shape, dtype=bool)
        for y, x in cells:
            layer[y, x] = True
        pigment = ranked[cap][len(cells)-1]['pigment']
        output[np.kron(layer, masks[cap])] = pigment
    vertical = int(rng.integers(min(3, 30-height)+1))
    horizontal = int(rng.integers(min(3, 30-width)+1))
    top_pad, left_pad = int(rng.integers(vertical+1)), int(rng.integers(horizontal+1))
    grid = np.pad(grid, ((top_pad,vertical-top_pad),(left_pad,horizontal-left_pad)),
                  constant_values=bg)
    if rng.random() < .5:
        grid, output = np.fliplr(grid), np.fliplr(output)
    rotation = params['rotation']
    return dict(input=np.rot90(grid, rotation).copy(), output=np.rot90(output, rotation).copy())


ALTERNATIVES = ('ascending', 'left', 'right', 'longest', 'shortest', 'total_color',
                'max_span', 'bounding_box', 'position_match', 'solid_tiles',
                'source_color', 'rotate_stamp')


def _training_failures(pairs):
    parsed = [_decode(pair['input']) for pair in pairs]
    return [alt for alt in ALTERNATIVES
            if all(np.array_equal(pair['output'], _render(p, alt))
                   for pair, p in zip(pairs, parsed))]


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train',num_train,2,10), ('num_test',num_test,1,4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not lo <= int(value) <= hi:
            raise ValueError(f'{name} outside {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    params = dict(palette=[int(c) for c in rng.permutation(10)],
                  mixed_palette=bool(rng.random() < .7), rotation=int(rng.integers(4)))
    shift = str(rng.choice(['topology', 'scale', 'families', 'recombine']))
    scales = [2, 3, 4]
    held_scale = int(rng.choice([3,4]))
    if shift == 'scale':
        scales.remove(held_scale)
    ns = [int(x) for x in rng.permutation(scales)[:2]]
    ns += [int(rng.choice(scales)) for _ in range(num_train-2)]
    max_k = 3 if shift == 'families' else 4
    ks = [int(rng.integers(2, max_k+1)) for _ in range(num_train)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    # Build two complementary teaching examples; retry only their geometries
    # and palettes if an explicit incorrect rule happens to fit both outputs.
    for _ in range(80):
        first = _make_pair(rng, params, ns[0], ks[0], probe=1)
        second = _make_pair(rng, params, ns[1], ks[1], probe=2)
        if (_signature(first['input']) != _signature(second['input'])
                and not _training_failures([first, second])):
            puzzle['train'] = [first, second]
            seen = {_signature(first['input']), _signature(second['input'])}
            break
    else:
        raise RuntimeError('bounded contrast construction exhausted')
    for split, count in (('train',num_train-2), ('test',num_test)):
        for index in range(count):
            if split == 'train':
                n, k = ns[index+2], ks[index+2]
            else:
                n = held_scale if shift == 'scale' else int(rng.choice(ns))
                k = 4 if shift == 'families' else int(rng.choice(ks))
            for _ in range(40):
                pair = _make_pair(rng, params, n, k,
                                  target_four=split == 'test' and shift == 'topology',
                                  extra_regions=split == 'test' and shift == 'recombine')
                signature = _signature(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train','test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, parsed_train = set(), []
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input','output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            parsed = _decode(inp)
            _require(np.array_equal(out, _render(parsed)), 'incorrect regional expansion')
            _require(set(map(int, np.unique(out))) <= {parsed['background'],
                     *(bar['pigment'] for family in parsed['bars'].values() for bar in family)},
                     'output palette')
            if split == 'train':
                parsed_train.append(parsed)
    ranks = {len(cells) for p in parsed_train for cap, cells in p['islands']}
    _require({1,2,3} <= ranks, 'teach ordinal progression before possible rank four')
    _require(any(len(cells) == 2 and len(p['bars'][cap]) >= 3
                 for p in parsed_train for cap, cells in p['islands']), 'intermediate selection')
    _require(any(len({len(cells) for identity, cells in p['islands'] if identity == cap}) > 1
                 for p in parsed_train for cap in p['templates']), 'regional query contrast')
    _require(not _training_failures(puzzle['train']), 'surviving training shortcut')
    return True


def self_test():
    """Bounded, input-derived structural coverage and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, motifs, maps, populations, ranks, palettes = set(), set(), set(), set(), set(), set()
    dimensions, family_counts, lengths, disconnected, shared = set(), set(), set(), 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train','test'):
            for pair in puzzle[split]:
                p = _decode(pair['input'])
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), 'list input')
                shapes.add((pair['input'].shape, pair['output'].shape))
                dimensions.add(p['n'])
                family_counts.add(len(p['templates']))
                palettes.add((p['background'],p['separator']))
                order = p['library_order']
                normalized = np.full(p['board'].shape, -1, dtype=np.int8)
                for i, cap in enumerate(order):
                    normalized[p['board'] == cap] = i
                    motifs.add((p['n'],p['templates'][cap].tobytes()))
                    populations.add(len(p['bars'][cap]))
                maps.add(_signature(normalized))
                ranks.update(len(cells) for cap, cells in p['islands'])
                lengths.update(bar['length'] for bar in p['spatial'])
                disconnected += any(sum(identity == cap for identity, cells in p['islands']) > 1
                                    for cap in p['templates'])
                payload_sets = [{b['pigment'] for b in p['bars'][cap]} for cap in order]
                shared += any(payload_sets[i] & payload_sets[j]
                              for i in range(len(order)) for j in range(i))
    _require(dimensions == {2,3,4} and family_counts == {2,3,4}, 'scale/family coverage')
    _require(ranks == {1,2,3,4} and populations == {2,3,4,5,6}, 'area/population coverage')
    _require(len(shapes) >= 60 and len(maps) >= 150 and len(motifs) >= 100,
             'structural variation across 42 seeds')
    _require(len(palettes) >= 30 and disconnected >= 100 and shared >= 50,
             'palette and regional/shared-payload variation')
    _require({3,4,5,11,12} <= lengths, 'close and long lengths')
    first = puzzles[0]
    again = generate_puzzle(0)
    _require(all(np.array_equal(x[role], y[role]) for s in first
                 for x, y in zip(first[s], again[s]) for role in x), 'determinism')
    # Whole-picture rotations, reflections, and arbitrary color renaming
    # preserve the output coordinate convention without a hidden orientation.
    pair = first['test'][0]
    mapping = np.array([8,3,0,9,4,2,7,1,6,5], dtype=np.int8)
    for turns in range(4):
        for reflected in (False, True):
            x = np.fliplr(pair['input']) if reflected else pair['input']
            y = np.fliplr(pair['output']) if reflected else pair['output']
            _require(np.array_equal(solve(mapping[np.rot90(x, turns)]),
                                    mapping[np.rot90(y, turns)]), 'geometry/color equivariance')
    # Reorder the actual library cards and occupied rack columns separately.
    # Identity correspondence and family length ranks must survive both.
    p = _decode(pair['input'])
    rr, cc = np.nonzero(pair['input'] == p['separator'])
    core = np.rot90(pair['input'][rr.min():rr.max()+1, cc.min():cc.max()+1], p['turns'])
    expected = np.rot90(pair['output'], p['turns'])
    if p['reflected']:
        core, expected = np.fliplr(core), np.fliplr(expected)
    dividers = np.flatnonzero(np.all(core == p['separator'], axis=1))
    d1, d2 = map(int, dividers[1:3])
    columns = [bar['column']+1 for bar in p['spatial']]
    moved = core.copy()
    moved[d2+1:-1, columns] = core[d2+1:-1, columns[::-1]]
    _require(np.array_equal(solve(moved), expected), 'rack-order independence')
    moved = core.copy()
    n, k = p['n'], len(p['templates'])
    for i in range(k):
        j = k-1-i
        moved[1:d1, i*(n+1)+1:i*(n+1)+n+1] = core[1:d1, j*(n+1)+1:j*(n+1)+n+1]
    _require(np.array_equal(solve(moved), expected), 'library-order independence')
    # A valid-looking second bar with the first bar's length creates an
    # undefined ordinal and must be rejected rather than spatially tie-broken.
    cap, family = next(iter(p['bars'].items()))
    bad_input = core.copy()
    x, length = family[1]['column']+1, family[0]['length']
    bad_input[d2+1:-1, x] = p['background']
    top = core.shape[0]-1-length
    bad_input[top, x] = bad_input[-2, x] = cap
    bad_input[top+1:-2, x] = family[1]['pigment']
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied lengths accepted')
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            validate_generated_puzzle(generate_puzzle(42, a, b))
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    bad = copy.deepcopy(first)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, grid_shapes=len(shapes), normalized_maps=len(maps),
                binary_motifs=len(motifs), background_frame_assignments=len(palettes),
                island_areas=sorted(ranks), populations=sorted(populations),
                disconnected_identity_pairs=disconnected, shared_payload_pairs=shared)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
