"""Damaged transparencies: joint translation, rectangle coverage and occlusion.

Only NumPy and the Python standard library are used. The solver reads only its
input; support generation is deliberately separate from layout enumeration.
"""

import itertools

import numpy as np


_PAINT_ORDER = (1, 0, 2)  # low to high: middle, top, bottom source band


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "colors must be 0..9")
    return grid


def _count(value, lo, hi, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), name + " must be an integer")
    _require(lo <= int(value) <= hi, name + " out of range")
    return int(value)


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [cells.pop()]
    while todo:
        r, c = todo.pop()
        for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if cell in cells:
                cells.remove(cell)
                todo.append(cell)
    return not cells


def _crop(mask, scar):
    rr, cc = np.where(mask)
    slices = (slice(int(rr.min()), int(rr.max()) + 1),
              slice(int(cc.min()), int(cc.max()) + 1))
    return mask[slices].copy(), scar[slices].copy()


def _placements(piece, height, width):
    mask, scar = piece
    ph, pw = mask.shape
    # Place bitmaps directly in output coordinates, without intermediate grids.
    support = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    defects = [(int(r), int(c)) for r, c in np.argwhere(scar)]
    result = []
    for row in range(height - ph + 1):
        for col in range(width - pw + 1):
            occupied = sum(1 << ((r + row) * width + c + col) for r, c in support)
            damaged = sum(1 << ((r + row) * width + c + col) for r, c in defects)
            result.append((occupied, damaged, (row, col)))
    return result


def _layouts(pieces, height, width, stop_after_two=True):
    """Enumerate joint fits and count the two separate constraint ablations.

    There is no assumption about which role owns the spanning piece. Its only
    possible translation fixes the common origin. Early exit is used only when
    rejecting an ambiguous sample, never to declare a layout unique.
    """
    options = [_placements(piece, height, width) for piece in pieces]
    full = (1 << (height * width)) - 1
    solutions, geometric, concealed = [], 0, 0
    for top, middle, bottom in itertools.product(*options):
        # Top damage must lie under bottom. Middle damage must lie under either
        # higher support; any damaged top cover is already hidden by bottom.
        hidden = not (top[1] & ~bottom[0] or middle[1] & ~(top[0] | bottom[0]))
        concealed += hidden
        if (top[0] | middle[0] | bottom[0]) != full:
            continue
        geometric += 1
        if not hidden:
            continue
        solutions.append((top[2], middle[2], bottom[2]))
        if stop_after_two and len(solutions) == 2:
            return solutions, geometric, concealed
    return solutions, geometric, concealed


def _render(pieces, positions, height, width, background, damage, colors,
            order=_PAINT_ORDER):
    out = np.full((height, width), background, dtype=np.int8)
    for role in order:
        mask, scar = pieces[role]
        row, col = positions[role]
        ph, pw = mask.shape
        view = out[row:row + ph, col:col + pw]
        view[mask] = colors[role]
        view[scar] = damage
    return out


def _parse(input_grid):
    grid = _grid(input_grid)
    _require(grid.shape[0] % 3 == 0, "three equal source bands required")
    band_h = grid.shape[0] // 3
    bands = [grid[k * band_h:(k + 1) * band_h] for k in range(3)]
    values = [set(map(int, np.unique(band))) for band in bands]
    common = values[0] & values[1] & values[2]
    _require(len(common) == 1, "one common background required")
    background = next(iter(common))
    marked = (values[0] & values[1]) - {background}
    _require(len(marked) == 1, "one shared damage color required")
    damage = next(iter(marked))
    _require(damage not in values[2], "highest layer cannot be damaged")
    colors, pieces = [], []
    for role, band in enumerate(bands):
        expected = {background, damage} if role < 2 else {background}
        payload = values[role] - expected
        _require(len(payload) == 1, "one payload color per band required")
        colors.append(next(iter(payload)))
        _require(bool(np.all(band[0] == background)
                      and np.all(band[-1] == background)
                      and np.all(band[:, 0] == background)
                      and np.all(band[:, -1] == background)), "band border must be background")
        mask, scar = band != background, band == damage
        _require(_connected(mask), "each support must be four-connected")
        pieces.append(_crop(mask, scar))
    _require(len(set([background, damage] + colors)) == 5, "five distinct roles required")
    height = max(piece[0].shape[0] for piece in pieces)
    width = max(piece[0].shape[1] for piece in pieces)
    dominant = [k for k, piece in enumerate(pieces)
                if piece[0].shape == (height, width)]
    _require(len(dominant) == 1, "one piece must span the minimum box")
    anchor = dominant[0]
    _require(all(k == anchor or (piece[0].shape[0] < height
                                 and piece[0].shape[1] < width)
                 for k, piece in enumerate(pieces)), "other pieces need smaller boxes")
    return pieces, height, width, background, damage, tuple(colors), anchor


def solve(input_grid: np.ndarray) -> np.ndarray:
    pieces, height, width, background, damage, colors, _ = _parse(input_grid)
    solutions, _, _ = _layouts(pieces, height, width)
    _require(len(solutions) == 1, "expected one damage-free minimum-area assembly")
    return _render(pieces, solutions[0], height, width, background, damage, colors)


def _small_shape(rng, height, width):
    style = int(rng.integers(4))
    mask = np.ones((height, width), dtype=bool)
    if style == 2:
        cuts = rng.integers(1, width + 1, size=height)
        cuts[int(rng.integers(height))] = width
        mask = np.arange(width)[None, :] < cuts[:, None]
    elif style == 3:
        mask[:] = False
        mask[0, :] = True
        mask[:, 0] = True
        target = int(rng.integers(height + width - 1, height * width + 1))
        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(~mask):
                if any(0 <= rr < height and 0 <= cc < width and mask[rr, cc]
                       for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
                    frontier.append((int(r), int(c)))
            r, c = frontier[int(rng.integers(len(frontier)))]
            mask[r, c] = True
    if style in (0, 1):
        # Solid, edge-notched and enclosed-hole supports all remain possible.
        for _ in range(int(rng.integers(0, max(2, height * width // 3)))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            old = bool(mask[r, c])
            mask[r, c] = False
            if not _connected(mask) or not mask.any(0).all() or not mask.any(1).all():
                mask[r, c] = old
    if rng.random() < 0.5:
        mask = mask[::-1]
    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    return mask.copy()


def _sample_supports(rng, anchor, height, width):
    masks = np.zeros((3, height, width), dtype=bool)
    for role in range(3):
        if role == anchor:
            continue
        ph, pw = int(rng.integers(2, height)), int(rng.integers(2, width))
        small = _small_shape(rng, ph, pw)
        row, col = int(rng.integers(height - ph + 1)), int(rng.integers(width - pw + 1))
        masks[role, row:row + ph, col:col + pw] = small
    others = np.any(np.delete(masks, anchor, axis=0), axis=0)
    masks[anchor] = True
    removal_rate = float(rng.uniform(0.22, 0.65))
    for r, c in rng.permutation(np.argwhere(others)):
        if rng.random() < removal_rate:
            masks[anchor, r, c] = False
            if (not _connected(masks[anchor]) or not masks[anchor].any(0).all()
                    or not masks[anchor].any(1).all()):
                masks[anchor, r, c] = True
    codes = (4 * masks[0].astype(np.int8) + 2 * masks[1].astype(np.int8)
             + masks[2].astype(np.int8))
    if not all(np.count_nonzero(codes == code) >= 2 for code in (3, 5, 6)):
        return None
    if not np.any(codes == 2):  # retain some visible lowest-layer ink
        return None
    protected = np.zeros_like(masks)
    for code, roles in ((3, (1,)), (5, (0,)), (6, (0, 1))):
        sites = np.argwhere(codes == code)
        row, col = sites[int(rng.integers(len(sites)))]
        for role in roles:
            protected[role, row, col] = True
    scars = np.zeros_like(masks)
    for role, higher in ((0, masks[2]), (1, masks[0] | masks[2])):
        sites = np.argwhere(masks[role] & higher & ~protected[role])
        if not len(sites):
            return None
        rate = float(rng.uniform(0.25, 0.85))
        for row, col in sites:
            if rng.random() < rate:
                scars[role, row, col] = True
        if not scars[role].any():
            row, col = sites[int(rng.integers(len(sites)))]
            scars[role, row, col] = True
    pieces = [_crop(mask, scar) for mask, scar in zip(masks, scars)]
    return pieces


def _scatter(rng, pieces, background, damage, colors, shape=None):
    max_h = max(mask.shape[0] for mask, _ in pieces)
    max_w = max(mask.shape[1] for mask, _ in pieces)
    if shape is None:
        band_h = int(rng.integers(max_h + 2, min(10, max_h + 4) + 1))
        band_w = max_w + int(rng.integers(2, 8))
    else:
        band_h, band_w = shape
    grid = np.full((3 * band_h, band_w), background, dtype=np.int8)
    for role, (mask, scar) in enumerate(pieces):
        ph, pw = mask.shape
        row = int(rng.integers(1, band_h - ph))
        col = int(rng.integers(1, band_w - pw))
        view = grid[role * band_h + row:role * band_h + row + ph, col:col + pw]
        view[mask] = colors[role]
        view[scar] = damage
    return grid


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(map(int, grid.flat))


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < 0.5)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    anchor_cycle = list(map(int, rng.permutation(3)))
    train_anchors = [anchor_cycle[k % 3] for k in range(num_train)]
    test_anchors = [int(rng.choice(sorted(set(train_anchors)))) for _ in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, anchors in (('train', train_anchors), ('test', test_anchors)):
        for index, anchor in enumerate(anchors):
            for _ in range(1800):
                if split == 'train':
                    height, width = int(rng.integers(4, 8)), int(rng.integers(4, 9))
                    if index == 1 and (height, width) == puzzle['train'][0]['output'].shape:
                        continue
                elif rng.random() < 0.5:
                    height, width = 8, int(rng.integers(5, 11))
                else:
                    height, width = int(rng.integers(4, 8)), int(rng.integers(9, 11))
                pieces = _sample_supports(rng, anchor, height, width)
                if pieces is None:
                    continue
                solutions, geometric, concealed = _layouts(pieces, height, width)
                if len(solutions) != 1 or min(geometric, concealed) < 2:
                    continue
                roles = palette if shared_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
                if not shared_palette and split == 'train' and index == 1:
                    first = _parse(puzzle['train'][0]['input'])
                    if roles == (first[3], first[4]) + first[5]:
                        continue
                background, damage, *colors = roles
                output = _render(pieces, solutions[0], height, width, background, damage, colors)
                naive = _render(pieces, ((0, 0),) * 3, height, width, background, damage, colors)
                if np.array_equal(naive, output):
                    continue
                input_grid = _scatter(rng, pieces, background, damage, colors)
                if _key(input_grid) in seen:
                    continue
                seen.add(_key(input_grid))
                puzzle[split].append({'input': input_grid, 'output': output})
                break
            else:
                raise RuntimeError("bounded sampling exhausted while seeking a unique assembly")
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
    x, y = _grid(pair['input']), _grid(pair['output'])
    pieces, height, width, bg, damage, colors, anchor = _parse(x)
    solutions, geometric, concealed = _layouts(pieces, height, width)
    _require(len(solutions) == 1 and min(geometric, concealed) >= 2,
             "coverage and depth must both resolve ambiguity")
    positions = solutions[0]
    expected = _render(pieces, positions, height, width, bg, damage, colors)
    _require(np.array_equal(y, expected), "incorrect registered composite")
    _require(set(map(int, np.unique(y))) == set(colors), "output must show all three inks only")
    _require(any(position != (0, 0) for position in positions), "nontrivial translation required")
    naive = _render(pieces, ((0, 0),) * 3, height, width, bg, damage, colors)
    _require(not np.array_equal(y, naive), "upper-left alignment shortcut")
    placed_mask, placed_scar = [], []
    for (mask, scar), (row, col) in zip(pieces, positions):
        a, b = np.zeros((height, width), bool), np.zeros((height, width), bool)
        ph, pw = mask.shape
        a[row:row + ph, col:col + pw] = mask
        b[row:row + ph, col:col + pw] = scar
        placed_mask.append(a)
        placed_scar.append(b)
    masks, scars = np.array(placed_mask), np.array(placed_scar)
    _require(bool(np.all(np.any(masks, axis=0))), "gapless rectangle required")
    _require(int(masks.sum()) > height * width, "overlaps required")
    _require(bool(scars[0].any() and scars[1].any() and not scars[2].any()), "damage roles")
    codes = 4 * masks[0].astype(int) + 2 * masks[1].astype(int) + masks[2].astype(int)
    for code, lower_roles in ((3, (1,)), (5, (0,)), (6, (0, 1))):
        _require(np.count_nonzero(codes == code) >= 2, "pairwise overlap evidence")
        witness = codes == code
        for role in lower_roles:
            witness = witness & ~scars[role]
        _require(bool(witness.any()), "ordinary-ink precedence witness missing")
    matching_orders = [order for order in itertools.permutations(range(3))
                       if np.array_equal(y, _render(pieces, positions, height, width,
                                                    bg, damage, colors, order))]
    _require(matching_orders == [_PAINT_ORDER], "precedence must be distinguishable")
    return {'pieces': pieces, 'positions': positions, 'shape': (height, width),
            'palette': (bg, damage) + colors, 'anchor': anchor, 'geometric': geometric,
            'concealed': concealed,
            'codes': codes, 'scars': scars}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "pair lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _count(a, 2, 10, 'num_train')
    _count(b, 1, 4, 'num_test')
    _require(a > b, "train count must exceed test count")
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            analysis = _analyze(pair)
            key = _key(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info[split].append(analysis)
    _require(info['train'][0]['anchor'] != info['train'][1]['anchor'],
             "first two demonstrations need different spanning roles")
    _require(info['train'][0]['shape'] != info['train'][1]['shape'],
             "first two demonstrations need different output sizes")
    if a >= 3:
        _require({p['anchor'] for p in info['train']} == {0, 1, 2}, "spanning role coverage")
    known = {p['anchor'] for p in info['train']}
    for p in info['train']:
        h, w = p['shape']
        _require(4 <= h <= 7 and 4 <= w <= 8, "training extent regime")
    for p in info['test']:
        h, w = p['shape']
        _require(p['anchor'] in known, "unshown test spanning role")
        _require((h == 8 and 5 <= w <= 10) or (4 <= h <= 7 and 9 <= w <= 10),
                 "test must extend the demonstrated size range")


def self_test():
    """Bounded checks across 42 seeds, with structural diversity and ablations."""
    batch = [generate_puzzle(seed) for seed in range(42)]
    shapes, palettes, structures, supports, backgrounds, damages = set(), set(), set(), set(), set(), set()
    scar_counts, overlap_counts, anchors, candidate_counts = set(), set(), set(), set()
    varied_palettes = shared_palettes = 0
    for seed, puzzle in enumerate(batch):
        validate_generated_puzzle(puzzle)
        local_palettes = set()
        for pair in puzzle['train'] + puzzle['test']:
            info = _analyze(pair)
            shapes.add(info['shape'])
            palettes.add(info['palette'])
            local_palettes.add(info['palette'])
            backgrounds.add(info['palette'][0])
            damages.add(info['palette'][1])
            anchors.add(info['anchor'])
            scar_counts.add(tuple(map(int, info['scars'].sum(axis=(1, 2)))))
            overlap_counts.add(tuple(int(np.count_nonzero(info['codes'] == code)) for code in (3, 5, 6, 7)))
            candidate_counts.add(info['geometric'])
            structures.add((info['shape'], tuple(map(int, info['codes'].flat)),
                            tuple(map(bool, info['scars'].flat))))
            for mask, scar in info['pieces']:
                supports.add((mask.shape, tuple(map(bool, mask.flat))))
            _require(np.array_equal(solve(np.array(pair['input']).copy()), pair['output']), "solve mismatch")
        varied_palettes += len(local_palettes) > 1
        shared_palettes += len(local_palettes) == 1
        # Independent source translations must not move anything in the answer.
        first = puzzle['test'][0]
        pieces, h, w, bg, damage, colors, _ = _parse(first['input'])
        shifted = _scatter(np.random.default_rng(seed + 10000), pieces, bg, damage, colors)
        _require(np.array_equal(solve(shifted), first['output']), "storage position affected answer")
    _require(len({_puzzle_key(p) for p in batch}) == 42, "duplicate puzzles")
    _require(len(shapes) >= 25 and len(palettes) >= 50, "shape/palette diversity")
    _require(len(structures) >= 190 and len(supports) >= 180, "support geometry diversity")
    _require(backgrounds == damages == set(range(10)) and anchors == {0, 1, 2}, "role diversity")
    _require(len(scar_counts) >= 35 and len(overlap_counts) >= 100 and len(candidate_counts) >= 10,
             "damage/overlap/registration diversity")
    _require(min(varied_palettes, shared_palettes) >= 8, "palette regime diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(20261016 + a * 10 + b, a, b)
            _require((len(puzzle['train']), len(puzzle['test'])) == (a, b), "explicit counts")
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_key(batch[0]) == _puzzle_key(generate_puzzle(0)), "nondeterminism")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
