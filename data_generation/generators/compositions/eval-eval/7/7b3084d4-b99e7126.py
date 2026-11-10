"""Chromatic echo jigsaw: jointly infer a rigid mosaic and its coarse echo.

No parent imports, source fixtures, files, seed state, or demonstrations are
consulted by solve. Construction and inference share explicit geometric helpers;
the supplied verifier therefore remains generator-owned verification.
"""
from functools import lru_cache
from itertools import product

import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "Integer grids required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _tight(a):
    r, c = np.nonzero(a >= 0)
    return a[r.min():r.max()+1, c.min():c.max()+1].copy()


def _connected(mask, neighbors=NEIGHBORS):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    pending = [cells.pop()]
    while pending:
        r, c = pending.pop()
        for dr, dc in neighbors:
            p = (r+dr, c+dc)
            if p in cells:
                cells.remove(p)
                pending.append(p)
    return not cells


def _orientations(piece):
    seen, result = set(), []
    for k in range(4):
        q = np.rot90(piece, k).copy()
        if _key(q) not in seen:
            seen.add(_key(q))
            result.append(q)
    return result


def _template_key(tile, marker):
    """Normalize carrier names but retain the marker's semantic identity."""
    roles = {int(marker): 0}
    return tile.shape, tuple(roles.setdefault(int(c), len(roles)) for c in tile.flat)


def _shape_orientations(mask):
    tight = _tight(np.where(mask, 1, -1))
    return len(_orientations(tight))


def _mask_symmetries(mask):
    """Relative scale transforms indistinguishable from identity up to shift."""
    def cropped_key(a):
        return _key(_tight(np.where(a, 1, -1)))
    transforms = [np.rot90(mask, k) for k in range(4)]
    transforms += [np.fliplr(a) for a in transforms]
    target = cropped_key(mask)
    return {i for i, a in enumerate(transforms) if cropped_key(a) == target}


def _coarse_sets(positions, side, rows, cols, clues):
    result = []
    for y in range(rows-side+1):
        for x in range(cols-side+1):
            active = frozenset((y+r, x+c) for r, c in positions)
            if clues < active:
                result.append(active)
    return result


def _search(pieces, side, marker, selector=None, lattice=None, clues=None,
            limit=2):
    """Exact cover; coarse filtering happens at selected-piece placement time.

    Returns (fully colored tile, active coarse set) pairs. With no selector it
    enumerates A-only assemblies, paired with None. Colored rotations are
    deduplicated without merging distinct marker orientations.
    """
    options = []
    for piece in pieces:
        marked = bool(np.any(piece == marker))
        selected = selector is not None and bool(np.any(piece == selector))
        seen, candidates = set(), []
        for q in _orientations(piece):
            h, w = q.shape
            if h > side or w > side or (marked and q[0, 0] != marker):
                continue
            ys = (0,) if marked else range(side-h+1)
            xs = (0,) if marked else range(side-w+1)
            for y in ys:
                for x in xs:
                    cells = tuple((int(r+y)*side+int(c+x), int(q[r, c]))
                                  for r, c in np.argwhere(q >= 0))
                    if cells in seen:
                        continue
                    seen.add(cells)
                    mask = sum(1 << pos for pos, _ in cells)
                    echoes = None
                    if selected:
                        positions = tuple(divmod(pos, side) for pos, _ in cells)
                        echoes = _coarse_sets(positions, side, *lattice, clues)
                        if not echoes:
                            continue
                    candidates.append((mask, cells, echoes))
        if not candidates:
            return []
        options.append(candidates)
    full = (1 << (side*side))-1
    placed, result, signatures = {}, [], set()

    def recurse(occupied, remaining):
        if limit is not None and len(result) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            tile = np.empty((side, side), dtype=np.int8)
            echoes = [None]
            for _, cells, coarse in placed.values():
                for pos, color in cells:
                    tile.flat[pos] = color
                if coarse is not None:
                    echoes = coarse
            for active in echoes:
                signature = (tile.tobytes(), active)
                if signature not in signatures:
                    signatures.add(signature)
                    result.append((tile.copy(), active))
                    if limit is not None and len(result) >= limit:
                        return
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                if candidate[0] & first and not candidate[0] & occupied:
                    placed[i] = candidate
                    recurse(occupied | candidate[0], rest)
                    if limit is not None and len(result) >= limit:
                        return
            placed.pop(i, None)

    recurse(0, tuple(range(len(pieces))))
    return result


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and i == result[-1][1]+1:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _read_input(grid):
    a = _grid(grid)
    divider = int(a[0, 0])
    yr = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    xr = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    _require(len(yr) >= 4 and len(xr) >= 4, "Missing regular lattice")
    _require(yr[0][0] == xr[0][0] == 0 and yr[-1][1] == a.shape[0]-1
             and xr[-1][1] == a.shape[1]-1, "Complete border required")
    thicknesses = {b-a+1 for a, b in yr+xr}
    gaps = {yr[i+1][0]-yr[i][1]-1 for i in range(len(yr)-1)}
    gaps |= {xr[i+1][0]-xr[i][1]-1 for i in range(len(xr)-1)}
    _require(len(thicknesses) == len(gaps) == 1, "Equal square interiors required")
    t, side = thicknesses.pop(), gaps.pop()
    rows, cols = len(yr)-1, len(xr)-1
    _require(side in (3, 4) and t in (1, 2) and min(rows, cols) >= side,
             "Unsupported lattice geometry")
    tiles, uniform = {}, {}
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(side+t), t+c*(side+t)
            tile = a[y:y+side, x:x+side].copy()
            _require(divider not in tile, "Divider may not occur inside tiles")
            tiles[r, c] = tile
            if np.all(tile == tile[0, 0]):
                uniform.setdefault(int(tile[0, 0]), []).append((r, c))
    _require(len(uniform) == 2, "Exactly empty and solid clue tile types required")
    ordered = sorted(uniform, key=lambda col: len(uniform[col]), reverse=True)
    background, selector = ordered
    _require(len(uniform[background]) > len(uniform[selector]) >= 2,
             "Ordinary blanks must outnumber at least two solid clues")
    sources, pieces = [], []
    for coord, tile in tiles.items():
        if np.all(tile == tile[0, 0]):
            continue
        occupied = tile != background
        _require(0 < int(occupied.sum()) < side*side and _connected(occupied),
                 "One corner-connected incomplete piece required per source tile")
        sources.append(coord)
        pieces.append(_tight(np.where(occupied, tile, -1)))
    _require(2 <= len(pieces) <= 5 and sum(int((p >= 0).sum()) for p in pieces) == side*side,
             "Source inventory must fill exactly one square")
    inventory = np.concatenate([p[p >= 0] for p in pieces])
    colors, counts = np.unique(inventory, return_counts=True)
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1]
    _require(len(singletons) == 1, "Exactly one source singleton marker required")
    marker = singletons[0]
    _require(marker not in (divider, background, selector), "Marker has its own role")
    carriers, anchor_carrier = [], None
    for piece in pieces:
        values = set(map(int, piece[piece >= 0]))
        marked = marker in values
        carrier = values-{marker}
        _require(len(carrier) == 1, "Each source piece has its own carrier")
        carrier = carrier.pop()
        _require(carrier not in (divider, background), "Carrier roles must differ")
        _require(int((piece >= 0).sum()) >= (3 if marked else 2), "Piece too small")
        carriers.append(carrier)
        if marked:
            anchor_carrier = carrier
    _require(len(set(carriers)) == len(pieces), "Distinct source carriers required")
    _require(selector in carriers and selector != anchor_carrier,
             "Clues select an unmarked source carrier")
    _require(int(np.sum(inventory == selector)) >= 3, "Stencil needs at least three cells")
    return dict(grid=a, side=side, thickness=t, rows=rows, cols=cols,
                divider=divider, background=background, selector=selector,
                marker=marker, anchor_carrier=anchor_carrier, carriers=carriers,
                pieces=pieces, sources=tuple(sources), clues=frozenset(uniform[selector]))


def _render(facts, tile, active):
    side, t = facts['side'], facts['thickness']
    result = np.full(facts['grid'].shape, facts['divider'], dtype=np.int8)
    for r in range(facts['rows']):
        for c in range(facts['cols']):
            y, x = t+r*(side+t), t+c*(side+t)
            result[y:y+side, x:x+side] = tile if (r, c) in active else facts['background']
    return result


def _infer(input_grid):
    f = _read_input(input_grid)
    answers = _search(f['pieces'], f['side'], f['marker'], f['selector'],
                      (f['rows'], f['cols']), f['clues'], limit=2)
    _require(len(answers) == 1, "Require exactly one joint mosaic/completion")
    tile, active = answers[0]
    return _render(f, tile, active), f, tile, active


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone; never mutate it."""
    return _infer(input_grid)[0]


def _relabel(labels):
    order = [int(labels[0, 0])]+[int(i) for i in np.unique(labels) if i != labels[0, 0]]
    result = labels.copy()
    for i, old in enumerate(order):
        result[labels == old] = i
    return result


def _perturb(rng, labels, steps):
    count = int(labels.max())+1
    for _ in range(steps):
        r, c = map(int, rng.integers(labels.shape[0], size=2))
        dr, dc = EDGE[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if (r, c) == (0, 0) or not (0 <= rr < len(labels) and 0 <= cc < len(labels)):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.sum(labels == a) <= (3 if a == 0 else 2):
            continue
        labels[r, c] = b
        if not all(_connected(labels == i) for i in range(count)):
            labels[r, c] = a
    return labels


def _partition(rng, side, count):
    style = int(rng.choice(5, p=(.35, .25, .15, .15, .10)))
    diagonal = bool(rng.random() < .22)
    labels = np.full((side, side), -1, dtype=np.int8)
    if style <= 1 or (style == 2 and count > side):
        points = [0]+list(map(int, rng.choice(np.arange(1, side*side), count-1, replace=False)))
        for i, pos in enumerate(points):
            labels.flat[pos] = i
        weights = np.ones(count) if style == 0 else np.exp(rng.uniform(-1.3, 1.3, count))
        while np.any(labels < 0):
            frontiers = []
            for i in range(count):
                neighbors = NEIGHBORS if diagonal else EDGE
                frontier = sorted({(int(r)+dr, int(c)+dc)
                                   for r, c in np.argwhere(labels == i) for dr, dc in neighbors
                                   if 0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] < 0})
                frontiers.append(frontier)
            i = min((i for i in range(count) if frontiers[i]),
                    key=lambda j: np.sum(labels == j)/weights[j]+float(rng.random())*.6)
            choices = frontiers[i]
            if diagonal and rng.random() < .8:
                edge_choices = [(r, c) for r, c in choices if any(
                    0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] == i
                    for dr, dc in EDGE)]
                if edge_choices:
                    choices = edge_choices
            r, c = choices[int(rng.integers(len(choices)))]
            labels[r, c] = i
    elif style == 2:
        breaks = sorted(map(int, rng.choice(np.arange(1, side), count-1, replace=False)))
        for i, (a, b) in enumerate(zip([0]+breaks, breaks+[side])):
            labels[a:b] = i
        if rng.integers(2):
            labels = labels.T.copy()
        labels = _perturb(rng, labels, int(rng.integers(side+1)))
    elif style == 3:
        rectangles = [(0, 0, side, side)]
        while len(rectangles) < count:
            cuts = []
            for i, (r, c, h, w) in enumerate(rectangles):
                cuts.extend((i, 0, k) for k in range(1, h) if min(k, h-k)*w >= 2)
                cuts.extend((i, 1, k) for k in range(1, w) if min(k, w-k)*h >= 2)
            if not cuts:
                return None
            i, axis, k = cuts[int(rng.integers(len(cuts)))]
            r, c, h, w = rectangles.pop(i)
            rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                              [(r, c, h, k), (r, c+k, h, w-k)])
        for i, (r, c, h, w) in enumerate(rectangles):
            labels[r:r+h, c:c+w] = i
        labels = _perturb(rng, _relabel(labels), int(rng.integers(side*3+1)))
    else:
        labels[:] = 1
        if rng.integers(2):
            labels[0, :3] = 0
        else:
            labels[0, :2] = 0
            labels[1, 0] = 0
        for i in range(2, count):
            found = False
            for _ in range(20):
                free = list(map(tuple, np.argwhere(labels == 1)))
                if len(free) < 4:
                    break
                start = free[int(rng.integers(len(free)))]
                carved = {start}
                for _ in range(int(rng.integers(1, 4))):
                    choices = sorted({(r+dr, c+dc) for r, c in carved for dr, dc in EDGE
                                      if 0 <= r+dr < side and 0 <= c+dc < side
                                      and labels[r+dr, c+dc] == 1 and (r+dr, c+dc) not in carved})
                    if choices:
                        carved.add(choices[int(rng.integers(len(choices)))])
                trial = labels.copy()
                for r, c in carved:
                    trial[r, c] = i
                if len(carved) >= 2 and np.sum(trial == 1) >= 2 and _connected(trial == 1):
                    labels = trial
                    found = True
                    break
            if not found:
                return None
    labels = _relabel(labels)
    if np.sum(labels == 0) < 3 or any(np.sum(labels == i) < 2 for i in range(1, count)):
        return None
    if not all(_connected(labels == i) for i in range(count)):
        return None
    return labels


def _pieces(labels, carriers, marker):
    colored = np.array(carriers, dtype=np.int8)[labels]
    colored[0, 0] = marker
    return [_tight(np.where(labels == i, colored, -1)) for i in range(len(carriers))]


@lru_cache(maxsize=14)
def _subset_codes(n):
    codes = np.arange(1, 1 << n, dtype=np.uint32)
    sizes = np.array([int(code).bit_count() for code in codes], dtype=np.int8)
    codes.flags.writeable = sizes.flags.writeable = False
    return codes, sizes


def _choose_fragment(tiles, target, selector, geometry, piece_count, rng, sparse):
    side, t, rows, cols = geometry
    points = tuple(map(tuple, np.argwhere(target == selector)))
    n = len(points)
    codes, sizes = _subset_codes(n)
    max_visible = min(n-1, (rows*cols-piece_count-1)//2)
    keep = (sizes >= 2) & (sizes <= max_visible)
    if sparse is True:
        keep &= sizes*2 <= n
    elif sparse is False:
        keep &= sizes*2 > n
    if not keep.any():
        return None
    translations = list(product(range(rows-side+1), range(cols-side+1)))
    for k in rng.permutation(len(translations)):
        y, x = translations[int(k)]
        active = tuple((int(r)+y, int(c)+x) for r, c in points)
        valid = codes[keep].copy()
        full = (1 << n)-1
        covers = set()
        for tile in tiles:
            mask_points = tuple(map(tuple, np.argwhere(tile == selector)))
            for yy, xx in translations:
                other = frozenset((int(r)+yy, int(c)+xx) for r, c in mask_points)
                if np.array_equal(tile, target) and (yy, xx) == (y, x):
                    continue
                cover = sum(1 << i for i, pos in enumerate(active) if pos in other)
                covers.add(cover)
        if full in covers:
            continue
        for cover in sorted(covers, reverse=True):
            valid = valid[(valid & np.uint32(cover)) != valid]
            if not len(valid):
                break
        if len(valid):
            possible_sizes = sorted({int(code).bit_count() for code in valid})
            size = int(rng.choice(possible_sizes))
            code = int(rng.choice([v for v in valid if int(v).bit_count() == size]))
            clues = frozenset(pos for i, pos in enumerate(active) if code & (1 << i))
            return frozenset(active), clues
    return None


def _blueprint(labels, carriers, marker, geometry, rng, sparse, excluded,
               small_selector=False, asymmetric=False, prior_symmetries=None):
    side = len(labels)
    pieces = _pieces(labels, carriers, marker)
    assemblies = _search(pieces, side, marker, limit=64)
    if not 2 <= len(assemblies) < 64:
        return None
    tiles = [a for a, _ in assemblies]
    areas = [int((p >= 0).sum()) for p in pieces]
    selections = [carriers[i] for i in range(1, len(carriers)) if areas[i] >= 3
                  and (not small_selector or areas[i] < max(areas))]
    for i in rng.permutation(len(tiles)):
        tile = tiles[int(i)]
        if _template_key(tile, marker) in excluded:
            continue
        for selector in rng.permutation(selections):
            selector = int(selector)
            if asymmetric and _shape_orientations(tile == selector) != 4:
                continue
            if (prior_symmetries is not None and
                    _mask_symmetries(tile == selector) & prior_symmetries != {0}):
                continue
            chosen = _choose_fragment(tiles, tile, selector, geometry, len(pieces), rng, sparse)
            if chosen is not None:
                active, clues = chosen
                # Scatter relative to the selected assembly, so a forced
                # rotation cannot accidentally cancel a construction rotation.
                final_pieces = []
                for i, carrier in enumerate(carriers):
                    mask = tile == carrier
                    if i == 0:
                        mask |= tile == marker
                    final_pieces.append(_tight(np.where(mask, tile, -1)))
                return final_pieces, tile, selector, active, clues
    return None


def _scatter(rng, blueprint, geometry, palette, require_selector_rotation):
    pieces, tile, selector, active, clues = blueprint
    side, t, rows, cols = geometry
    background, divider, marker = palette[:3]
    grid = np.full((rows*(side+t)+t, cols*(side+t)+t), divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y, x = t+r*(side+t), t+c*(side+t)
            grid[y:y+side, x:x+side] = selector if (r, c) in clues else background
    available = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in clues]
    # Source locations witness both consumption at inactive sites and copying
    # at missing active sites, without imposing a preferred row or column.
    missing, inactive = sorted(active-clues), sorted(set(available)-active)
    chosen = [missing[int(rng.integers(len(missing)))], inactive[int(rng.integers(len(inactive)))]]
    rest = [p for p in available if p not in chosen]
    chosen += [rest[int(i)] for i in rng.choice(len(rest), len(pieces)-2, replace=False)]
    chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]
    rotated = False
    for i, (piece, (r, c)) in enumerate(zip(pieces, chosen)):
        ks = list(range(4))
        if require_selector_rotation and np.any(piece == selector):
            ks = [k for k in ks if not np.array_equal(np.rot90(piece, k), piece)]
        if (np.any(piece == marker) or i == len(pieces)-1) and not rotated:
            changed = [k for k in ks if not np.array_equal(np.rot90(piece, k), piece)]
            if changed:
                ks = changed
        k = int(rng.choice(ks))
        q = np.rot90(piece, k)
        rotated |= not np.array_equal(q, piece)
        y = t+r*(side+t)+int(rng.integers(side-q.shape[0]+1))
        x = t+c*(side+t)+int(rng.integers(side-q.shape[1]+1))
        view = grid[y:y+q.shape[0], x:x+q.shape[1]]
        view[q >= 0] = q[q >= 0]
    facts = dict(grid=grid, side=side, thickness=t, rows=rows, cols=cols,
                 background=background, divider=divider)
    return {'input': grid, 'output': _render(facts, tile, active)}


def _geometry(rng, side):
    t = int(rng.choice((1, 2), p=(.75, .25)))
    largest = (30-t)//(side+t)
    rows = int(rng.integers(side, largest+1))
    cols = int(rng.integers(side, largest+1))
    return side, t, rows, cols


@lru_cache(maxsize=2)
def _fallback_labels(side):
    """Finite catalog from all small connected partitions, not solved examples.

    Used only if the procedural sampler exhausts its budget. Side-four variants
    extend the marked piece into the extra row and column; every candidate still
    undergoes the same ambiguity and joint-uniqueness checks.
    """
    result = []
    for tail in product(range(3), repeat=8):
        a = np.array((0,)+tail, dtype=np.int8).reshape(3, 3)
        if np.sum(a == 0) < 3 or any(np.sum(a == i) < 2 for i in (1, 2)):
            continue
        if not all(_connected(a == i) for i in range(3)):
            continue
        if side == 4:
            b = np.zeros((4, 4), dtype=np.int8)
            b[:3, :3] = a
            a = b
        if all(_connected(a == i) for i in range(3)):
            result.append(a)
    # Marked bars offer a guaranteed sparse completion at either scale.
    for transpose in (False, True):
        a = np.ones((side, side), dtype=np.int8)
        a[0] = 0
        result.append(a.T.copy() if transpose else a)
    return tuple(result)


def _make_pair(rng, side, sparse, base_palette, stable_palette, excluded,
               small_selector, asymmetric, require_selector_rotation, prior_symmetries=None):
    for attempt in range(384):
        count = int(rng.choice((2, 3, 4) if side == 3 else (2, 3, 4, 5),
                               p=(.2, .6, .2) if side == 3 else (.15, .45, .3, .1)))
        if small_selector and side == 3:
            count = 3
        labels = _partition(rng, side, count)
        if labels is None:
            continue
        palette = base_palette if stable_palette else tuple(map(int, rng.permutation(10)))
        carriers, marker = list(palette[3:3+count]), palette[2]
        geometry = _geometry(rng, side)
        blueprint = _blueprint(labels, carriers, marker, geometry, rng, sparse,
                               excluded, small_selector, asymmetric, prior_symmetries)
        if blueprint is not None:
            return _scatter(rng, blueprint, geometry, palette, require_selector_rotation)
    # A small coarse board fixes the stencil's origin; it still cannot select
    # between different colored rigid assemblies without the actual clue cells.
    geometry = (side, 1, side, side)
    catalog = _fallback_labels(side)
    for index in rng.permutation(len(catalog)):
        labels = catalog[int(index)]
        count = int(labels.max())+1
        palette = base_palette if stable_palette else tuple(map(int, rng.permutation(10)))
        blueprint = _blueprint(labels, list(palette[3:3+count]), palette[2], geometry,
                               rng, sparse, excluded, small_selector, asymmetric, prior_symmetries)
        if blueprint is not None:
            return _scatter(rng, blueprint, geometry, palette, require_selector_rotation)
    raise RuntimeError("No valid blueprint survived the bounded construction")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                  (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    scale_shift = bool(rng.integers(2))
    stable_palette = bool(rng.random() < .30)
    base_palette = tuple(map(int, rng.permutation(10)))
    puzzle, used, training_templates = {'train': [], 'test': []}, set(), set()
    first_symmetries = None
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            side = 3 if scale_shift and split == 'train' else 4
            sparse = None if scale_shift else split == 'test'
            excluded = training_templates if split == 'test' or i == 1 else set()
            for _ in range(16):
                pair = _make_pair(rng, side, sparse, base_palette, stable_palette, excluded,
                                  small_selector=(split == 'train' and i == 0),
                                  asymmetric=(split == 'train' and i < 2),
                                  require_selector_rotation=(split == 'train' and i == 0),
                                  prior_symmetries=(first_symmetries if split == 'train' and i == 1 else None))
                if _key(pair['input']) not in used:
                    break
            else:
                raise RuntimeError('Unable to construct a distinct input')
            puzzle[split].append(pair)
            used.add(_key(pair['input']))
            if split == 'train':
                _, f, tile, _ = _infer(pair['input'])
                training_templates.add(_template_key(tile, f['marker']))
                if i == 0:
                    first_symmetries = _mask_symmetries(tile == f['selector'])
    return puzzle


def _features(input_grid, tile, active, facts):
    f = facts
    selected_source = next(p for p in f['pieces'] if np.any(p == f['selector']))
    selected_output = _tight(np.where(tile == f['selector'], tile, -1))
    rotation = False
    chiral = False
    for p, carrier in zip(f['pieces'], f['carriers']):
        mask = tile == carrier
        if carrier == f['anchor_carrier']:
            mask |= tile == f['marker']
        q = _tight(np.where(mask, tile, -1))
        _require(any(np.array_equal(q, v) for v in _orientations(p)),
                 "A piece was deformed or reflected")
        rotation |= not np.array_equal(p, q)
        reflected = np.fliplr(p)
        chiral |= not any(np.array_equal(reflected, v) for v in _orientations(p))
    stencil = tile == f['selector']
    count = int(stencil.sum())
    return dict(side=f['side'], thickness=f['thickness'], lattice=(f['rows'], f['cols']),
                shape=input_grid.shape, pieces=len(f['pieces']), area=count,
                clue_count=len(f['clues']), missing=len(active-f['clues']),
                selector=f['selector'], palette=tuple(map(int, np.unique(input_grid))),
                roles=(f['background'], f['divider'], f['marker'], f['selector']),
                stencil=_key(stencil), template=_template_key(tile, f['marker']),
                symmetries=_mask_symmetries(stencil),
                selected_rotated=not np.array_equal(selected_source, selected_output),
                rotated=rotation, asymmetric=_shape_orientations(stencil) == 4,
                chiral=chiral, diagonal=any(not _connected(p >= 0, EDGE) for p in f['pieces']),
                small_selector=count < max(int((p >= 0).sum()) for p in f['pieces']),
                source_active=bool(set(f['sources']) & (active-f['clues'])),
                source_inactive=bool(set(f['sources'])-active),
                fragment=tuple(sorted(f['clues'])), active=tuple(sorted(active)),
                area_ratio=max(int((p >= 0).sum()) for p in f['pieces']) /
                           min(int((p >= 0).sum()) for p in f['pieces']))


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    expected, f, tile, active = _infer(inp)
    _require(np.array_equal(out, expected), "Incorrect complete-tile echo")
    _require(inp.shape == out.shape, "Preserve the lattice dimensions")
    _require(tile[0, 0] == f['marker'] and int(np.sum(tile == f['marker'])) == 1,
             "One preserved upper-left marker per copy")
    inventory = np.concatenate([p[p >= 0] for p in f['pieces']])
    _require(np.array_equal(np.sort(inventory), np.sort(tile.flat)),
             "Prototype must preserve all source colors and cells")
    alternatives = _search(f['pieces'], f['side'], f['marker'], limit=2)
    _require(len(alternatives) == 2, "Coarse clues must resolve A-only ambiguity")
    alt = next(a for a, _ in alternatives if not np.array_equal(a, tile))
    _require(not np.array_equal(_render(f, alt, active), out),
             "Omitting the coarse constraint must change an available answer")
    solid = np.full(tile.shape, f['selector'], dtype=np.int8)
    _require(not np.array_equal(_render(f, solid, active), out),
             "Omitting rigid mosaic reconstruction must change the answer")
    _require(f['clues'] < active, "Clues must be an incomplete echo")
    _require(len(active) == int(np.sum(tile == f['selector'])), "Same footprint at both scales")
    return _features(inp, tile, active, f)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _validate_pair(pair)
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, "Repeated input")
            seen.add(key)
            facts[split].append(f)
    train, test = facts['train'], facts['test']
    _require(train[0]['template'] != train[1]['template'], "Two distinct teaching mosaics required")
    _require(all(f['rotated'] for f in train+test), "Demonstrate genuine rigid rotations")
    _require(any(f['selected_rotated'] for f in train), "Raw selector orientation shortcut needs a witness")
    _require(any(f['small_selector'] for f in train), "Largest-piece shortcut needs a witness")
    _require(any(f['asymmetric'] for f in train), "Demonstrate orientation correspondence")
    orientation_rules = set(range(8))
    for f in train[:2]:
        orientation_rules &= f['symmetries']
    _require(orientation_rules == {0}, "First two examples must distinguish relative scale orientation")
    _require(all(f['source_active'] and f['source_inactive'] for f in train+test),
             "Demonstrate source consumption both inside and outside the echo")
    _require(all(f['template'] not in {a['template'] for a in train} for f in test),
             "Tests must use new normalized mosaics")
    scale = all(f['side'] == 3 for f in train) and all(f['side'] == 4 for f in test)
    sparse = (all(f['side'] == 4 and 2*f['clue_count'] > f['area'] for f in train)
              and all(f['side'] == 4 and 2*f['clue_count'] <= f['area'] for f in test))
    _require(scale or sparse, "Missing the documented scale or sparsity shift")
    return None


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                              for p in puzzle[split])) for split in ('train', 'test'))


def self_test():
    """Bounded semantic and structural checks on 42 seeds, plus negative cases."""
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_puzzle_key(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                records.append((split, _validate_pair(pair)))
    _require(len(signatures) == 42, "Seeds must produce distinct puzzles")
    fs = [f for _, f in records]
    _require({f['side'] for f in fs} == {3, 4}, "Exercise both microscopic scales")
    _require({f['pieces'] for f in fs} == {2, 3, 4, 5}, "Exercise all source-piece counts")
    _require({f['thickness'] for f in fs} == {1, 2}, "Exercise both divider thicknesses")
    _require(len({f['shape'] for f in fs}) >= 20, "Vary actual lattice geometry")
    _require(len({f['template'] for f in fs}) >= 80, "Vary colored partition structure")
    _require(len({f['stencil'] for f in fs}) >= 40, "Vary selector footprints")
    _require(len({f['roles'] for f in fs}) >= 80, "Vary visible palette roles")
    _require(any(f['diagonal'] for f in fs), "Exercise corner attachments")
    _require(any(f['chiral'] for f in fs), "Exercise handed piece patterns")
    _require(max(f['area_ratio'] for f in fs) >= 4, "Exercise unequal piece areas")
    _require(any(f['missing'] == 1 for f in fs), "Retain one-missing-tile completions")
    _require(any(2*f['clue_count'] < f['area'] for f in fs), "Exercise genuinely sparse clues")
    for split in ('train', 'test'):
        local = [f for s, f in records if s == split]
        _require(any(f['lattice'][0] != f['lattice'][1] for f in local), "Rectangular lattices required")
    p = generate_puzzle(711, 3, 2)
    _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(711, 3, 2)), "Determinism failed")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(701+a*10+b, a, b)
            _require(len(q['train']) == a and len(q['test']) == b, "Explicit counts ignored")
            validate_generated_puzzle(q)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    pair = p['test'][0]
    before = np.asarray(pair['input']).copy()
    _require(np.array_equal(solve(before), pair['output']) and np.array_equal(before, pair['input']),
             "Solver must be input-only and nonmutating")
    damaged = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')
    # An incorrect ink-only copy changes a nonselector cell of an active tile.
    _, f, tile, active = _infer(pair['input'])
    r, c = sorted(active-f['clues'])[0]
    dr, dc = map(int, np.argwhere(tile != f['selector'])[0])
    ink_only = {k: np.array(v, copy=True) for k, v in pair.items()}
    y = f['thickness']+r*(f['side']+f['thickness'])+dr
    x = f['thickness']+c*(f['side']+f['thickness'])+dc
    ink_only['output'][y, x] = f['background']
    try:
        _validate_pair(ink_only)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incomplete full-tile copying accepted')
    # A well-formed two-clue input has two fitting registrations. A complete
    # subset match cannot be resolved by largest overlap or enumeration order.
    labels = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=np.int8)
    proto = np.array([[9, 2, 2], [3, 3, 3], [3, 3, 3]], dtype=np.int8)
    active = frozenset((r, c) for r in (1, 2) for c in (2, 3, 4))
    clues = frozenset(((2, 2), (2, 4)))
    blueprint = (_pieces(labels, [2, 3], 9), proto, 3, active, clues)
    ambiguous = _scatter(np.random.default_rng(8), blueprint, (3, 1, 6, 6),
                         (0, 1, 9, 2, 3), False)['input']
    f_ambiguous = _read_input(ambiguous)
    _require(len(_search(f_ambiguous['pieces'], 3, 9, 3, (6, 6), clues, limit=2)) == 2,
             "Negative fixture must exhibit real joint ambiguity")
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('A two-clue registration tie was accepted')
    # Remove all but one solid clue from an otherwise valid input. This violates
    # the visible family convention; inference may not silently pick a packing.
    _, f, _, _ = _infer(pair['input'])
    ambiguous = np.array(pair['input'], copy=True)
    side, t = f['side'], f['thickness']
    for r, c in sorted(f['clues'])[1:]:
        y, x = t+r*(side+t), t+c*(side+t)
        ambiguous[y:y+side, x:x+side] = f['background']
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('Insufficient clues accepted')
    # Conditional variation: identical inventory, palette and lattice may admit
    # many different registered fragments, all under the same rule.
    labels = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=np.int8)
    pieces = _pieces(labels, [2, 3], 9)
    tiles = [a for a, _ in _search(pieces, 3, 9, limit=64)]
    fragments = set()
    for seed in range(42):
        chosen = _choose_fragment(tiles, tiles[0], 3, (3, 1, 6, 6), 2,
                                  np.random.default_rng(seed), None)
        _require(chosen is not None, "Constructive conditional witness failed")
        fragments.add(chosen[1])
    _require(len(fragments) >= 15, "Fragment variation must exceed palette changes")
    return {'seeds': 42, 'pairs': len(fs), 'distinct_templates': len({f['template'] for f in fs}),
            'distinct_stencils': len({f['stencil'] for f in fs}),
            'grid_shapes': len({f['shape'] for f in fs}),
            'piece_counts': sorted({f['pieces'] for f in fs}),
            'conditional_fragments': len(fragments)}
