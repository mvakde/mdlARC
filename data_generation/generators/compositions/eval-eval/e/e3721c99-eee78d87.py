"""Print a swatch through apertures; classify the seam-joined ink by topology.

The solver reads only pixels. All color mappings are supplied by cavity glyphs
in the input. The framed 3x3 swatch, not an aperture silhouette, is the tile.
"""
from __future__ import annotations

from itertools import permutations
import numpy as np


def _components(mask):
    """Four-neighbor components, in row-major discovery order."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    # Flat mutable bytes avoid NumPy scalar indexing in the flood-fill loop.
    # A zero sentinel border makes all four neighbor offsets safe. Discovery
    # and stack order remain identical to the original coordinate traversal.
    stride = w + 2
    remaining = bytearray((h+2)*stride)
    raw = mask.tobytes()
    for r in range(h):
        start = (r+1)*stride+1
        remaining[start:start+w] = raw[r*w:(r+1)*w]
    result = []
    for start in range(stride+1, (h+1)*stride):
        if not remaining[start]:
            continue
        remaining[start] = 0
        pending, cells = [start], []
        while pending:
            point = pending.pop()
            y, x = divmod(point, stride)
            cells.append((y-1, x-1))
            for neighbor in (point-stride, point+stride, point-1, point+1):
                if remaining[neighbor]:
                    remaining[neighbor] = 0
                    pending.append(neighbor)
        result.append(cells)
    return result


def _crop(cells):
    points = np.asarray(cells, int)
    lo, hi = points.min(0), points.max(0) + 1
    mask = np.zeros(tuple(hi-lo), bool)
    mask[tuple((points-lo).T)] = True
    return mask, tuple(int(x) for x in lo)


def _holes(mask):
    padded = np.pad(np.asarray(mask, bool), 1)
    h, w = padded.shape
    return [c for c in _components(~padded)
            if all(0 < r < h-1 and 0 < x < w-1 for r, x in c)]


def _hole_count(mask):
    return len(_holes(mask))


def _full_box(mask):
    return (mask.shape == (3, 3) and 1 < int(mask.sum()) < 9
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _ink(aperture, tile, order=(1, 2, 0), transpose=False):
    t = tile.T if transpose else tile
    t = t[np.ix_(order, order)]
    rows, cols = np.indices(aperture.shape)
    return aperture & t[rows % 3, cols % 3]


def _printed_components(aperture, tile, order=(1, 2, 0), transpose=False):
    ink = _ink(aperture, tile, order, transpose)
    result = []
    for cells in _components(ink):
        mask, pos = _crop(cells)
        cavities = _holes(mask)
        # Enclosed regions must really be carrier, not a separate ink island.
        # This holds for every non-solid 3x3 periodic motif and our apertures;
        # retain the explicit check instead of assuming it in validation.
        clear = all(not ink[pos[0]+r-1, pos[1]+c-1]
                    for hole in cavities for r, c in hole)
        result.append(dict(cells=cells, mask=mask, holes=len(cavities), clear=clear))
    return ink, result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('grids must be integer 2D arrays, dimensions 1..30, colors 0..9')
    return a.astype(np.int8, copy=True)


def _parse(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    candidates = []
    barriers = []
    for r in range(1, h-1):
        if np.all(grid[r] == grid[r, 0]):
            barriers.append((int(grid[r, 0]), (r, 0, 1, w),
                             ((0, 0, r, w), (r+1, 0, h-r-1, w))))
    for c in range(1, w-1):
        if np.all(grid[:, c] == grid[0, c]):
            barriers.append((int(grid[0, c]), (0, c, h, 1),
                             ((0, 0, h, c), (0, c+1, h, w-c-1))))
    for sep, barrier_box, regions in barriers:
        for pi in (0, 1):
            pr, pc, ph, pw = regions[pi]
            wr, wc, wh, ww = regions[1-pi]
            panel = grid[pr:pr+ph, pc:pc+pw]
            work = grid[wr:wr+wh, wc:wc+ww]
            # The existing frame requirement permits exactly 16 separator
            # pixels in the panel. Reject other counts before flood filling.
            separator_pixels = panel == sep
            if np.count_nonzero(separator_pixels) != 16:
                continue
            frames = []
            for cells in _components(separator_pixels):
                fm, (r, c) = _crop(cells)
                if fm.shape != (5, 5):
                    continue
                perimeter = np.ones((5, 5), bool)
                perimeter[1:4, 1:4] = False
                if np.array_equal(fm, perimeter):
                    frames.append((r, c))
            if len(frames) != 1:
                continue
            fr, fc = frames[0]
            swatch = panel[fr+1:fr+4, fc+1:fc+4]
            colors = [int(c) for c in np.unique(swatch)]
            if len(colors) != 2 or sep in colors:
                continue
            for bg, query in (colors, colors[::-1]):
                tile = swatch == query
                if not _full_box(tile) or not np.all(np.isin(work, (bg, query))):
                    continue
                inside = np.zeros(panel.shape, bool)
                inside[fr+1:fr+4, fc+1:fc+4] = True
                if np.any((panel == query) & ~inside):
                    continue
                mapping, entries, valid = {}, [], True
                for color in np.unique(panel):
                    color = int(color)
                    if color in (bg, query, sep):
                        continue
                    comps = _components(panel == color)
                    if len(comps) != 1:
                        valid = False
                        break
                    glyph, pos = _crop(comps[0])
                    cavities = _holes(glyph)
                    k = len(cavities)
                    if not 0 <= k <= 6 or k in mapping:
                        valid = False
                        break
                    if any(panel[pos[0]+r-1, pos[1]+c-1] != bg
                           for hole in cavities for r, c in hole):
                        valid = False
                        break
                    mapping[k] = color
                    entries.append((pos, k, color))
                if not valid or not 2 <= len(mapping) <= 3:
                    continue
                apertures = []
                for cells in _components(work == query):
                    mask, pos = _crop(cells)
                    # No other aperture may be nested in its enclosed regions.
                    if any(work[pos[0]+r-1, pos[1]+c-1] != bg
                           for hole in _holes(mask) for r, c in hole):
                        valid = False
                        break
                    ink, components = _printed_components(mask, tile)
                    if not all(c['clear'] for c in components):
                        valid = False
                        break
                    apertures.append(dict(mask=mask, pos=pos, cells=cells,
                                          ink=ink, components=components))
                if not valid or not apertures:
                    continue
                candidates.append(dict(grid=grid, bg=bg, query=query, sep=sep,
                                       tile=tile.copy(), mapping=mapping, entries=entries,
                                       apertures=apertures, work_box=(wr, wc, wh, ww),
                                       panel_box=regions[pi], barrier_box=barrier_box))
    if len(candidates) != 1:
        raise ValueError('input must have one framed-swatch/legend/aperture interpretation')
    return candidates[0]


def _render(info, order=(1, 2, 0), transpose=False):
    out = info['grid'].copy()
    wr, wc, _, _ = info['work_box']
    for a in info['apertures']:
        r, c = a['pos']
        patch = out[wr+r:wr+r+a['mask'].shape[0], wc+c:wc+c+a['mask'].shape[1]]
        patch[a['mask']] = info['bg']
        # Parsing already computed the direct print from these input pixels.
        # Alternative phase checks still recompute their own topology.
        components = (a['components'] if order == (1, 2, 0) and not transpose else
                      _printed_components(a['mask'], info['tile'], order, transpose)[1])
        for component in components:
            color = info['mapping'].get(component['holes'], info['bg'])
            patch[tuple(np.asarray(component['cells']).T)] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The complete transformation, using no examples or generation state."""
    return _render(_parse(input_grid))


# Enumerated masks are data-free procedural possibilities, not parent fixtures.
_TILES = [np.array([(code >> bit) & 1 for bit in range(9)], bool).reshape(3, 3)
          for code in range(1, 511)]
_TILES = [m for m in _TILES if _full_box(m)]


def _sampling_pools():
    printable, teaching, annular = [], [], []
    for tile in _TILES:
        if not any(c['holes'] > 0 for c in _printed_components(np.ones((9, 9), bool), tile)[1]):
            continue
        printable.append(tile)
        if _hole_count(tile) == 0 and not any(
                np.array_equal(tile, x) for x in (tile.T, tile[::-1], tile[:, ::-1])):
            teaching.append(tile)
        for h, w in ((9, 9), (9, 10), (10, 9), (10, 10)):
            aperture = np.ones((h, w), bool)
            aperture[1:-1, 1:-1] = False
            aperture[1:3, 1:3] = True
            if any(0 < c['holes'] <= 6 for c in _printed_components(aperture, tile)[1]):
                annular.append(tile)
                break
    return printable, teaching, annular


_PRINT_TILES, _TEACHING_TILES, _ANNULAR_TILES = _sampling_pools()


def _window_bank(rng, tile, maximum):
    bank, seen = [], set()
    def add(mask, kind):
        mask = np.asarray(mask, bool)
        if (not mask.any() or len(_components(mask)) != 1
                or not mask[0].any() or not mask[-1].any()
                or not mask[:, 0].any() or not mask[:, -1].any()):
            return
        key = mask.shape, mask.tobytes()
        if key in seen:
            return
        ink, components = _printed_components(mask, tile)
        if not components or not all(c['clear'] for c in components):
            return
        seen.add(key)
        bank.append(dict(mask=mask, kind=kind, counts={c['holes'] for c in components},
                         ink=ink, components=components))
    for h in range(4, maximum+1):
        for w in range(4, maximum+1):
            add(np.ones((h, w), bool), 'rectangle')
            for side in range(4):
                # Open bays have varied depth, breadth and offset; no hidden marks.
                m = np.ones((h, w), bool)
                length, span = (h, w) if side < 2 else (w, h)
                breadth = int(rng.integers(1, min(3, span-2)+1))
                start = int(rng.integers(1, span-breadth))
                depth = int(rng.integers(max(1, length//3), length-1))
                if side == 0:
                    m[:depth, start:start+breadth] = False
                elif side == 1:
                    m[-depth:, start:start+breadth] = False
                elif side == 2:
                    m[start:start+breadth, :depth] = False
                else:
                    m[start:start+breadth, -depth:] = False
                add(m, 'bay')
            # Stair-step cut, a thick U, joined lobes, and an annular aperture.
            m = np.ones((h, w), bool)
            for r in range(min(h-1, 4)):
                m[r, :max(0, min(w-1, 4-r))] = False
            add(m, 'stair')
            m = np.ones((h, w), bool)
            thick = int(rng.integers(1, min(h, w)//2))
            m[:h-thick, thick:w-thick] = False
            add(m, 'U')
            m = np.ones((h, w), bool)
            mid, neck = h//2, int(rng.integers(1, w-1))
            m[mid, :] = False
            m[mid, neck] = True
            add(m, 'lobes')
            m = np.ones((h, w), bool)
            m[1:h-1, 1:w-1] = False
            if h >= 6 and w >= 6:
                m[1:3, 1:3] = True  # asymmetric thick corner
            add(m, 'annulus')
    return bank


def _choose_scene(rng, bank, target_kind, avoid_class=None):
    choices = []
    by_shape = {}
    for b in bank:
        if b['kind'] not in ('rectangle', 'annulus') and 0 in b['counts']:
            by_shape.setdefault(b['mask'].shape, []).append(b)
    distractors = {k: [c for c in bank if any(v > 0 and v != k for v in c['counts'])]
                   for k in range(1, 7)}
    for a in bank:
        if target_kind == 'train' and a['kind'] != 'rectangle':
            continue
        if target_kind == 'grown' and (a['kind'] != 'rectangle' or max(a['mask'].shape) < 9):
            continue
        if target_kind == 'cut' and (a['kind'] in ('rectangle', 'annulus') or max(a['mask'].shape) < 9):
            continue
        if target_kind == 'annulus' and (a['kind'] != 'annulus' or max(a['mask'].shape) < 9):
            continue
        for k in sorted(a['counts']):
            if not 1 <= k <= 6 or k == avoid_class:
                continue
            bs = [b for b in by_shape.get(a['mask'].shape, ()) if b['counts'] != a['counts']]
            cs = distractors[k]
            if bs and cs:
                choices.append((a, k, bs, cs))
    if not choices:
        return None
    a, k, bs, cs = choices[int(rng.integers(len(choices)))]
    b = bs[int(rng.integers(len(bs)))]
    c = cs[int(rng.integers(len(cs)))]
    mapping_keys = [0, k]
    windows = [a, b, c]
    areas = {}
    for window in windows:
        for component in window['components']:
            if component['holes'] in mapping_keys:
                areas.setdefault(component['holes'], set()).add(len(component['cells']))
    if not any(len(v) > 1 for v in areas.values()):
        ds = [d for d in bank if any(component['holes'] in areas
                    and len(component['cells']) not in areas[component['holes']]
                    for component in d['components'])]
        if not ds:
            return None
        d = ds[int(rng.integers(len(ds)))]
        windows.append(d)
    elif rng.random() < .40:
        d = bank[int(rng.integers(len(bank)))]
        windows.append(d)
    # Choose this after the optional fourth aperture so it is truly unused.
    used = set().union(*(window['counts'] for window in windows))
    unused = [v for v in range(1, 7) if v != k and v not in used]
    if unused and rng.random() < .35:
        mapping_keys.append(int(rng.choice(unused)))
    return windows, mapping_keys, k


def _glyph(k, rng):
    # Cavities can be in a compact array or an elongated tally; class is visible.
    if k <= 3 and rng.random() < .5:
        m = np.ones((3, max(3, 2*k+1)), bool)
        m[1, 1:2*k:2] = False
    else:
        rows = 1 if k <= 2 else 2
        cols = max(1, (k+rows-1)//rows)
        m = np.ones((2*rows+1, max(3, 2*cols+1)), bool)
        slots = list(range(rows*cols))
        rng.shuffle(slots)
        for slot in slots[:k]:
            m[1+2*(slot//cols), 1+2*(slot%cols)] = False
    if max(m.shape) <= 5 and rng.random() < .5:
        m = m.T.copy()
    return m


def _pack(rng, masks, min_width, max_height=22):
    """Finite shelf fallback plus unconstrained free placement when it fits."""
    n = len(masks)
    arrangements = []
    for order in permutations(range(n)):
        for split in range(1, n):
            groups = (order[:split], order[split:])
            heights = [max(masks[i].shape[0] for i in g) for g in groups]
            widths = [sum(masks[i].shape[1] for i in g)+len(g)-1 for g in groups]
            if sum(heights)+1 <= max_height and max(widths) <= 30:
                arrangements.append((groups, heights, widths))
    if not arrangements:
        return None
    groups, heights, widths = arrangements[int(rng.integers(len(arrangements)))]
    H = int(rng.integers(sum(heights)+1, max_height+1))
    W = int(rng.integers(max(min_width, max(widths)), 31))
    # Free placements use bounding-box clearance, not a fixed slot lattice.
    for _ in range(10):
        free = np.ones((H, W), bool)
        positions = [None]*n
        good = True
        for i in rng.permutation(n):
            h, w = masks[i].shape
            summed = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            blocked = summed[h:, w:]-summed[:-h, w:]-summed[h:, :-w]+summed[:-h, :-w]
            choices = np.argwhere(blocked == 0)
            if not len(choices):
                good = False
                break
            edge = choices[(choices[:, 0] == 0) | (choices[:, 1] == 0)
                           | (choices[:, 0]+h == H) | (choices[:, 1]+w == W)]
            if len(edge) and rng.random() < .45:
                choices = edge
            r, c = [int(x) for x in choices[int(rng.integers(len(choices)))]]
            positions[i] = (r, c)
            free[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = False
        if good:
            return (H, W), positions
    positions = [None]*n
    row_gap = int(rng.integers(1, H-sum(heights)+1))
    r = int(rng.integers(H-sum(heights)-row_gap+1))
    for group, height, width in zip(groups, heights, widths):
        c = int(rng.integers(W-width+1))
        for i in group:
            h, w = masks[i].shape
            positions[i] = (r+int(rng.integers(height-h+1)), c)
            c += w+1
        r += height+row_gap
    return (H, W), positions


def _make_pair(rng, roles, side, target_kind, avoid_class=None, avoid_density=None):
    bg, query, sep = roles
    pool = (_TEACHING_TILES if target_kind == 'train' else
            _ANNULAR_TILES if target_kind == 'annulus' else _PRINT_TILES)
    if avoid_density is not None:
        pool = [tile for tile in pool if tile.sum() != avoid_density]
    for _ in range(48):
        tile = pool[int(rng.integers(len(pool)))].copy()
        maximum = 8 if target_kind == 'train' else 10
        bank = _window_bank(rng, tile, maximum)
        scene = _choose_scene(rng, bank, target_kind, avoid_class)
        if scene is None:
            continue
        windows, keys, positive_class = scene
        rng.shuffle(keys)
        glyphs = [_glyph(k, rng) for k in keys]
        ph = max(5, max(g.shape[0] for g in glyphs))+2
        # Disjoint literal roles, locally identified by frame/field/key geometry.
        colors = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])[:len(keys)]]
        items = [(np.ones((5, 5), bool), None)] + list(zip(glyphs, colors))
        rng.shuffle(items)
        min_width = sum(m.shape[1] for m, _ in items)+len(items)+1
        if min_width > 30:
            continue
        # Build in header coordinates, then rotate only the header layout.
        # The swatch itself is inserted in global coordinates after rotation.
        if side % 2 == 0:
            packed = _pack(rng, [a['mask'] for a in windows], min_width, 29-ph)
            if packed is None:
                continue
            (wh, ww), positions = packed
            header = np.full((ph, ww), bg, np.int8)
        else:
            packed = _pack(rng, [a['mask'].T for a in windows], min_width, 29-ph)
            if packed is None:
                continue
            (ww, wh), transposed_positions = packed
            positions = [(c, r) for r, c in transposed_positions]
            header = np.full((ph, wh), bg, np.int8)
        cursor = int(rng.integers(header.shape[1]-min_width+3))
        for mask, color in items:
            rr = int(rng.integers(ph-mask.shape[0]+1))
            cc = cursor
            if color is None:
                header[rr:rr+5, cc:cc+5] = sep
                header[rr+1:rr+4, cc+1:cc+4] = bg
            else:
                header[rr:rr+mask.shape[0], cc:cc+mask.shape[1]][mask] = color
            cursor += mask.shape[1]+1
        header = np.rot90(header, side).copy()
        # Find the framed square after the header rotation, preserving the
        # swatch's global orientation and the fixed global row/column convention.
        frame_cells = _components(header == sep)[0]
        _, (fr, fc) = _crop(frame_cells)
        header[fr+1:fr+4, fc+1:fc+4][tile] = query
        work = np.full((wh, ww), bg, np.int8)
        expected_work = work.copy()
        mapping = dict(zip(keys, colors))
        for a, (r, c) in zip(windows, positions):
            h, w = a['mask'].shape
            work[r:r+h, c:c+w][a['mask']] = query
            patch = expected_work[r:r+h, c:c+w]
            for component in a['components']:
                patch[tuple(np.asarray(component['cells']).T)] = mapping.get(component['holes'], bg)
        if side == 0:
            grid = np.vstack((header, np.full((1, ww), sep, np.int8), work))
            out = np.vstack((header, np.full((1, ww), sep, np.int8), expected_work))
        elif side == 2:
            grid = np.vstack((work, np.full((1, ww), sep, np.int8), header))
            out = np.vstack((expected_work, np.full((1, ww), sep, np.int8), header))
        elif side == 1:
            grid = np.hstack((header, np.full((wh, 1), sep, np.int8), work))
            out = np.hstack((header, np.full((wh, 1), sep, np.int8), expected_work))
        else:
            grid = np.hstack((work, np.full((wh, 1), sep, np.int8), header))
            out = np.hstack((expected_work, np.full((wh, 1), sep, np.int8), header))
        info = _parse(grid)
        if target_kind == 'train' and not _rank_witness(info):
            continue
        if target_kind == 'train' and np.array_equal(out, _counterfactual(info, 'global')):
            continue
        # Phase and handedness have actual observable consequences in teaching.
        if target_kind == 'train' and any(np.array_equal(out, _render(info, order, transpose))
                for order, transpose in (((0, 1, 2), False), ((1, 0, 2), False),
                                         ((1, 2, 0), True))):
            continue
        return dict(input=grid, output=out), positive_class, int(tile.sum())
    raise RuntimeError('bounded scene construction exhausted; no family exception used')


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_roles = tuple(int(x) for x in rng.permutation(10)[:3])
    stable_roles = bool(rng.random() < .65)
    base_side = int(rng.integers(4))
    stable_side = bool(rng.random() < .6)
    test_kind = str(rng.choice(('grown', 'cut', 'annulus'), p=(.25, .55, .20)))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_class = first_density = None
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            for _ in range(8):
                roles = base_roles if stable_roles else tuple(int(x) for x in rng.permutation(10)[:3])
                side = base_side if stable_side else int(rng.integers(4))
                kind = 'train' if split == 'train' else test_kind
                pair, k, density = _make_pair(rng, roles, side, kind,
                                              first_class if split == 'train' and i == 1 else None,
                                              first_density if split == 'train' and i == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 0:
                        first_class, first_density = k, density
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, np.int8)
    return a.shape, a.tobytes()


def _rank_witness(info):
    # A concrete wrong rule numbers key glyphs along the panel's long axis.
    _, _, ph, pw = info['panel_box']
    axis = 1 if pw > ph else 0
    entries = sorted(info['entries'], key=lambda e: e[0][axis])
    rank = {k: i for i, (_, k, _) in enumerate(entries)}
    return any(c['holes'] in rank and rank[c['holes']] != c['holes']
               for a in info['apertures'] for c in a['components'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, taught = set(), set()
    seam = cut = erased = area_witness = False
    train_max, test_max = 0, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(out, _render(info)), 'wrong seam topology, print, color or preservation'
            wr, wc, wh, ww = info['work_box']
            body, answer = inp[wr:wr+wh, wc:wc+ww], out[wr:wr+wh, wc:wc+ww]
            assert np.all(answer[body == info['bg']] == info['bg'])
            assert not np.any(answer == info['query'])
            assert not np.array_equal(inp, out)
            local_max = max(max(a['mask'].shape) for a in info['apertures'])
            if split == 'test':
                test_max.append(local_max)
                continue
            train_max = max(train_max, local_max)
            assert _rank_witness(info), 'legend rank must be distinguished from its topology'
            assert not np.array_equal(out, _counterfactual(info, 'global')), 'missing local-origin witness'
            by_shape, by_class = {}, {}
            for a in info['apertures']:
                counts = {c['holes'] for c in a['components']}
                by_shape.setdefault(a['mask'].shape, []).append(counts)
                for component in a['components']:
                    k = component['holes']
                    if k in info['mapping']:
                        taught.add(k)
                        by_class.setdefault(k, set()).add(len(component['cells']))
                        seam |= k > 0 and _hole_count(info['tile']) == 0
                    else:
                        erased |= k > 0
                    assert component['clear']
            cut |= any(len({tuple(sorted(c)) for c in counts}) > 1 for counts in by_shape.values())
            area_witness |= any(len(areas) > 1 for areas in by_class.values())
            # Every training pair distinguishes the three concrete orientation shortcuts.
            for order, transpose in (((0, 1, 2), False), ((1, 0, 2), False), ((1, 2, 0), True)):
                assert not np.array_equal(out, _render(info, order, transpose))
    assert 0 in taught and len(taught-{0}) >= 2, 'two positive topology classes must be demonstrated'
    assert seam and cut and erased, 'missing seam creation, clipping contrast or topology erasure evidence'
    assert area_witness, 'training must distinguish cavity count from ink area'
    assert all(n > train_max for n in test_max), 'missing visible extent shift'


def self_test():
    """Bounded semantic fixtures, counterfactuals, 42-seed diversity and counts."""
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    assert _hole_count(diagonal) == 1
    diagonal[0, 1] = False
    assert _hole_count(diagonal) == 0
    tile = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], bool)
    assert _hole_count(tile) == 0
    aperture = np.ones((6, 6), bool)
    ink, comps = _printed_components(aperture, tile)
    assert len(comps) == 1 and comps[0]['holes'] == 1
    opened = aperture.copy()
    opened[:4, 2] = False
    _, open_comps = _printed_components(opened, tile)
    assert all(c['holes'] == 0 for c in open_comps)
    assert np.array_equal(ink, np.array([[int(x) for x in row] for row in
                          ('001001', '001001', '111111', '001001', '001001', '111111')], bool))
    # Hand-authored complete input and exact label: one stitched loop versus
    # the same 6x6 aperture with an open bay. Neither fixture is generated.
    fixture = np.zeros((16, 18), np.int8)
    fixture[:5, :5] = 8
    fixture[1:4, 1:4] = np.where(tile, 5, 0)
    fixture[1:4, 7:10] = 2
    fixture[1:4, 13:16] = np.array([[3, 3, 3], [3, 0, 3], [3, 3, 3]])
    fixture[7] = 8
    fixture[9:15, 1:7] = 5
    fixture[9:15, 10:16][opened] = 5
    expected = fixture.copy()
    expected[8:] = 0
    expected[9:15, 1:7][ink] = 3
    expected[9:15, 10:16][_ink(opened, tile)] = 2
    assert np.array_equal(solve(fixture), expected)
    # Change the visible one-hole legend to an unlisted two-hole glyph.
    fixture[1:4, 13:18] = np.array([[3, 3, 3, 3, 3], [3, 0, 3, 0, 3], [3, 3, 3, 3, 3]])
    expected[1:4, 13:18] = fixture[1:4, 13:18]
    expected[9:15, 1:7] = 0
    assert np.array_equal(solve(fixture), expected)
    motifs, geometries, shapes, classes, palette_roles, outputs, populations = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed in (0, 17, 41):
            assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
        for split in p:
            for pair in p[split]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
                motifs.add(info['tile'].tobytes())
                shapes.add(pair['input'].shape)
                classes.add(tuple(sorted(info['mapping'])))
                palette_roles.add((info['bg'], info['query'], info['sep']))
                outputs.add(_key(pair['output']))
                populations.add(len(info['apertures']))
                for a in info['apertures']:
                    geometries.add(_key(a['mask']))
                # Omitting A (one uniform listed color for every ink component)
                # and omitting B (classifying input apertures) must each fail.
                if split == 'train':
                    no_a = _counterfactual(info, 'no_a')
                    no_b = _counterfactual(info, 'no_b')
                    assert not np.array_equal(no_a, pair['output'])
                    assert not np.array_equal(no_b, pair['output'])
    assert len(motifs) >= 35 and len(geometries) >= 120
    assert len(shapes) >= 35 and len(classes) >= 8 and len(palette_roles) >= 25
    assert len(outputs) == 210 and populations == {3, 4}
    # verify.py additionally exercises every legal explicit combination.
    for a, b in ((2, 1), (6, 3), (10, 4)):
        p = generate_puzzle(1000+10*a+b, a, b)
        validate_generated_puzzle(p)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    damaged = generate_puzzle(51, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')


def _counterfactual(info, mode):
    out = info['grid'].copy()
    wr, wc, _, _ = info['work_box']
    for a in info['apertures']:
        r, c = a['pos']
        patch = out[wr+r:wr+r+a['mask'].shape[0], wc+c:wc+c+a['mask'].shape[1]]
        patch[a['mask']] = info['bg']
        if mode == 'no_a':
            patch[a['ink']] = info['mapping'].get(0, next(iter(info['mapping'].values())))
        elif mode == 'no_b':
            patch[a['mask']] = info['mapping'].get(_hole_count(a['mask']), info['bg'])
        elif mode == 'global':
            shifted = np.roll(info['tile'], (-(wr+r) % 3, -(wc+c) % 3), axis=(0, 1))
            _, components = _printed_components(a['mask'], shifted)
            for component in components:
                patch[tuple(np.asarray(component['cells']).T)] = info['mapping'].get(component['holes'], info['bg'])
        else:
            raise ValueError('unknown counterfactual')
    return out


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


if __name__ == '__main__':
    self_test()
    print('self_test passed')
