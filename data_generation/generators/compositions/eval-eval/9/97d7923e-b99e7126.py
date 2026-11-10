"""Rank hidden bars, then complete only the selected self-similar copies.

Each exceptional tile contains a capped miniature bar. Its entire non-background
stroke is a stencil for its incomplete bar on the tile lattice. Unary keys rank
the FULL lengths, not the number or span of the tiles currently supplied.
"""
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][-1] + 1 == i:
            result[-1].append(i)
        else:
            result.append([i])
    return result


def _tile_axis(divider_flags):
    interiors = _runs(np.flatnonzero(~divider_flags))
    lines = _runs(np.flatnonzero(divider_flags))
    _require(len(interiors) >= 2 and len(lines) == len(interiors) - 1, "missing internal lattice")
    _require(interiors[0][0] == 0 and interiors[-1][-1] == len(divider_flags) - 1,
             "lattice has internal dividers and no outer frame")
    _require(len({len(v) for v in interiors}) == len({len(v) for v in lines}) == 1,
             "irregular lattice spacing")
    return [v[0] for v in interiors], len(interiors[0]), len(lines[0])


def _origins(mask, rows, cols, supplied):
    h, w = mask.shape
    return [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)
            if all(0 <= y - r < h and 0 <= x - c < w and mask[y - r, x - c]
                   for y, x in supplied)]


def _parse_canonical(a):
    """Read an upper unary header and horizontal miniature bars from pixels."""
    divider = int(a[0, 0])
    full_rows = np.flatnonzero(np.all(a == divider, axis=1))
    _require(len(full_rows) > 0, "missing header separator")
    kh = int(full_rows[0])
    _require(1 <= kh <= 3, "invalid unary header height")
    header = a[:kh]
    _require(np.all(header[:, (0, -1)] == divider), "header corners must be divider color")
    keys = {}
    for x in range(a.shape[1]):
        ys = np.flatnonzero(header[:, x] != divider)
        if not len(ys):
            continue
        cap = int(header[0, x])
        _require(np.array_equal(ys, np.arange(len(ys))) and np.all(header[ys, x] == cap),
                 "keys must be straight aligned unary runs")
        _require(cap not in keys, "one key per family")
        keys[cap] = len(ys)
    _require(1 <= len(keys) <= 3 and max(keys.values()) == kh, "invalid header families")
    body = a[kh + 1:]
    _require(body.size > 0, "missing tile field")
    ys, h, rt = _tile_axis(np.all(body == divider, axis=1))
    xs, w, ct = _tile_axis(np.all(body == divider, axis=0))
    _require(rt == ct and rt in (1, 2) and 2 <= h <= 4 and w in (4, 5), "invalid tile geometry")
    rows, cols = len(ys), len(xs)
    _require(rows >= h and cols >= w, "whole microscopic stencil window must fit")
    records = {}
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tile = body[y:y + h, x:x + w]
            _require(divider not in tile, "divider color inside a tile")
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    ordered = sorted(records.values(), key=lambda item: len(item[1]), reverse=True)
    ordinary, ordinary_positions = ordered[0]
    _require(len(ordinary_positions) > rows * cols // 2, "ordinary tile must be a strict majority")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _require(len(ordinary_colors) == 2 and not ordinary_colors.intersection(keys),
             "ordinary tile needs two non-control colors")
    bars = []
    for tile, supplied in ordered[1:]:
        controls = set(map(int, np.unique(tile))).intersection(keys)
        _require(len(controls) == 1, "glyph must have one matching control identity")
        cap = controls.pop()
        endpoints = np.argwhere(tile == cap)
        _require(len(endpoints) == 2 and endpoints[0, 0] == endpoints[1, 0], "two matching horizontal caps required")
        local_row = int(endpoints[0, 0])
        left, right = map(int, endpoints[:, 1])
        length = right - left + 1
        _require(3 <= length <= w, "bar needs caps and a nonempty payload")
        payload = int(tile[local_row, left + 1])
        _require(payload not in ordinary_colors | set(keys) | {divider}
                 and np.all(tile[local_row, left + 1:right] == payload), "uniform disjoint payload required")
        mask = np.zeros((h, w), dtype=bool)
        mask[local_row, left:right + 1] = True
        backgrounds = np.unique(tile[~mask])
        _require(len(backgrounds) == 1 and int(backgrounds[0]) in ordinary_colors,
                 "glyph background must be an ordinary color")
        _require(np.any(ordinary[~mask] != tile[~mask]), "whole-tile copying must affect background")
        origins = _origins(mask, rows, cols, supplied)
        _require(len(origins) == 1, "fragment must have exactly one full-window registration")
        origin = origins[0]
        active = {(origin[0] + int(y), origin[1] + int(x)) for y, x in np.argwhere(mask)}
        _require(set(supplied) < active, "every candidate must be incomplete")
        bars.append(dict(cap=cap, payload=payload, length=length, tile=tile, mask=mask,
                         origin=origin, supplied=set(supplied), active=active,
                         local_row=local_row, local_start=left))
    _require(len({int(b['tile'][~b['mask']][0]) for b in bars}) == 1,
             "glyphs must share a visible background role")
    families = {cap: [b for b in bars if b['cap'] == cap] for cap in keys}
    for cap, family in families.items():
        _require(2 <= len(family) <= 3 and len({b['length'] for b in family}) == len(family),
                 "families need two or three uniquely ranked competitors")
        _require(keys[cap] <= len(family), "requested ordinal does not exist")
    occupied = set()
    for b in bars:
        _require(not occupied.intersection(b['active']), "completed candidate strokes may not overlap")
        occupied.update(b['active'])
    return dict(keys=keys, divider=divider, header_height=kh, ordinary=ordinary,
                bars=bars, families=families, row_starts=ys, col_starts=xs,
                geometry=(h, w, rt, rows, cols))


def _chosen(meta, rule="true"):
    if rule == "all":
        return meta['bars']
    chosen = []
    for cap, family in meta['families'].items():
        rank = meta['keys'][cap]
        if rule == "visible_count":
            order = sorted(family, key=lambda b: (-len(b['supplied']), min(b['supplied'])))
        elif rule == "visible_span":
            order = sorted(family, key=lambda b: (-(max(x for _, x in b['supplied']) -
                                                  min(x for _, x in b['supplied']) + 1), min(b['supplied'])))
        elif rule in ("spatial_top", "spatial_bottom"):
            order = sorted(family, key=lambda b: min(b['supplied']), reverse=rule == "spatial_bottom")
        else:
            order = sorted(family, key=lambda b: b['length'], reverse=rule != "ascending")
        index = 0 if rule == "longest" else len(order) - 1 if rule == "shortest" else rank - 1
        chosen.append(order[index])
    return chosen


def _complete(a, meta, rule="true", stroke_only=False):
    result = a.copy()
    h, w, _, _, _ = meta['geometry']
    for bar in _chosen(meta, rule):
        for r, c in bar['active'] - bar['supplied']:
            y = meta['header_height'] + 1 + meta['row_starts'][r]
            x = meta['col_starts'][c]
            target = result[y:y + h, x:x + w]
            if stroke_only:
                target[bar['mask']] = bar['tile'][bar['mask']]
            else:
                target[:] = bar['tile']
    return result


def _decode(a):
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(a, -rotation)
        try:
            meta = _parse_canonical(canonical)
        except ValueError:
            continue
        candidates.append((rotation, canonical, meta))
    _require(len(candidates) == 1, "a unique header direction and miniature-bar axis are required")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the header, templates, registrations and length ranks from input only."""
    rotation, canonical, meta = _decode(_grid(input_grid))
    return np.ascontiguousarray(np.rot90(_complete(canonical, meta), rotation))


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _fragment_options(mask, rows, cols, origin, active, minimum):
    points = sorted(active)
    result = []
    for code in range(1, (1 << len(points)) - 1):
        fragment = {p for i, p in enumerate(points) if code & (1 << i)}
        if len(fragment) >= minimum and _origins(mask, rows, cols, fragment) == [origin]:
            result.append(fragment)
    return result


def _sample_fragment(rng, options):
    sizes = sorted({len(f) for f in options})
    size = int(rng.choice(sizes))
    candidates = [f for f in options if len(f) == size]
    return candidates[int(rng.integers(len(candidates)))].copy()


def _paint(geometry, divider, ordinary, keys, key_columns, bars, complete=False):
    h, w, t, rows, cols = geometry
    kh = max(keys.values())
    width = cols * w + (cols - 1) * t
    height = kh + 1 + rows * h + (rows - 1) * t
    a = np.full((height, width), divider, dtype=np.int8)
    for cap, x in key_columns.items():
        a[:keys[cap], x] = cap
    selected = {cap: sorted([b for b in bars if b['cap'] == cap], key=lambda b: -b['length'])[n - 1]
                for cap, n in keys.items()}
    tiles = {}
    for b in bars:
        points = b['active'] if complete and selected[b['cap']] is b else b['supplied']
        for point in points:
            tiles[point] = b['tile']
    for r in range(rows):
        for c in range(cols):
            y, x = kh + 1 + r * (h + t), c * (w + t)
            a[y:y + h, x:x + w] = tiles.get((r, c), ordinary)
    return a


def _make_pair(rng, cfg, target, teaching_rank=None):
    shift = cfg['shift']
    family_count = 3 if target and shift == 'families' else int(rng.integers(1, 3))
    if teaching_rank == 2:
        family_count = 2  # Even two-example puzzles demonstrate grouping by cap.
    # A five-column tile is needed for three distinct capped-bar lengths.
    forced_w = 5 if teaching_rank == 2 or (target and shift in ('ordinal', 'sparsity')) else None
    if teaching_rank == 1 and shift == 'sparsity':
        forced_w = 5
    if target and shift == 'registration':
        forced_w = 4
    w = forced_w or int(rng.choice((4, 5)))
    populations = [int(rng.integers(2, w - 1)) for _ in range(family_count)]
    if teaching_rank == 2 or (teaching_rank == 1 and w == 5) or (target and shift == 'ordinal'):
        populations[0] = 3
    ranks = [int(rng.integers(1, min(2, n) + 1)) for n in populations]
    if teaching_rank is not None:
        ranks[0] = teaching_rank
    if target and shift == 'ordinal':
        ranks[0] = 3
    kh = max(ranks)
    minimum = 2 if (not target and shift == 'sparsity') else 1
    geometry_options = []
    for h in range(2, 5):
        for t in (1, 2):
            max_cols = (30 + t) // (w + t)
            for cols in range(w, max_cols + 1):
                if not target and shift == 'registration' and cols != w:
                    continue
                if target and shift == 'registration' and cols <= w:
                    continue
                # All registration probes may need both endpoints supplied.
                min_clues = sum(populations) * (2 if cols > w else minimum)
                if teaching_rank is not None:
                    anchor_min = (7 if minimum == 2 and teaching_rank == 2 else
                                  6 if w == 5 else 3)
                    min_clues = max(min_clues, anchor_min + minimum * sum(populations[1:]))
                min_rows = max(h, sum(populations), (2 * min_clues + 1 + cols - 1) // cols)
                max_rows = (30 - kh - 1 + t) // (h + t)
                if min_rows <= max_rows:
                    geometry_options.append((h, w, t, cols, min_rows, max_rows))
    _require(geometry_options, "no feasible geometry")
    h, w, t, cols, low, high = geometry_options[int(rng.integers(len(geometry_options)))]
    rows = int(rng.integers(low, high + 1))
    geometry = (h, w, t, rows, cols)
    palette = cfg['palette']
    if cfg['stable_palette']:
        divider, base, accent = palette[:3]
        caps = palette[3:3 + family_count]
        payload_pool = palette[6:6 + int(rng.integers(2 if teaching_rank == 2 else 1, 4))]
    else:
        rest = [c for c in range(10) if c != palette[3]]
        rng.shuffle(rest)
        divider, base, accent = rest[:3]
        caps = [palette[3]] + rest[3:3 + family_count - 1]
        payload_pool = rest[3 + family_count - 1:]
        payload_pool = payload_pool[:int(rng.integers(2 if teaching_rank == 2 else 1,
                                                   min(3, len(payload_pool)) + 1))]
    ordinary = np.where(rng.integers(2, size=(h, w)), accent, base).astype(np.int8)
    # Every row has both ordinary colors: copying the non-stroke background matters.
    for r in range(h):
        p, q = map(int, rng.choice(w, 2, replace=False))
        ordinary[r, p], ordinary[r, q] = base, accent
    keys = dict(zip(caps, ranks))
    key_order = [int(i) for i in rng.permutation(family_count)]
    width = cols * w + (cols - 1) * t
    key_left = int(rng.integers(1, width - 2 * family_count + 1))
    key_columns = {caps[f]: key_left + 2 * i for i, f in enumerate(key_order)}
    bar_specs = []
    for f, n in enumerate(populations):
        lengths = [int(v) for v in rng.choice(np.arange(3, w + 1), n, replace=False)]
        bar_specs += [(f, length) for length in lengths]
    row_assignment = [int(r) for r in rng.choice(rows, len(bar_specs), replace=False)]
    if teaching_rank == 2:
        family_indices = [i for i, (f, _) in enumerate(bar_specs) if f == 0]
        middle_length_index = next(i for i in family_indices if bar_specs[i][1] == 4)
        # Spatial second-from-either-side is the middle row; choose an extreme row.
        extreme_index = min(family_indices, key=lambda i: row_assignment[i])
        row_assignment[middle_length_index], row_assignment[extreme_index] = (
            row_assignment[extreme_index], row_assignment[middle_length_index])
    bars, options_by_bar, protected = [], [], set()
    for i, ((f, length), r) in enumerate(zip(bar_specs, row_assignment)):
        local_row = int(rng.integers(max(0, r - (rows - h)), min(h - 1, r) + 1))
        start = int(rng.integers(w - length + 1))
        cap, payload = caps[f], int(rng.choice(payload_pool))
        if teaching_rank == 2:
            if f == 0:
                payload = payload_pool[i % len(payload_pool)]
            elif i == populations[0]:
                payload = bars[0]['payload']  # Shared across cap families.
        tile = np.full((h, w), base, dtype=np.int8)
        tile[local_row, start:start + length] = payload
        tile[local_row, start] = tile[local_row, start + length - 1] = cap
        mask = tile != base
        origin = (r - local_row, int(rng.integers(cols - w + 1)))
        active = {(r, origin[1] + start + j) for j in range(length)}
        options = _fragment_options(mask, rows, cols, origin, active, minimum)
        _require(options, "no unique strict fragment")
        supplied = _sample_fragment(rng, options)
        if target and shift == 'sparsity':
            supplied = _sample_fragment(rng, [v for v in options if len(v) == 1])
        if teaching_rank is not None and f == 0:
            # Controlled witnesses expose disagreement between visible and full length.
            points = sorted(active)
            if teaching_rank == 1:
                if length == w:
                    supplied = set(points[:minimum])
                elif length == 4:
                    supplied = set((points[0], points[1], points[-1]))
                else:
                    supplied = {points[0], points[-1]}
            elif length == 4:
                supplied = set(points[:minimum])
            elif length == 3:
                supplied = {points[0], points[-1]}
            else:
                supplied = {points[0], points[1], points[-1]}
            # Training registration examples use a full-width window. Else ensure
            # the specially chosen witness still uniquely determines the origin.
            if _origins(mask, rows, cols, supplied) != [origin]:
                supplied = _sample_fragment(rng, options)
            protected.add(i)
        bars.append(dict(cap=cap, payload=payload, length=length, tile=tile, mask=mask,
                         supplied=supplied, active=active, origin=origin))
        options_by_bar.append(options)
    quota = (rows * cols - 1) // 2
    while sum(len(b['supplied']) for b in bars) > quota:
        reducible = [i for i, b in enumerate(bars) if i not in protected
                     and min(map(len, options_by_bar[i])) < len(b['supplied'])]
        _require(reducible, "clue quota exhausted")
        i = int(rng.choice(reducible))
        smaller = [f for f in options_by_bar[i] if len(f) < len(bars[i]['supplied'])]
        bars[i]['supplied'] = _sample_fragment(rng, smaller)
    pair = dict(input=_paint(geometry, divider, ordinary, keys, key_columns, bars),
                output=_paint(geometry, divider, ordinary, keys, key_columns, bars, complete=True))
    if rng.integers(2):
        pair = {k: np.fliplr(a) for k, a in pair.items()}
    rotation = cfg['rotation']
    if cfg['mixed_axis']:
        rotation = (rotation + (teaching_rank - 1 if teaching_rank else int(rng.integers(2)))) % 4
    return {k: np.ascontiguousarray(np.rot90(a, rotation)) for k, a in pair.items()}


def _unique_pair(pair, seen, anchor):
    if _key(pair['input']) not in seen:
        return pair
    # Finite color-permutation fallback. At least four non-anchor colors occur,
    # so there are >3000 distinct versions, versus at most thirteen used inputs.
    rest = [c for c in range(10) if c != anchor]
    for perm in permutations(rest):
        mapping = np.arange(10, dtype=np.int8)
        mapping[rest] = perm
        candidate = {k: mapping[a] for k, a in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise ValueError("unique color variants exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    cfg = dict(palette=[int(v) for v in rng.permutation(10)],
               shift=str(rng.choice(('families', 'ordinal', 'registration', 'sparsity'))),
               stable_palette=bool(rng.random() < .3), rotation=int(rng.integers(4)),
               mixed_axis=bool(rng.random() < .35))
    teaching = [None] * ntrain
    indices = [int(i) for i in rng.choice(ntrain, 2, replace=False)]
    teaching[indices[0]], teaching[indices[1]] = 1, 2
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            rank = teaching[i] if split == 'train' else None
            # Witnesses need a full-width registration: it isolates the ordinal
            # lesson from the extra translation freedom reserved for some tests.
            local_cfg = cfg.copy()
            if rank is not None and cfg['shift'] != 'sparsity':
                local_cfg['shift'] = 'registration'
            pair = _make_pair(rng, local_cfg, split == 'test', rank)
            pair = _unique_pair(pair, seen, cfg['palette'][3])
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "output dimensions must be preserved")
    rotation, canonical, meta = _decode(inp)
    expected = _complete(canonical, meta)
    _require(np.array_equal(np.rot90(out, -rotation), expected), "incorrect selective whole-tile completion")
    _require(np.any(inp != out), "every example must change")
    return rotation, canonical, meta


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            sig = _key(_grid(pair['input']))
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            analysis = _analyze(pair)
            if split == 'train':
                train.append(analysis)
    histories = {}
    for _, _, meta in train:
        for cap, n in meta['keys'].items():
            histories.setdefault(cap, set()).add(n)
    _require(any({1, 2} <= v for v in histories.values()), "training needs a repeated-control ordinal contrast")
    _require(any(meta['keys'][cap] == 2 and len(fam) == 3
                 for _, _, meta in train for cap, fam in meta['families'].items()),
             "training needs an intermediate rank")
    _require(any(len(m['families']) >= 2
                 and any(len({b['payload'] for b in fam}) > 1 for fam in m['families'].values())
                 and any(b['cap'] != c['cap'] and b['payload'] == c['payload']
                         for b in m['bars'] for c in m['bars']) for _, _, m in train),
             "training must distinguish cap grouping from payload grouping")
    for rule in ('all', 'visible_count', 'visible_span', 'ascending', 'longest', 'shortest',
                 'spatial_top', 'spatial_bottom'):
        _require(any(not np.array_equal(_complete(a, m, rule), _complete(a, m)) for _, a, m in train),
                 f"training fails to distinguish {rule}")
    _require(any(not np.array_equal(_complete(a, m, stroke_only=True), _complete(a, m))
                 for _, a, m in train), "whole-background copying needs a training witness")


def measure_pair(pair):
    rotation, _, meta = _analyze(pair)
    bars = meta['bars']
    return dict(shape=list(np.asarray(pair['input']).shape), rotation=rotation,
                geometry=list(meta['geometry']), families=len(meta['keys']),
                keys={str(c): n for c, n in meta['keys'].items()},
                populations=[len(f) for f in meta['families'].values()],
                lengths=[b['length'] for b in bars],
                supplied_counts=[len(b['supplied']) for b in bars],
                missing_counts=[len(b['active'] - b['supplied']) for b in bars],
                local_strokes=[[b['local_row'], b['local_start'], b['length']] for b in bars],
                origins=[list(b['origin']) for b in bars],
                macro_rows=[min(b['active'])[0] for b in bars],
                payload_classes=len({b['payload'] for b in bars}),
                ordinary_pattern=''.join('1' if v == meta['ordinary'][0, 0] else '0'
                                         for v in meta['ordinary'].flat),
                palette=list(map(int, np.unique(pair['input']))))


def self_test() -> dict:
    """Bounded 42-seed semantic, variation, metamorphic and contract checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [measure_pair(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    _require(len({tuple(r['shape']) for r in records}) >= 15, "canvas variation")
    _require(len({tuple(r['geometry']) for r in records}) >= 25, "lattice variation")
    _require({1, 2, 3} <= {r['families'] for r in records}, "family-count variation")
    _require({2, 3} <= {n for r in records for n in r['populations']}, "competitor-count variation")
    _require({1, 2, 3} <= {n for r in records for n in r['keys'].values()}, "ordinal coverage")
    _require({2, 3, 4} <= {r['geometry'][0] for r in records}, "tile heights")
    _require({4, 5} <= {r['geometry'][1] for r in records}, "tile widths")
    _require({1, 2} <= {r['geometry'][2] for r in records}, "divider thicknesses")
    _require({1, 2, 3, 4} <= {n for r in records for n in r['supplied_counts']}, "clue-count coverage")
    _require(len({r['ordinary_pattern'] for r in records}) >= 35, "ordinary-motif variation")
    _require(len({tuple(r['palette']) for r in records}) >= 20, "palette variation")
    _require({0, 1, 2, 3} <= {r['rotation'] for r in records}, "orientation coverage")
    _require(any(max(r['shape']) == 30 for r in records), "pixel-budget boundary")
    _require(any(r['geometry'][4] > r['geometry'][1] for r in records), "translation freedom")
    _require(len({tuple(v) for r in records for v in r['local_strokes']}) >= 15,
             "miniature geometry variation")
    _require(len({tuple(v) for r in records for v in r['origins']}) >= 15, "origin variation")
    signatures = {tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]) for puzzle in puzzles}
    _require(len(signatures) == 42, "seed diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(137 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    again = generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in again for p, q in zip(again[s], puzzles[0][s])
                 for k in ('input', 'output')), "seed reproduction")
    pair = puzzles[0]['test'][0]
    original_input = pair['input'].copy()
    mapping = np.array([7, 0, 9, 2, 5, 3, 8, 1, 6, 4], dtype=np.int8)
    for k in range(4):
        for reflect in (False, True):
            inp = np.rot90(np.fliplr(original_input) if reflect else original_input, k)
            out = np.rot90(np.fliplr(pair['output']) if reflect else pair['output'], k)
            _require(np.array_equal(solve(mapping[inp]), mapping[out]), "geometry/palette equivariance")
    _require(np.array_equal(original_input, pair['input']), "solve mutated input")
    # Fixed-geometry probe with TWO candidates on one coarse row. This is not a
    # sampling template: it exercises a compatible case outside the row allocator.
    geometry = (2, 4, 1, 4, 6)
    ordinary = np.array([[5, 7, 5, 7], [7, 5, 7, 5]], dtype=np.int8)
    probe_bars = []
    for cap, length, local_row, start, origin in (
            (2, 3, 0, 0, (1, 0)), (6, 3, 0, 1, (1, 2)),
            (2, 4, 0, 0, (2, 1)), (6, 4, 1, 0, (2, 1))):
        tile = np.full((2, 4), 5, dtype=np.int8)
        tile[local_row, start:start + length] = 9
        tile[local_row, start] = tile[local_row, start + length - 1] = cap
        active = {(origin[0] + local_row, origin[1] + start + j) for j in range(length)}
        points = sorted(active)
        probe_bars.append(dict(cap=cap, length=length, tile=tile, mask=tile != 5,
                               origin=origin, active=active, supplied={points[0], points[-1]}))
    probe_keys, probe_columns = {2: 2, 6: 2}, {2: 1, 6: 3}
    probe_input = _paint(geometry, 0, ordinary, probe_keys, probe_columns, probe_bars)
    probe_output = _paint(geometry, 0, ordinary, probe_keys, probe_columns, probe_bars, True)
    _require(np.array_equal(solve(probe_input.tolist()), probe_output), "disjoint same-row candidates")
    rekeyed = dict(probe_keys)
    rekeyed[2] = 1
    rekey_input = _paint(geometry, 0, ordinary, rekeyed, probe_columns, probe_bars)
    rekey_output = _paint(geometry, 0, ordinary, rekeyed, probe_columns, probe_bars, True)
    _require(np.array_equal(solve(rekey_input), rekey_output)
             and not np.array_equal(rekey_output, probe_output), "visible re-keying must change the answer")
    ambiguous_bars = [dict(b) for b in probe_bars]
    ambiguous_bars[2]['supplied'] = {(2, 2)}
    ambiguous_input = _paint(geometry, 0, ordinary, probe_keys, probe_columns, ambiguous_bars)
    tied_bars = [dict(b) for b in probe_bars]
    tied_tile = np.array([[2, 9, 2, 5], [5, 5, 5, 5]], dtype=np.int8)
    tied_bars[2].update(tile=tied_tile, length=3, mask=tied_tile != 5,
                        active={(2, 1), (2, 2), (2, 3)}, supplied={(2, 1), (2, 3)})
    tied_input = _paint(geometry, 0, ordinary, probe_keys, probe_columns, tied_bars)
    for invalid in (ambiguous_input, tied_input):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous registration or tied family accepted")
    mask = np.zeros((2, 5), dtype=bool)
    mask[0] = True
    active = {(1, x) for x in range(5)}
    options = _fragment_options(mask, 4, 5, (1, 0), active, 1)
    fragments = {_key(np.array(sorted(_sample_fragment(np.random.default_rng(s), options))))
                 for s in range(42)}
    _require(len(fragments) >= 12 and {1, 2, 3, 4} <= {key[0][0] for key in fragments},
             "conditional fragment diversity at fixed geometry and origin")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    r, c = np.argwhere(bad['test'][0]['input'] != bad['test'][0]['output'])[0]
    bad['test'][0]['output'][r, c] = bad['test'][0]['input'][r, c]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged completion accepted")
    return dict(seeds=42, pairs=len(records), canvas_shapes=len({tuple(r['shape']) for r in records}),
                geometries=len({tuple(r['geometry']) for r in records}),
                ordinary_motifs=len({r['ordinary_pattern'] for r in records}),
                palettes=len({tuple(r['palette']) for r in records}),
                conditional_fragments=len(fragments), independent_solver=False)
