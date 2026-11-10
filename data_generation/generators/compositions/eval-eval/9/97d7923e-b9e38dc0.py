"""Unary selection by illuminated receiver length; the spotlight stays implicit.

Only standard-library Python and NumPy are used. No parent code/data are imported.
Construction uses a union of shadow masks. solve independently scans opaque lanes
and reconstructs every role and ordinal from the input.
"""
from __future__ import annotations

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or int(value) < low or
            (high is not None and int(value) > high)):
        raise ValueError(f"invalid {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.tobytes()


def _runs(indices):
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def _fit_flank(values, sign):
    """Fit the longest periodic terminal suffix, requiring two outward steps."""
    ys = np.asarray(values, dtype=int)
    xs = np.arange(len(ys))
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            pred = ys[-1] + sign * ((xs - xs[-1] + phase) // period)
            unequal = np.flatnonzero(pred != ys)
            start = int(unequal[-1]) + 1 if len(unequal) else 0
            if sign * (int(ys[-1]) - int(ys[start])) >= 2:
                candidates.append((len(ys) - start, period, phase))
    _require(bool(candidates), "insufficient terminal boundary evidence")
    longest = max(c[0] for c in candidates)
    best = [c for c in candidates if c[0] == longest]
    # No arbitrary ambiguity resolution: maximal fits must have the same future.
    futures = {tuple((t + phase) // period for t in range(1, 31))
               for _, period, phase in best}
    _require(len(futures) == 1, "ambiguous boundary cadence")
    _, period, phase = min(best, key=lambda c: c[1:])
    return period, phase, longest


def _parse_canonical(a, bg, wall):
    """Read a right-facing scene. Invalid role/orientation hypotheses raise."""
    h, w = a.shape
    points = np.argwhere(a == wall)
    _require(len(points) >= 11, "not an emitter wall")
    rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
    _require(mouth - rear >= 4 and mouth < w - 3, "emitter axial span")
    cap = np.flatnonzero(a[:, rear] == wall)
    _require(len(cap) >= 3 and np.all(np.diff(cap) == 1), "closed rear")
    upper, lower = [], []
    for x in range(rear + 1, mouth + 1):
        rows = np.flatnonzero(a[:, x] == wall)
        _require(len(rows) == 2 and rows[1] - rows[0] >= 2, "thin paired rails")
        upper.append(int(rows[0]))
        lower.append(int(rows[1]))
    _require(abs(upper[0] - int(cap[0])) <= 1 and
             abs(lower[0] - int(cap[-1])) <= 1, "rear connection")
    _require(all(abs(b - c) <= 1 for rail in (upper, lower)
                 for b, c in zip(rail, rail[1:])), "rail continuity")
    fits = (_fit_flank(upper, -1), _fit_flank(lower, 1))
    footprint = np.zeros(a.shape, dtype=bool)
    bounds = []
    for x in range(rear + 1, w):
        if x <= mouth:
            lo, hi = upper[x - rear - 1], lower[x - rear - 1]
        else:
            lo = upper[-1] - (x - mouth + fits[0][1]) // fits[0][0]
            hi = lower[-1] + (x - mouth + fits[1][1]) // fits[1][0]
        bounds.append((lo, hi))
        footprint[max(0, lo + 1):min(h, hi), x] = True
    chamber = footprint.copy()
    chamber[:, mouth + 1:] = False

    # Two equal endpoints surrounding a nonbackground uniform interior identify
    # receivers. Column segmentation handles even immediately adjacent bars.
    bars = []
    for x in range(mouth + 1, w):
        for run in _runs(np.flatnonzero(a[:, x] != bg)):
            y0, y1 = int(run[0]), int(run[-1])
            if y1 - y0 < 2:
                continue
            values = a[y0:y1 + 1, x]
            if values[0] == values[-1] and values[0] != values[1] and np.all(values[1:-1] == values[1]):
                bars.append(dict(x=x, top=y0, bottom=y1, cap=int(values[0]),
                                 body=int(values[1]), raw=y1 - y0 - 1))
    # Inscriptions precede opaque marks. Their first axial section therefore
    # supplies a keyed cap identity; its receivers identify the neutral body.
    # An unkeyed multicolor obstacle can resemble cap/body/cap and must stay opaque.
    inscriptions = np.argwhere(chamber & (a != bg) & (a != wall))
    _require(len(inscriptions) > 0, "missing chamber inscriptions")
    first_x = int(inscriptions[:, 1].min())
    lead_colors = {int(a[y, x]) for y, x in inscriptions if x == first_x}
    _require(len(lead_colors) == 1, "ambiguous first inscription")
    lead = lead_colors.pop()
    bodies = {b['body'] for b in bars if b['cap'] == lead}
    _require(len(bodies) == 1, "keyed receivers identify one body color")
    neutral = bodies.pop()
    bars = [b for b in bars if b['body'] == neutral]
    _require(len(bars) >= 2 and len({b['x'] for b in bars}) == len(bars), "receiver rack")
    caps = sorted({b['cap'] for b in bars})
    _require(1 <= len(caps) <= 3, "receiver color roles")
    _require(len({bg, wall, neutral, *caps}) == 3 + len(caps), "disjoint role colors")
    receiver_mask = np.zeros_like(footprint)
    interior_mask = np.zeros_like(footprint)
    for b in bars:
        receiver_mask[b['top']:b['bottom'] + 1, b['x']] = True
        interior_mask[b['top'] + 1:b['bottom'], b['x']] = True
    _require(np.array_equal(a == neutral, interior_mask), "stray receiver body color")
    ranks, key_mask, keys = {}, np.zeros_like(footprint), []
    for color in caps:
        places = np.argwhere(chamber & (a == color))
        _require(1 <= len(places) <= 3, "missing unary inscription")
        _require(len(set(places[:, 1])) == 1 and np.all(np.diff(places[:, 0]) == 1), "unary run")
        ranks[color] = len(places)
        keys.append((int(places[0, 0]), int(places[0, 1]), len(places)))
        key_mask[places[:, 0], places[:, 1]] = True
        _require(np.all(((a == color) & ~receiver_mask) == (key_mask & (a == color))), "stray cap color")
    _require((len({k[0] for k in keys}) == 1 or len({k[0] + k[2] for k in keys}) == 1)
             and len({k[1] for k in keys}) == len(keys), "aligned compact inscriptions")
    _require(max(k[1] for k in keys) - min(k[1] for k in keys) <= 2 * len(keys), "scattered key")
    extras = (a != bg) & (a != wall) & (a != neutral) & ~np.isin(a, caps)
    _require(not np.any(extras[:, :max(k[1] for k in keys) + 1]), "obstacles precede inscriptions")
    _require(not np.any(extras[:, sorted({b['x'] for b in bars})]), "obstacle overlaps receiver column")
    families = {color: [b for b in bars if b['cap'] == color] for color in caps}
    _require(all(2 <= len(bs) <= 5 and ranks[c] <= len(bs) for c, bs in families.items()), "population/rank")
    return dict(a=a, bg=bg, wall=wall, neutral=neutral, caps=caps, ranks=ranks,
                bars=bars, families=families, rear=rear, mouth=mouth, fits=fits,
                footprint=footprint, chamber=chamber, extras=extras, bounds=bounds)


def _decode(inp):
    a = _grid(inp)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    _require(frequencies[bg] > a.size // 2, "background must dominate input")
    matches = []
    for rotation in range(4):
        canonical = np.rot90(a, -rotation)
        for wall in np.unique(canonical):
            if wall == bg or np.count_nonzero(canonical == wall) < 11:
                continue
            try:
                analysis = _parse_canonical(canonical, bg, int(wall))
            except AssertionError:
                continue
            analysis['rotation'] = rotation
            matches.append(analysis)
    _require(len(matches) == 1, "scene must have one input-visible interpretation")
    return matches[0]


def _measure(analysis, lane_scan=True, ignore_shadows=False):
    a, footprint = analysis['a'], analysis['footprint']
    light = footprint.copy()
    if not ignore_shadows:
        if lane_scan:
            for y in range(a.shape[0]):
                blocked = False
                for x in range(analysis['rear'] + 1, a.shape[1]):
                    if not footprint[y, x]:
                        continue
                    if analysis['extras'][y, x]:
                        blocked = True
                    if blocked:
                        light[y, x] = False
        else:
            for y, x in np.argwhere(analysis['extras'] & footprint):
                light[y, x:] = False
    records = []
    for bar in analysis['bars']:
        b = dict(bar)
        ys = np.arange(b['top'] + 1, b['bottom'])
        b['lit_rows'] = ys[light[ys, b['x']]]
        b['score'] = len(b['lit_rows'])
        b['cone_score'] = int(footprint[ys, b['x']].sum())
        records.append(b)
    return light, records


def _choose(analysis, records, property_name='score'):
    selected = []
    for color in analysis['caps']:
        group = [b for b in records if b['cap'] == color]
        if property_name == 'score':
            _require(len({b['score'] for b in group}) == len(group), "tied visible lengths")
        ordered = sorted(group, key=lambda b: (-b[property_name], b['x']))
        b = ordered[analysis['ranks'][color] - 1]
        if property_name == 'score':
            _require(b['score'] > 0, "selected receiver must have visible cells")
        selected.append(b)
    return selected


def _answer(analysis, lane_scan=True, ignore_shadows=False, selection='score', all_bars=False, whole=False):
    _, records = _measure(analysis, lane_scan, ignore_shadows)
    chosen = records if all_bars else _choose(analysis, records, selection)
    out = analysis['a'].copy()
    for b in chosen:
        rows = np.arange(b['top'] + 1, b['bottom']) if whole else b['lit_rows']
        out[rows, b['x']] = b['cap']
    return np.ascontiguousarray(np.rot90(out, analysis['rotation']))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels alone: no RNG, metadata, training, or lookup."""
    return _answer(_decode(input_grid))


def _features(analysis):
    _, records = _measure(analysis)
    chosen = _choose(analysis, records)
    raw_choices = _choose(analysis, records, 'raw')
    cone_choices = _choose(analysis, records, 'cone_score')
    for b in records:
        b['fragment'] = max((len(run) for run in _runs(b['lit_rows'])), default=0)
    fragment_choices = _choose(analysis, records, 'fragment')
    raw_diff = {b['x'] for b in chosen} != {b['x'] for b in raw_choices}
    cone_diff = {b['x'] for b in chosen} != {b['x'] for b in cone_choices}
    fragment_diff = {b['x'] for b in chosen} != {b['x'] for b in fragment_choices}
    split = any(len(_runs(b['lit_rows'])) >= 2 for b in chosen)
    partial = any(b['score'] < b['raw'] for b in chosen)
    geometry_partial = any(b['cone_score'] < b['raw'] for b in chosen)
    shadow_partial = any(b['score'] < b['cone_score'] for b in chosen)
    spatial = {}
    for reverse in (False, True):
        columns = []
        for color in analysis['caps']:
            bs = sorted([b for b in records if b['cap'] == color], key=lambda b: b['x'], reverse=reverse)
            columns.append(bs[analysis['ranks'][color] - 1]['x'])
        spatial[reverse] = set(columns) != {b['x'] for b in chosen}
    valid_raw_witness = any(len({b['raw'] for b in records if b['cap'] == c}) == len(analysis['families'][c])
                            and next(b['x'] for b in chosen if b['cap'] == c) !=
                            next(b['x'] for b in raw_choices if b['cap'] == c) for c in analysis['caps'])
    valid_cone_witness = any(len({b['cone_score'] for b in records if b['cap'] == c}) == len(analysis['families'][c])
                             and next(b['x'] for b in chosen if b['cap'] == c) !=
                             next(b['x'] for b in cone_choices if b['cap'] == c) for c in analysis['caps'])
    valid_fragment_witness = any(len({b['fragment'] for b in records if b['cap'] == c}) == len(analysis['families'][c])
                                 and next(b['x'] for b in chosen if b['cap'] == c) !=
                                 next(b['x'] for b in fragment_choices if b['cap'] == c) for c in analysis['caps'])
    points = np.argwhere(analysis['a'] == analysis['wall'])
    crop = (analysis['a'] == analysis['wall'])[points[:, 0].min():points[:, 0].max()+1,
                                              analysis['rear']:analysis['mouth']+1]
    extras = analysis['extras']
    grouped = bool(np.any(extras[1:] & extras[:-1]) or np.any(extras[:, 1:] & extras[:, :-1]))
    sequence = [b['cap'] for b in records]
    family_runs = [c for i, c in enumerate(sequence) if i == 0 or sequence[i-1] != c]
    interleaved = len(family_runs) > len(analysis['caps'])
    return dict(records=records, chosen=chosen, raw_diff=raw_diff, cone_diff=cone_diff,
                fragment_diff=fragment_diff, fragment_witness=valid_fragment_witness, grouped=grouped,
                interleaved=interleaved,
                raw_witness=valid_raw_witness, cone_witness=valid_cone_witness,
                split=split, partial=partial, geometry_partial=geometry_partial,
                shadow_partial=shadow_partial, spatial=spatial,
                outline=(crop.shape, crop.tobytes()),
                blocker_count=int((analysis['extras'] & analysis['footprint']).sum()))


def _geometry(rng, nfamilies, nreceivers):
    periods = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    if rng.random() < 0.5:
        periods[1] = periods[0]
    phases = [int(rng.integers(p)) for p in periods]
    rear_length = int(rng.integers(max(3, nfamilies + 1), max(3, nfamilies + 1) + 3))
    profiles = []
    for _ in range(2):
        vals = [int(rng.integers(2, 5))]
        for j in range(1, rear_length):
            low = 2 if j <= nfamilies else 1
            vals.append(int(np.clip(vals[-1] + rng.integers(-1, 2), low, 4)))
        profiles.append(vals)
    if rng.random() < 0.2:
        profiles[1] = profiles[0].copy()
        periods[1], phases[1] = periods[0], phases[0]
    flare = 2 * max(periods) + 1
    rails = [v + [v[-1] + (t + phase) // period for t in range(1, flare + 1)]
             for v, period, phase in zip(profiles, periods, phases)]
    length = len(rails[0])
    rear_max = min(3, 30 - length - max(nreceivers + 3, 9) - 1)
    rear = int(rng.integers(max(0, rear_max) + 1))
    mouth = rear + length
    min_width = mouth + max(nreceivers + 3, 9) + 1
    _require(min_width <= 30, "axial budget")
    width = int(rng.integers(min_width, 31))
    span = max(rails[0]) + max(rails[1]) + 1
    height = int(rng.integers(max(12, span + 1), 31))
    center = int(rng.integers(max(rails[0]), height - max(rails[1])))
    wall = np.zeros((height, width), dtype=bool)
    wall[center-profiles[0][0]:center+profiles[1][0]+1, rear] = True
    footprint = np.zeros_like(wall)
    for j, (up, down) in enumerate(zip(*rails), 1):
        x = rear + j
        wall[center-up, x] = wall[center+down, x] = True
        footprint[center-up+1:center+down, x] = True
    for x in range(mouth + 1, width):
        t = x - rear - rear_length
        lo = center - profiles[0][-1] - (t + phases[0]) // periods[0]
        hi = center + profiles[1][-1] + (t + phases[1]) // periods[1]
        footprint[max(0, lo+1):min(height, hi), x] = True
    return wall, footprint, rear, mouth, center


def _place_obstacles(rng, a, footprint, wall, columns, colors, bg, rear, mouth, center, nfamilies, style):
    if style == 'clean':
        return
    h, w = a.shape
    forbidden = np.zeros(w, dtype=bool)
    forbidden[columns] = True

    def available_cells():
        usable = (a == bg) & ~wall & ~forbidden[None, :]
        if style == 'single':
            opaque = np.isin(a, colors)
            adjacent = np.zeros_like(opaque)
            adjacent[1:] |= opaque[:-1]
            adjacent[:-1] |= opaque[1:]
            adjacent[:, 1:] |= opaque[:, :-1]
            adjacent[:, :-1] |= opaque[:, 1:]
            usable &= ~adjacent
        return usable

    groups = int(rng.integers(2, 5))
    for i in range(groups):
        shape = ((0, 0),)
        if style == 'grouped':
            shapes = [((0, 0), (1, 0)), ((0, 0), (1, 0), (0, 1), (1, 1)),
                      ((0, 0), (1, 0), (1, 1)), ((0, 0),)]
            shape = shapes[int(rng.integers(len(shapes)))]
        # Test every anchor in arrays rather than repeatedly checking Python
        # tuples. Transposing before enumeration preserves the original x-then-y
        # candidate order, so the random choices and generated pixels are exact.
        usable = available_cells()
        anchors = np.ones_like(footprint)
        intersects = np.zeros_like(footprint)
        for dy, dx in shape:
            shifted = np.zeros_like(footprint)
            shifted[:h-dy, :w-dx] = usable[dy:, dx:]
            anchors &= shifted
            intersects[:h-dy, :w-dx] |= footprint[dy:, dx:]
        anchors &= intersects
        anchors[:max(0, center-4)] = False
        anchors[min(h, center+5):] = False
        anchors[:, :rear+nfamilies+2] = False
        anchors[:, w-1:] = False
        if i == 0:
            anchors[:, columns[0]:] = False
        if i == 1:
            anchors[:, :columns[0]+1] = False
            anchors[:, columns[-1]:] = False
        available = np.argwhere(anchors.T)
        if len(available):
            x, y = available[int(rng.integers(len(available)))]
            for dy, dx in shape:
                a[y+dy, x+dx] = colors[i % len(colors)]
    # Extras of the same colors can be outside the footprint; such cells do not
    # start a shadow, even when their lane enters the expanding beam later.
    for i in range(int(rng.integers(0, 4))):
        outside = available_cells() & ~footprint
        outside[:, :rear+nfamilies+2] = False
        choices = np.argwhere(outside)
        if len(choices):
            y, x = choices[int(rng.integers(len(choices)))]
            a[y, x] = colors[i % len(colors)]


def _sample_pair(rng, palette, rotation, kind, style, rank_shift=False, population_shift=False,
                 training_rank_limit=3):
    bg, wall_color, neutral = map(int, palette[:3])
    nfamilies = (int(rng.integers(1, 3)) if kind == 'geometry' else
                 int(rng.integers(2, 4)) if kind == 'shadow' else int(rng.integers(1, 4)))
    counts = [int(rng.integers(2, 5)) for _ in range(nfamilies)]
    if kind in ('geometry', 'shadow'):
        counts[0] = int(rng.integers(3, 5))
    if rank_shift:
        counts[0] = max(3, counts[0])
    if population_shift:
        counts[0] = 5
    while sum(counts) > 9:
        first_min = 5 if population_shift else 3 if rank_shift or kind != 'ordinary' else 2
        candidates = [i for i, n in enumerate(counts) if n > (first_min if i == 0 else 2)]
        counts[int(rng.choice(candidates))] -= 1
    ranks = [int(rng.integers(1, min(training_rank_limit, n) + 1)) for n in counts]
    if kind == 'geometry':
        ranks[0] = 1
    elif kind == 'shadow':
        ranks[0] = 2
    elif rank_shift:
        ranks[0] = 3
    caps = [int(c) for c in palette[3:3 + nfamilies]]
    wall, footprint, rear, mouth, center = _geometry(rng, nfamilies, sum(counts))
    a = np.full(wall.shape, bg, dtype=np.int8)
    a[wall] = wall_color
    for i, (color, rank) in enumerate(zip(caps, ranks)):
        a[center-1:center-1+rank, rear+1+i] = color
    columns = sorted(int(x) for x in rng.choice(np.arange(mouth+1, a.shape[1]), sum(counts), replace=False))
    _place_obstacles(rng, a, footprint, wall, columns, palette[6:8], bg,
                     rear, mouth, center, nfamilies, style)
    shadow = np.zeros_like(footprint)
    extra = ~np.isin(a, [bg, wall_color, *caps])
    for y, x in np.argwhere(extra & footprint):
        shadow[y, x:] = True
    light = footprint & ~shadow
    sequence = [f for f, n in enumerate(counts) for _ in range(n)]
    if rng.random() < 0.75:
        rng.shuffle(sequence)
    used_scores = [set() for _ in counts]
    h = a.shape[0]
    for x, family in zip(columns, sequence):
        candidates = []
        # Exact integer prefix counts avoid a NumPy reduction for every possible
        # receiver segment. Candidate order and RNG consumption stay unchanged.
        prefix = [0] + np.cumsum(light[:, x], dtype=int).tolist()
        for top in range(h - 2):
            for bottom in range(top + 2, min(h, top + 23)):
                score = prefix[bottom] - prefix[top+1]
                if score in used_scores[family]:
                    continue
                candidates.append((top, bottom, score))
        _require(bool(candidates), "receiver score budget")
        # Choose a visible score first so very long segments do not dominate.
        scores = sorted({v[2] for v in candidates})
        score = int(rng.choice(scores))
        choices = [v for v in candidates if v[2] == score]
        top, bottom, _ = choices[int(rng.integers(len(choices)))]
        a[top, x] = a[bottom, x] = caps[family]
        a[top+1:bottom, x] = neutral
        used_scores[family].add(score)
    _require(np.count_nonzero(a == bg) > a.size // 2, "construction density")
    inp = np.ascontiguousarray(np.rot90(a, rotation))
    analysis = _decode(inp)
    f = _features(analysis)
    if kind == 'geometry':
        _require(f['raw_witness'] and f['geometry_partial'] and all(f['spatial'].values()), "geometry teaching contrast")
    elif kind == 'shadow':
        _require(f['cone_witness'] and f['fragment_witness'] and f['shadow_partial'] and f['split']
                 and f['interleaved'], "shadow teaching contrast")
        _require(len(np.unique(analysis['a'][analysis['extras'] & analysis['footprint']])) >= 2, "two opaque colors")
    if style == 'grouped':
        _require(f['grouped'], "grouped obstacle shift")
    return {'input': inp, 'output': _answer(analysis, lane_scan=False)}


def _fallback_pair(palette, rotation, kind, style, rank_shift, population_shift, rank_limit, seen, anchor):
    """One checked proposal per finite regime, then guarantee uniqueness.

    These are proposal RNG seeds, not stored grids or solver lookup records.
    self_test checks every admitted fallback regime. Relabeling commutes with
    parsing/physics, so any requested palette and cardinal direction work.
    """
    proposal_seed = {'geometry': 19, 'shadow': 93, 'ordinary': 0}[kind]
    pair = _sample_pair(np.random.default_rng(proposal_seed), palette, rotation, kind, style,
                        rank_shift=rank_shift, population_shift=population_shift,
                        training_rank_limit=rank_limit)
    others = [c for c in range(10) if c != anchor]
    # All 18 dihedral relabelings of the nine non-anchor colors are distinct
    # on any two visible non-anchor roles. At most 13 inputs precede this one.
    for sign in (1, -1):
        for offset in range(9):
            mapping = np.arange(10, dtype=np.int8)
            for j, color in enumerate(others):
                mapping[color] = others[(sign * j + offset) % 9]
            candidate = {k: mapping[v] for k, v in pair.items()}
            if _signature(candidate['input']) not in seen:
                return candidate
    raise AssertionError('uniqueness proof violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base_palette = rng.permutation(10)
    shared_palette = bool(rng.random() < 0.25)
    anchor_cap = int(base_palette[3])
    preferred_rotation = int(rng.integers(4))
    regime = str(rng.choice(['population', 'grouped', 'rank', 'orientation']))
    shared_orientation = bool(rng.random() < 0.25)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = ('geometry' if i == 0 else 'shadow' if i == 1 else 'ordinary') if split == 'train' else 'ordinary'
            style = 'clean' if kind == 'geometry' else 'single' if kind == 'shadow' else str(rng.choice(['clean', 'single', 'grouped']))
            if regime == 'grouped':
                style = 'grouped' if split == 'test' else ('clean' if kind == 'geometry' else 'single')
            rotation = preferred_rotation if shared_orientation else int(rng.integers(4))
            if regime == 'orientation':
                rotation = (preferred_rotation + 1) % 4 if split == 'test' else preferred_rotation
            palette = base_palette.copy() if shared_palette else rng.permutation(10)
            # The same cap identity changes ordinal in the two teaching scenes.
            if kind != 'ordinary':
                anchor_index = int(np.flatnonzero(palette == anchor_cap)[0])
                palette[3], palette[anchor_index] = palette[anchor_index], palette[3]
            rank_shift = regime == 'rank' and split == 'test'
            population_shift = regime == 'population' and split == 'test'
            rank_limit = 2 if regime == 'rank' and split == 'train' else 3
            accepted = False
            for _ in range(400):
                try:
                    pair = _sample_pair(rng, palette, rotation, kind, style,
                                        rank_shift=rank_shift, population_shift=population_shift,
                                        training_rank_limit=rank_limit)
                except AssertionError:
                    continue
                sig = _signature(pair['input'])
                if sig in seen:
                    continue
                puzzle[split].append(pair)
                seen.add(sig)
                accepted = True
                break
            if not accepted:
                pair = _fallback_pair(palette, rotation, kind, style, rank_shift, population_shift,
                                      rank_limit, seen, anchor_cap)
                puzzle[split].append(pair)
                seen.add(_signature(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "list splits")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, teaching, cap_ranks = set(), [], {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape preservation")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            analysis = _decode(inp)
            _require(np.array_equal(_answer(analysis), out), "visibility/ordinal output mismatch")
            _require(np.array_equal(_answer(analysis, lane_scan=False), out), "shadow union/scan mismatch")
            canonical_out = np.rot90(out, -analysis['rotation'])
            changed = analysis['a'] != canonical_out
            _require(np.any(changed) and np.all(analysis['a'][changed] == analysis['neutral']), "only receiver interiors change")
            _require(np.all(np.isin(canonical_out[changed], analysis['caps'])), "fill with matching cap")
            _require(np.array_equal(inp[inp == analysis['bg']], out[inp == analysis['bg']]), "implicit field must not paint background")
            f = _features(analysis)
            if split == 'train':
                teaching.append((analysis, f))
                for cap, rank in analysis['ranks'].items():
                    cap_ranks.setdefault(cap, set()).add(rank)
    _require(any({1, 2} <= rs for rs in cap_ranks.values()), "same-cap ordinal contrast")
    _require(any(f['blocker_count'] == 0 and f['raw_witness'] and f['geometry_partial']
                 and all(f['spatial'].values()) for _, f in teaching), "clean geometry counterexample")
    _require(any(f['cone_witness'] and f['fragment_witness'] and f['shadow_partial'] and f['split'] and f['interleaved'] and
                 len(np.unique(a['a'][a['extras'] & a['footprint']])) >= 2 for a, f in teaching), "shadow-dependent ranking counterexample")
    _require(any(not np.array_equal(_answer(a, all_bars=True), _answer(a)) for a, _ in teaching), "omitting A must alter an answer")
    _require(any(not np.array_equal(_answer(a, selection='raw', whole=True), _answer(a)) for a, _ in teaching), "omitting B must alter an answer")


def self_test():
    """Bounded semantic checks and actual-pixel diversity over 42 fixed seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, outlines, layouts, periods, orientations, palettes = set(), set(), set(), set(), set(), set(), set()
    flags = {k: 0 for k in ('raw_diff', 'cone_diff', 'fragment_diff', 'split', 'geometry_partial', 'shadow_partial')}
    family_counts, populations, ranks, raw_ties = set(), set(), set(), 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _decode(pair['input'])
                f = _features(a)
                shapes.add(pair['input'].shape)
                outlines.add(f['outline'])
                layouts.add(tuple((b['x']-a['mouth'], b['top'], b['bottom'], b['score']) for b in f['records']))
                periods.add(tuple(fit[0] for fit in a['fits']))
                orientations.add(a['rotation'])
                palettes.add((a['bg'], a['wall'], a['neutral']))
                family_counts.add(len(a['caps']))
                ranks.update(a['ranks'].values())
                for cap, bs in a['families'].items():
                    populations.add(len(bs))
                    raw_ties += len({b['raw'] for b in bs}) < len(bs)
                for k in flags:
                    flags[k] += bool(f[k])
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve")
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(outlines) >= 100 and len(layouts) >= 150, "structural diversity")
    _require(orientations == {0, 1, 2, 3} and len(periods) >= 6 and len(palettes) >= 50, "geometry/palette diversity")
    _require(family_counts == {1, 2, 3} and {2, 3, 4, 5} <= populations and ranks == {1, 2, 3}, "rank and population diversity")
    _require(raw_ties >= 5 and all(v >= 25 for v in flags.values()), "interaction diversity")
    # Every fallback combination works, including minimum ordinal curricula.
    fallback_count = 0
    for limit in (2, 3):
        for kind, style in (('geometry', 'clean'), ('shadow', 'single'),
                            ('ordinary', 'clean'), ('ordinary', 'single'), ('ordinary', 'grouped')):
            pair = _fallback_pair(np.arange(10), 0, kind, style, False, False, limit, set(), 3)
            _require(np.array_equal(solve(pair['input']), pair['output']), "fallback semantics")
            fallback_count += 1
    for style in ('clean', 'single', 'grouped'):
        for rank_shift, population_shift in ((True, False), (False, True)):
            pair = _fallback_pair(np.arange(10), 0, 'ordinary', style, rank_shift, population_shift, 3, set(), 3)
            _require(np.array_equal(solve(pair['input']), pair['output']), "fallback query shift")
            fallback_count += 1
    seen_fallbacks = set()
    for _ in range(14):
        pair = _fallback_pair(np.arange(10), 0, 'shadow', 'single', False, False, 2, seen_fallbacks, 3)
        seen_fallbacks.add(_signature(pair['input']))
    _require(len(seen_fallbacks) == 14, "fallback uniqueness")
    fallback_puzzle, seen_fallbacks = {'train': [], 'test': []}, set()
    for i in range(14):
        split = 'train' if i < 10 else 'test'
        kind = 'geometry' if i == 0 else 'shadow' if i == 1 else 'ordinary'
        style = 'clean' if kind == 'geometry' else 'single' if kind == 'shadow' else ('clean', 'single', 'grouped')[i % 3]
        pair = _fallback_pair(np.arange(10), 0, kind, style, i >= 10, False,
                              2 if split == 'train' else 3, seen_fallbacks, 3)
        seen_fallbacks.add(_signature(pair['input']))
        fallback_puzzle[split].append(pair)
    validate_generated_puzzle(fallback_puzzle)
    # Hand-specified pixels and expected recolor cells, independent of the sampler.
    fixture = np.zeros((15, 15), dtype=np.int8)
    fixture[5:10, 0] = 1
    for x, y in enumerate((5, 5, 4, 4, 3, 3), 1):
        fixture[y, x] = fixture[14-y, x] = 1
    fixture[6:8, 1] = 2  # second visible-length rank
    for x, top, bottom in ((8, 1, 13), (11, 0, 14), (14, 4, 10)):
        fixture[top, x] = fixture[bottom, x] = 2
        fixture[top+1:bottom, x] = 3
    fixture[7, 7] = fixture[4, 10] = fixture[1, 7] = 4
    expected = fixture.copy()
    expected[[3, 4, 5, 6, 8, 9, 10, 11], 8] = 2
    _require(np.array_equal(solve(fixture), expected), "manual second-rank/thin-boundary regression")
    a = _decode(fixture)
    _, bs = _measure(a)
    _require([b['score'] for b in bs] == [8, 11, 4], "manual visibility counts")
    _require(4 in bs[0]['lit_rows'] and 4 not in bs[1]['lit_rows'], "occluders affect only downstream receivers")
    _require(1 in bs[1]['lit_rows'], "outside-beam obstacle is inert")
    _require(3 in bs[0]['lit_rows'] and 3 in bs[1]['lit_rows'], "receivers are transparent")
    _require(not np.array_equal(_answer(a, all_bars=True), expected), "manual omission of A")
    _require(not np.array_equal(_answer(a, selection='raw', whole=True), expected), "manual omission of B")
    # A capped-looking object in two opaque colors is not a keyed receiver.
    disguised_obstacle = fixture.copy()
    disguised_obstacle[1:4, 12] = [4, 5, 4]
    _require(np.array_equal(solve(disguised_obstacle), np.where(fixture == expected, disguised_obstacle, expected)),
             "foreign capped-looking object remains an obstacle")
    for seed in (0, 17, 41):
        repeated = generate_puzzle(seed)
        for s in ('train', 'test'):
            for p, q in zip(puzzles[seed][s], repeated[s]):
                _require(all(np.array_equal(p[k], q[k]) for k in p), "seed determinism")
    # Rotation, reflection, and arbitrary color-role renaming preserve the rule.
    source = next(p for p in puzzles[0]['train'] if len(set(_decode(p['input'])['ranks'].values())) > 1)
    mapping = np.array([6, 9, 4, 1, 7, 0, 8, 3, 5, 2], dtype=np.int8)
    for r in range(4):
        for flip in (False, True):
            inp = np.flipud(source['input']) if flip else source['input']
            out = np.flipud(source['output']) if flip else source['output']
            _require(np.array_equal(solve(mapping[np.rot90(inp, r)]), mapping[np.rot90(out, r)]), "geometric/color invariance")
    for n, m in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(123 + n + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    p = corrupted['test'][0]
    y, x = np.argwhere(p['input'] != p['output'])[0]
    p['output'][y, x] = p['input'][y, x]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {'seeds': 42, 'pairs': sum(len(p[s]) for p in puzzles for s in p),
            'shapes': len(shapes), 'outlines': len(outlines), 'receiver_layouts': len(layouts),
            'period_pairs': sorted(periods), 'palettes': len(palettes),
            'fallback_regimes_checked': fallback_count,
            'raw_tied_families': raw_ties, 'interactions': flags}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
