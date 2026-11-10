"""Ray-triggered, anchored unfolding of self-similar glyph obstacles.

Only NumPy and the standard library are used. solve sees the input alone.
Canonical coordinates point downward; the entire pixel scene may be rotated.
"""
from itertools import combinations
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid ARC colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _groups(indices):
    groups = []
    for i in map(int, indices):
        if groups and groups[-1][-1] + 1 == i:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _segment(a):
    divider = int(a[0, 0])
    rs = _groups(np.flatnonzero(np.all(a == divider, axis=1)))
    cs = _groups(np.flatnonzero(np.all(a == divider, axis=0)))
    _require(len(rs) >= 3 and len(cs) >= 3, "missing divider lattice")
    _require(rs[0][0] == cs[0][0] == 0 and rs[-1][-1] == a.shape[0]-1
             and cs[-1][-1] == a.shape[1]-1, "lattice needs a complete outer frame")
    thickness = {len(g) for g in rs + cs}
    _require(len(thickness) == 1, "divider thickness must be regular")
    t = thickness.pop()
    heights = {rs[i+1][0]-rs[i][-1]-1 for i in range(len(rs)-1)}
    widths = {cs[i+1][0]-cs[i][-1]-1 for i in range(len(cs)-1)}
    _require(len(heights) == len(widths) == 1, "unequal tile sizes")
    h, w = heights.pop(), widths.pop()
    _require(h in (2, 3) and w in (2, 3) and t in (1, 2), "unsupported tile geometry")
    records = {}
    for r in range(len(rs)-1):
        for c in range(len(cs)-1):
            y, x = rs[r][-1]+1, cs[c][-1]+1
            tile = a[y:y+h, x:x+w]
            _require(not np.any(tile == divider), "divider color inside a tile")
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    _require(len(records) == 3, "expected ordinary, guide, and glyph tiles")
    solid = [v for v in records.values() if len(np.unique(v[0])) == 1]
    _require(len(solid) == 1, "guide must be the only solid tile")
    guide_tile, guides = solid[0]
    other = [v for v in records.values() if v is not solid[0]]
    other.sort(key=lambda v: len(v[1]), reverse=True)
    ordinary, ordinary_positions = other[0]
    glyph, seeds = other[1]
    _require(len(ordinary_positions) > max(len(seeds), len(guides)), "ordinary tile must be strictly modal")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    _require(len(ordinary_colors) in (2, 3), "ordinary tile needs two or three colors")
    guide = int(guide_tile[0, 0])
    _require(guide not in ordinary_colors and guide not in glyph, "guide color has another role")
    extra = set(map(int, np.unique(glyph))) - ordinary_colors
    _require(len(extra) == 2, "glyph needs distinctive ink and anchor colors")
    singleton = [c for c in extra if np.count_nonzero(glyph == c) == 1]
    _require(len(singleton) == 1, "anchor must be the unique singleton novel color")
    anchor_color = singleton[0]
    ink = (extra - {anchor_color}).pop()
    mask = (glyph == ink) | (glyph == anchor_color)
    _require(3 <= int(mask.sum()) < h*w, "invalid ink stencil")
    _require(len(np.unique(glyph[~mask])) == 1 and int(glyph[~mask][0]) in ordinary_colors,
             "glyph background must be one ordinary color")
    _require(np.any(ordinary[~mask] != glyph[~mask]), "whole-tile copying needs a background witness")
    coarse = np.zeros((len(rs)-1, len(cs)-1), dtype=np.int8)
    for p in guides:
        coarse[p] = 1
    for p in seeds:
        coarse[p] = 2
    return dict(coarse=coarse, ordinary=ordinary, glyph=glyph, guide=guide, ink=ink,
                anchor_color=anchor_color, mask=mask, divider=divider, t=t,
                geometry=(h, w, t, *coarse.shape))


def _footprint(seed, mask, anchor):
    r, c = seed
    ar, ac = anchor
    return {(r+int(y)-ar, c+int(x)-ac) for y, x in np.argwhere(mask)}


def _stencil_key(mask, anchor):
    return tuple(sorted(_footprint((0, 0), mask, anchor)))


def _infer(input_grid):
    a = _grid(input_grid)
    local = _segment(a)
    candidates = []
    for k in range(4):
        coarse = np.rot90(local['coarse'], k).copy()
        glyph = np.rot90(local['glyph'], k).copy()
        mask = np.rot90(local['mask'], k).copy()
        anchor = tuple(map(int, np.argwhere(glyph == local['anchor_color'])[0]))
        entries = list(map(int, np.flatnonzero(coarse[0] == 1)))
        if anchor[0] != 0 or not entries or any(b-a < 2 for a, b in zip(entries, entries[1:])):
            continue
        expected_guide = np.zeros(coarse.shape, dtype=bool)
        valid = True
        lengths = []
        for c in entries:
            rows = np.flatnonzero(coarse[:, c] == 1)
            if not 2 <= len(rows) <= 3 or not np.array_equal(rows, np.arange(len(rows))):
                valid = False
                break
            lengths.append(len(rows))
            expected_guide[:len(rows), c] = True
        if not valid or not np.array_equal(coarse == 1, expected_guide):
            continue
        seeds = set(map(tuple, np.argwhere(coarse == 2)))
        occupied, footprints = set(), {}
        R, C = coarse.shape
        H, W = mask.shape
        for seed in sorted(seeds):
            r, c = seed
            # The window, not merely its ink, must fit. Seeds are downstream
            # of every input stub; anchor marks the upstream microscopic row.
            if r < max(lengths) or not (0 <= c-anchor[1] <= C-W and r+H <= R):
                valid = False
                break
            cells = _footprint(seed, mask, anchor)
            if occupied & cells or any(coarse[p] == 1 for p in cells):
                valid = False
                break
            occupied |= cells
            footprints[seed] = cells
        if valid and len(seeds) >= 2:
            info = dict(local)
            info.update(k=k, coarse=coarse, mask=mask, glyph=glyph,
                        ordinary=np.rot90(local['ordinary'], k).copy(), anchor=anchor,
                        seeds=seeds, footprints=footprints, entries=entries,
                        geometry=(*mask.shape, local['t'], R, C), stub_lengths=lengths)
            candidates.append(info)
    _require(len(candidates) == 1, "entry direction or anchored footprints are ambiguous")
    return candidates[0]


def _evolve(coarse, mask, anchor, *, unfold=True, activate_all=False, branch=True):
    """Synchronous coarse simulation. Growth occurs before collisions at a layer.

    Counterfactual options are used only to measure demonstration strength.
    State is derived from this input; copied glyphs are barriers, not new seeds.
    """
    R, C = coarse.shape
    seeds = set(map(tuple, np.argwhere(coarse == 2)))
    footprints = {p: _footprint(p, mask, anchor) for p in seeds}
    barriers = set(seeds)
    activated = set()
    if activate_all:
        activated = set(seeds)
        for p in seeds:
            barriers |= footprints[p] if unfold else {p}
    active = set(map(int, np.flatnonzero(coarse[0] == 1)))
    rays = set(map(tuple, np.argwhere(coarse == 1)))
    forward, events, activation_layers = set(), [], []
    simultaneous = multi_hit = clipped = shared = 0
    for r in range(1, R):
        forward.update((r, c) for c in active)
        new = {(r, c) for c in active if (r, c) in seeds} - activated
        if new:
            activation_layers.append((r, tuple(sorted(new))))
            activated |= new
            for p in new:
                barriers |= footprints[p] if unfold else {p}
        blocked = {c for y, c in barriers if y == r}
        hits, flowing = {}, set()
        for c in sorted(active):
            if c not in blocked:
                rays.add((r, c))
                flowing.add(c)
            else:
                left = right = c
                while left-1 in blocked:
                    left -= 1
                while right+1 in blocked:
                    right += 1
                hits.setdefault((left, right), []).append(c)
        simultaneous += len(hits) >= 2
        flanks = set()
        for (left, right), columns in sorted(hits.items()):
            events.append((r, left, right, tuple(columns)))
            multi_hit += len(columns) > 1
            clipped += left == 0 or right == C-1
            if not branch:
                continue
            for c in range(max(0, left-1), min(C, right+2)):
                if (r-1, c) not in barriers:
                    rays.add((r-1, c))
            for c in (left-1, right+1):
                if 0 <= c < C:
                    shared += c in flowing or c in flanks
                    flanks.add(c)
                    rays.add((r, c))
        active = flowing | flanks
    out = np.zeros(coarse.shape, dtype=np.int8)
    for p in rays - barriers:
        out[p] = 1
    for p in barriers:
        out[p] = 2
    stats = dict(activated=activated, dormant=seeds-activated, footprints=footprints,
                 forward=forward, events=events, activation_layers=activation_layers,
                 simultaneous=simultaneous, multi_hit=multi_hit, clipped=clipped,
                 merging=shared, expanded=barriers-seeds, rays=rays-barriers)
    return out, stats


def _paint(coarse, ordinary, glyph, guide, divider, thickness):
    H, W = ordinary.shape
    R, C = coarse.shape
    grid = np.full((R*(H+thickness)+thickness, C*(W+thickness)+thickness), divider, dtype=np.int8)
    for r in range(R):
        for c in range(C):
            y, x = thickness+r*(H+thickness), thickness+c*(W+thickness)
            tile = ordinary if coarse[r, c] == 0 else (glyph if coarse[r, c] == 2 else guide)
            grid[y:y+H, x:x+W] = tile
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _infer(input_grid)
    coarse, _ = _evolve(info['coarse'], info['mask'], info['anchor'])
    canonical = _paint(coarse, info['ordinary'], info['glyph'], info['guide'], info['divider'], info['t'])
    return np.rot90(canonical, -info['k']).copy()


def _mask(rng, shape):
    H, W = shape
    mask = np.zeros(shape, dtype=bool)
    ac = int(rng.integers(W))
    mask[0, ac] = True
    pool = [i for i in range(H*W) if i != ac]
    count = int(rng.integers(3, H*W))
    mask.flat[rng.choice(pool, count-1, replace=False)] = True
    return mask, (0, ac)


def _geometry(rng, mode):
    options = []
    for H, W, t in ((h, w, t) for h in (2, 3) for w in (2, 3) for t in (1, 2)):
        maxR, maxC = (30-t)//(H+t), (30-t)//(W+t)
        minR = max(6, H+4) if mode != 'parallel' else H+3
        minC = W+2 if mode != 'parallel' else 2*W+1
        if minR <= maxR and minC <= maxC:
            options.append((H, W, t, minR, maxR, minC, maxC))
    H, W, t, lowR, highR, lowC, highC = options[int(rng.integers(len(options)))]
    return H, W, t, int(rng.integers(lowR, highR+1)), int(rng.integers(lowC, highC+1))


def _build_scene(rng, geometry, mask, anchor, mode, entry_limit=3):
    H, W, t, R, C = geometry
    # Enumerate possible anchored seed locations; randomize choices instead of
    # choosing a preferred fragment or a fixed scene template.
    stub_length = int(rng.integers(2, 4)) if R-H >= 3 else 2
    locations = [(r, c+anchor[1]) for r in range(stub_length, R-H+1) for c in range(C-W+1)]
    potential = {p: _footprint(p, mask, anchor) for p in locations}
    row = int(rng.integers(stub_length, min(4, R-H+1)))
    roots = [p for p in locations if p[0] == row]
    if mode == 'parallel':
        count = 3 if entry_limit >= 3 and rng.random() < .35 else 2
        groups = [ps for ps in combinations(roots, count)
                  if all(abs(a[1]-b[1]) >= 2 and not potential[a] & potential[b]
                         for a, b in combinations(ps, 2))]
        if not groups and count == 3:
            groups = [ps for ps in combinations(roots, 2)
                      if all(abs(a[1]-b[1]) >= 2 and not potential[a] & potential[b]
                             for a, b in combinations(ps, 2))]
        if not groups:
            return None
        selected = list(groups[int(rng.integers(len(groups)))])
        entries = sorted(p[1] for p in selected)
    else:
        selected = [roots[int(rng.integers(len(roots)))]]
        entries = [selected[0][1]]
        if mode == 'combined':
            goal = int(rng.integers(2, entry_limit+1))
            extra = [c for c in range(C) if all(abs(c-e) >= 2 for e in entries)]
            rng.shuffle(extra)
            for c in extra:
                if all(abs(c-e) >= 2 for e in entries):
                    entries.append(c)
                    if len(entries) == goal:
                        break
            entries.sort()
            if len(entries) < 2:
                return None
    coarse = np.zeros((R, C), dtype=np.int8)
    coarse[:stub_length, entries] = 1
    for p in selected:
        coarse[p] = 2
    reserved = set().union(*(potential[p] for p in selected))
    if mode != 'parallel':
        _, s = _evolve(coarse, mask, anchor)
        later = [p for p in locations if p in s['forward'] and p[0] > row
                 and p[1] not in entries and not potential[p] & reserved]
        if not later:
            return None
        p = later[int(rng.integers(len(later)))]
        selected.append(p)
        coarse[p] = 2
        reserved |= potential[p]
    # Optional further contacted seeds vary cascade length, not just colors.
    if mode != 'parallel' and rng.random() < .45:
        _, s = _evolve(coarse, mask, anchor)
        extra = [p for p in locations if p in s['forward'] and not potential[p] & reserved]
        if extra:
            p = extra[int(rng.integers(len(extra)))]
            coarse[p] = 2
            selected.append(p)
            reserved |= potential[p]
    _, before = _evolve(coarse, mask, anchor)
    candidates = [p for p in locations if p not in before['forward'] and not potential[p] & reserved]
    rng.shuffle(candidates)
    for p in candidates:
        trial = coarse.copy()
        trial[p] = 2
        _, s = _evolve(trial, mask, anchor)
        if p in s['dormant'] and set(selected) <= s['activated']:
            coarse = trial
            reserved |= potential[p]
            break
    else:
        return None
    # Additional capsule placement may be causal or inert. The minimum witnesses
    # remain intact; broadening does not redefine any input interpretation.
    for _ in range(int(rng.integers(0, 3))):
        candidates = [p for p in locations if not potential[p] & reserved]
        if mode == 'parallel':
            _, current = _evolve(coarse, mask, anchor)
            candidates = [p for p in candidates if p not in current['forward']]
        if not candidates:
            break
        p = candidates[int(rng.integers(len(candidates)))]
        trial = coarse.copy()
        trial[p] = 2
        _, s = _evolve(trial, mask, anchor)
        if set(selected) <= s['activated'] and s['dormant'] and (
                mode != 'parallel' or len({r for r, _ in s['activated']}) == 1):
            coarse = trial
            reserved |= potential[p]
    out, s = _evolve(coarse, mask, anchor)
    if not s['dormant'] or len(s['activated']) < 2:
        return None
    if mode != 'parallel' and len({r for r, _ in s['activated']}) < 2:
        return None
    no_B, singleton = _evolve(coarse, mask, anchor, unfold=False)
    no_branch, b = _evolve(coarse, mask, anchor, branch=False)
    all_active, _ = _evolve(coarse, mask, anchor, activate_all=True)
    # A glyph addition alone would be a weak interaction witness. Require that
    # unfolding changes an actual ray tile, and branching changes activation.
    if np.array_equal(out == 1, no_B == 1) or np.array_equal(out, all_active):
        return None
    if mode != 'parallel' and not (s['activated'] - b['activated']):
        return None
    if mode != 'parallel' and not (s['activated'] - singleton['activated']):
        return None
    if np.array_equal(out, no_branch):
        return None
    return coarse, out, s


def _make_pair(rng, mode, palette, rotation, seen, fixed_geometry=None, entry_limit=3,
               forbidden_stencil=None):
    for attempt in range(180):
        geometry = fixed_geometry or _geometry(rng, mode)
        H, W, t, R, C = geometry
        mask, anchor = _mask(rng, (H, W))
        if _stencil_key(mask, anchor) == forbidden_stencil:
            continue
        scene = _build_scene(rng, geometry, mask, anchor, mode, entry_limit)
        if scene is None:
            continue
        coarse, out, s = scene
        divider, obg, texture, guide, ink, marker, third = palette
        colors = [obg, texture] + ([third] if rng.random() < .25 else [])
        labels = np.concatenate((np.arange(len(colors)), rng.integers(len(colors), size=H*W-len(colors))))
        rng.shuffle(labels)
        ordinary = np.array(colors, dtype=np.int8)[labels.reshape(H, W)]
        backgrounds = [c for c in colors if np.any(ordinary[~mask] != c)]
        background = int(rng.choice(backgrounds))
        glyph = np.where(mask, ink, background).astype(np.int8)
        glyph[anchor] = marker
        a = np.rot90(_paint(coarse, ordinary, glyph, guide, divider, t), rotation).copy()
        o = np.rot90(_paint(out, ordinary, glyph, guide, divider, t), rotation).copy()
        if _key(a) in seen:
            continue
        _infer(a)  # Input interpretation must succeed without any hidden data.
        return {'input': a, 'output': o}
    # Finite constructive recovery, with hundreds of different actual inputs.
    # It uses the same local rule; presentation preferences can be relaxed.
    divider, obg, texture, guide, ink, marker, _ = palette
    mask = np.array([[1, 1], [1, 0]], dtype=bool)
    if _stencil_key(mask, (0, 0)) == forbidden_stencil:
        mask = np.array([[1, 1], [0, 1]], dtype=bool)
    for R in (7, 8, 9):
        for r in (2, 3):
            for c in (2, 3, 4):
                if mode == 'parallel' and c == 4:
                    continue
                coarse = np.zeros((R, 9), dtype=np.int8)
                coarse[:2, [c] if mode == 'chain' else [c, c+4 if mode == 'parallel' else 8]] = 1
                coarse[r, c] = 2
                coarse[r if mode == 'parallel' else r+2, c+4 if mode == 'parallel' else c+2] = 2
                coarse[r+2, 0] = 2
                out, _ = _evolve(coarse, mask, (0, 0))
                for code in range(1, 15):
                    ordinary = np.array([texture if code & (1 << i) else obg for i in range(4)], dtype=np.int8).reshape(2, 2)
                    background = texture if ordinary[1, 1] == obg else obg
                    glyph = np.where(mask, ink, background).astype(np.int8)
                    glyph[0, 0] = marker
                    a = np.rot90(_paint(coarse, ordinary, glyph, guide, divider, 1), rotation).copy()
                    o = np.rot90(_paint(out, ordinary, glyph, guide, divider, 1), rotation).copy()
                    if _key(a) not in seen:
                        _infer(a)
                        return {'input': a, 'output': o}
    raise AssertionError("more fallback inputs excluded than legal puzzle size")


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(value >= low and (high is None or value <= high), name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 7, replace=False)))
    fixed_palette = bool(rng.random() < .35)
    orientation = str(rng.choice(['fixed', 'mixed', 'transfer']))
    base_rotation = int(rng.integers(4))
    fixed_geometry = None
    if rng.random() < .25:
        # Both witness modes have room in this geometry; masks and seed locations
        # still vary. Other puzzles vary tile shape and thickness between pairs.
        W = int(rng.choice([2, 3]))
        fixed_geometry = (2, W, 1, int(rng.integers(7, 10)), 7 if W == 3 else int(rng.integers(7, 10)))
    puzzle, seen, taught_entries = {'train': [], 'test': []}, set(), 2
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            if split == 'train':
                mode = ('chain' if i == 0 else 'parallel') if i < 2 else str(rng.choice(['chain', 'parallel']))
            else:
                mode = 'combined'
            local_palette = palette if fixed_palette else tuple(map(int, rng.choice(10, 7, replace=False)))
            rotation = base_rotation
            if orientation == 'mixed':
                rotation = (base_rotation + i + int(split == 'test')) % 4
            elif orientation == 'transfer':
                rotation = (base_rotation + (i % 2 if split == 'train' else 2 + i % 2)) % 4
            forbidden = None
            if split == 'train' and i == 1:
                first = _infer(puzzle['train'][0]['input'])
                forbidden = _stencil_key(first['mask'], first['anchor'])
            pair = _make_pair(rng, mode, local_palette, rotation, seen, fixed_geometry,
                              3 if split == 'train' else taught_entries, forbidden)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            if split == 'train':
                taught_entries = max(taught_entries, len(_infer(pair['input'])['entries']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    info = _infer(pair['input'])
    out, stats = _evolve(info['coarse'], info['mask'], info['anchor'])
    info.update(stats=stats, output_coarse=out)
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, o = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == o.shape and _key(a) not in seen, "duplicate input or changed shape")
            seen.add(_key(a))
            info = _pair_info(pair)
            expected = solve(a)
            _require(np.array_equal(expected, o), "wrong anchored unfolding/ray output")
            _require(not np.array_equal(a, o), "unchanged pair")
            # Original tiles and all dividers are preserved; new copies replace
            # whole ordinary tiles. Counterfactuals assess parent contributions.
            canon_a, canon_o = np.rot90(a, info['k']), np.rot90(o, info['k'])
            H, W, t, R, C = info['geometry']
            protect = canon_a == info['divider']
            _require(np.array_equal(canon_a[protect], canon_o[protect]), "divider changed")
            for r, c in np.argwhere(info['coarse'] != 0):
                y, x = t+int(r)*(H+t), t+int(c)*(W+t)
                _require(np.array_equal(canon_a[y:y+H, x:x+W], canon_o[y:y+H, x:x+W]), "original tile changed")
            s = info['stats']
            _require(s['expanded'] and len(s['activated']) >= 2 and s['dormant'], "missing causal unfolding witness")
            no_B, singleton = _evolve(info['coarse'], info['mask'], info['anchor'], unfold=False)
            all_active, _ = _evolve(info['coarse'], info['mask'], info['anchor'], activate_all=True)
            no_branch, b = _evolve(info['coarse'], info['mask'], info['anchor'], branch=False)
            _require(not np.array_equal(info['output_coarse'] == 1, no_B == 1), "blueprint never affects routing")
            _require(not np.array_equal(info['output_coarse'], all_active), "causal activation is redundant")
            _require(not np.array_equal(info['output_coarse'], no_branch), "branching is redundant")
            if split == 'train':
                _require(len(info['entries']) == 1 or len({r for r, _ in s['activated']}) == 1,
                         "training must separate cascades from parallel activation")
                training.append(info)
            else:
                _require(len(info['entries']) >= 2 and len({r for r, _ in s['activated']}) >= 2,
                         "test must combine multiple entries with a cascade")
                _require(s['activated'] - b['activated'], "test cascade does not depend on branching")
                _require(s['activated'] - singleton['activated'], "test cascade does not depend on the blueprint")
                _require(len(info['entries']) <= max(len(f['entries']) for f in training),
                         "test introduces an undemonstrated entry count")
    _require(len(training[0]['entries']) == 1, "first demonstration must isolate a cascade")
    first = training[0]
    _, stopped = _evolve(first['coarse'], first['mask'], first['anchor'], branch=False)
    _, singleton = _evolve(first['coarse'], first['mask'], first['anchor'], unfold=False)
    _require(first['stats']['activated'] - stopped['activated'], "training lacks branch-triggered growth")
    _require(first['stats']['activated'] - singleton['activated'], "training lacks blueprint-dependent activation")
    _require(len(training[1]['entries']) >= 2 and any(len(p) >= 2 for _, p in training[1]['stats']['activation_layers']),
             "second demonstration must show simultaneous activation")
    _require(_stencil_key(training[0]['mask'], training[0]['anchor']) !=
             _stencil_key(training[1]['mask'], training[1]['anchor']),
             "the first two demonstrations need different anchored stencils")


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, geometries, masks, scenes, palettes, directions, seed_counts, densities = set(), set(), set(), set(), set(), set(), set(), set()
    features = dict(clipped=0, simultaneous=0, multi_hit=0, merging=0, ternary=0)
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train'] + puzzle['test']:
            f = _pair_info(pair)
            s = f['stats']
            shapes.add(np.asarray(pair['input']).shape)
            geometries.add(f['geometry'])
            masks.add((_key(f['mask']), f['anchor']))
            scenes.add((_key(f['coarse']), _key(f['mask']), f['anchor']))
            palettes.add(tuple(map(int, np.unique(pair['input']))))
            directions.add(f['k'])
            seed_counts.add(len(f['seeds']))
            densities.add(int(f['mask'].sum()))
            for name in ('clipped', 'simultaneous', 'multi_hit', 'merging'):
                features[name] += bool(s[name])
            features['ternary'] += len(np.unique(f['ordinary'])) == 3
            copy = np.asarray(pair['input']).copy()
            predicted = solve(copy)
            _require(np.array_equal(copy, pair['input']) and np.array_equal(predicted, pair['output']), "solve mutates input or disagrees")
    _require(len(shapes) >= 12 and len(geometries) >= 20 and len(masks) >= 25, "geometry/design variation collapsed")
    _require(len(scenes) >= 160 and len(palettes) >= 35 and directions == set(range(4)), "scene/palette/direction diversity collapsed")
    _require(len(seed_counts) >= 3 and len(densities) >= 3 and all(features.values()), "family-specific variation missing")
    repeated = generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for s in puzzles[0] for a, b in zip(puzzles[0][s], repeated[s])
                 for k in ('input', 'output')), "seed reproducibility failed")
    for nt, nq in ((2, 1), (3, 2), (5, 4), (10, 4)):
        p = generate_puzzle(813, nt, nq)
        _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-built local regression: the marked 2x2 L unfolds on contact.
    coarse = np.zeros((7, 7), dtype=np.int8)
    coarse[:2, 2] = 1
    coarse[2, 2] = coarse[4, 4] = coarse[4, 0] = 2
    mask = np.array([[1, 1], [1, 0]], dtype=bool)
    out, s = _evolve(coarse, mask, (0, 0))
    _require(s['activated'] == {(2, 2), (4, 4)} and s['dormant'] == {(4, 0)}, "contact/cascade regression")
    _require(out[2, 3] == out[3, 2] == out[4, 5] == out[5, 4] == 2, "macro stencil regression")
    _require(np.all(out[1, 1:5] == 1) and out[3, 4] == 1, "upstream crosspiece/flank regression")
    # A lateral crosspiece brushes a seed without forward contact. It must stay
    # dormant, and its tile interrupts the newly drawn guide crosspiece.
    brush = np.zeros((7, 9), dtype=np.int8)
    brush[:2, 2] = 1
    brush[2, 2] = brush[4, 4] = brush[3, 6] = 2
    brushed, bs = _evolve(brush, mask, (0, 0))
    _require(bs['dormant'] == {(3, 6)} and brushed[3, 5] == 1 and brushed[3, 6] == 2
             and brushed[3, 7] != 2, "lateral contact activated or overwrote a dormant glyph")
    # The checker must reject damage to preserved structure and ink-only copying.
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged divider accepted")
    # Full glyph backgrounds, not just ink, must overwrite ordinary texture.
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    f = _pair_info(bad['train'][0])
    r, c = sorted(f['stats']['expanded'])[0]
    dy, dx = np.argwhere((f['ordinary'] != f['glyph']) & ~f['mask'])[0]
    H, W, t, _, _ = f['geometry']
    canon = np.rot90(bad['train'][0]['output'], f['k']).copy()
    canon[t+r*(H+t)+dy, t+c*(W+t)+dx] = f['ordinary'][dy, dx]
    bad['train'][0]['output'] = np.rot90(canon, -f['k']).copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("ink-only glyph copying accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), geometries=len(geometries),
                anchored_masks=len(masks), coarse_scenes=len(scenes), palettes=len(palettes),
                seed_counts=sorted(seed_counts), ink_counts=sorted(densities), features=features)
