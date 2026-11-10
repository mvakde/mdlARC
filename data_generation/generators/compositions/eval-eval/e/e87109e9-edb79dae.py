"""Stencil-filtered beams: keyed turns transport and intersect binary patterns.

Only standard-library Python and NumPy are used. No parent code, examples, files,
seed, or generation metadata are needed by solve().
"""
from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 487271257, 796535961, 466780805, 76869940, 835759091,
              961644072, 108526726, 769400858, 240611797, 737264966,
              774350602, 138033829, 867490233, 849455394, 860649859,
              286337107, 840879359, 112029830, 420207387, 451214659,
              236671977, 737863197, 809859255, 824326098, 967224816,
              431817472, 773030830, 637470652, 350192366, 376885527,
              570076593, 887796325, 509176040, 538195552, 93714786,
              337710335, 488888686, 526001626, 619290037, 310912698,
              647153175)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                    (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _geometry(tiles, background, source, turns):
    """Trace the original tile circuit without filtering, for layout checks.

    Records are (row, column, incoming heading, number of previous relays).
    The family excludes geometric cycles, even if a filter would hide them.
    """
    origin = tuple(map(int, np.argwhere(tiles == source)[0]))
    h, w = tiles.shape
    paths, chains = [], []
    for heading in range(4):
        r, c = origin
        d = heading
        seen, path, chain = set(), [], []
        while True:
            dr, dc = DIRS[d]
            r, c = r + dr, c + dc
            if not (0 <= r < h and 0 <= c < w):
                break
            state = (r, c, d)
            _require(state not in seen, "geometric circuit contains a cycle")
            seen.add(state)
            path.append((r, c, d, len(chain)))
            color = int(tiles[r, c])
            if color in turns:
                chain.append((r, c, color, d))
                d = (d + turns[color]) % 4
            else:
                _require(color in (background, source), "unkeyed circuit tile")
        paths.append(path)
        chains.append(chain)
    return paths, chains


def _parse(input_grid):
    a = _grid(input_grid)
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "missing instruction-band top")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing instruction-band bottom")
    bottom = int(bottoms[0])
    k = bottom - 4
    _require(2 <= k <= 4, "invalid visible stencil size")
    band = a[:bottom+1]
    walls = list(map(int, np.flatnonzero(np.all(band == frame, axis=0))))
    _require(walls[0] == 0 and walls[-1] == a.shape[1]-1, "missing band walls")
    _require(1 <= len(walls)-1 <= 3, "invalid key count")
    background = int(a[k+1, 1])
    _require(background != frame, "background and frame coincide")
    turns, stencils, mapping, panels = {}, {}, {}, {}
    expected_band = np.full_like(band, background)
    expected_band[[0, -1], :] = frame
    expected_band[:, walls] = frame
    for left, right in zip(walls, walls[1:]):
        _require(right-left-1 >= k+4, "key panel is too narrow")
        x, y = int(a[1, left+1]), int(a[1, right-1])
        _require((x == background) != (y == background), "ambiguous turn bar")
        color, turn, bar = (y, 1, right-1) if x == background else (x, -1, left+1)
        _require(color not in (frame, background) and color not in turns, "conflicting relay key")
        mask_tile = a[1:k+1, left+3:left+3+k]
        _require(np.all(np.isin(mask_tile, [background, color])), "invalid binary stencil")
        mask = mask_tile == color
        _require(mask.any(), "legend stencil must be nonempty")
        target = int(a[k+2, left+3])
        _require(target not in (frame, background), "invalid replacement ink")
        expected_band[1:k+1, bar] = color
        expected_band[1:k+1, left+3:left+3+k] = mask_tile
        expected_band[k+2, left+3:left+3+k] = target
        turns[color], stencils[color], mapping[color] = turn, mask.copy(), target
        panels[color] = (left, right)
    _require(np.array_equal(band, expected_band), "malformed instruction panel")
    crop = bottom+1
    board = a[crop:]
    _require(board.shape[0] >= 3 and board.shape[1] >= 3, "missing board")
    _require(np.all(board[[0, -1], :] == frame)
             and np.all(board[:, [0, -1]] == frame), "incomplete board frame")
    field = board[1:-1, 1:-1]
    _require(all(n % k == 0 for n in field.shape), "board must contain complete tiles")
    h, w = field.shape[0]//k, field.shape[1]//k
    tiles = field.reshape(h, k, w, k).transpose(0, 2, 1, 3)
    _require(np.all(tiles == tiles[:, :, :1, :1]), "input tiles must be solid")
    tiles = tiles[:, :, 0, 0].copy()
    other = set(map(int, np.unique(tiles))) - set(turns) - {background}
    _require(len(other) == 1, "source must have a unique unkeyed color")
    source = other.pop()
    _require(source != frame and np.count_nonzero(tiles == source) == 1, "source must be one solid square")
    _require(source not in mapping.values(), "replacement ink cannot equal emitter ink")
    _require(np.any(tiles == background), "board has no empty tile")
    paths, chains = _geometry(tiles, background, source, turns)
    return dict(grid=a, board=board, field=field, crop=crop, frame=frame,
                background=background, source=source, k=k, tiles=tiles,
                origin=tuple(map(int, np.argwhere(tiles == source)[0])),
                turns=turns, stencils=stencils, mapping=mapping, panels=panels,
                paths=paths, chains=chains)


def _render(info, *, filtering="intersection", rotate=True, straight=False,
            map_colors=True, twice=False):
    """Compute the union of four independent, stateful pattern streams.

    Optional arguments exist only for teaching-evidence ablations. solve uses
    the fixed family rule. Obstacles are always read from the original input.
    """
    k, tiles = info['k'], info['tiles']
    bg, source = info['background'], info['source']
    h, w = tiles.shape
    traffic = np.zeros((h*k, w*k), dtype=bool)
    activated, visits, events, prints = {}, set(), [], []
    for branch in range(4):
        r, c = info['origin']
        d, depth = branch, 0
        payload = np.ones((k, k), dtype=bool)
        seen = set()
        while True:
            dr, dc = DIRS[d]
            r, c = r+dr, c+dc
            if not (0 <= r < h and 0 <= c < w):
                break
            state = (r, c, d, payload.tobytes())
            _require(state not in seen, "live signal entered a cycle")
            seen.add(state)
            color = int(tiles[r, c])
            if color in info['turns']:
                turn = 0 if straight else info['turns'][color]
                before = payload.copy()
                if rotate and turn:
                    payload = np.rot90(payload, -turn).copy()
                d = (d+turn) % 4
                stencil = info['stencils'][color]
                if filtering == "solid":
                    payload = np.ones_like(payload)
                elif filtering == "reset":
                    payload = stencil.copy()
                else:
                    payload = payload & stencil
                depth += 1
                visits.add((r, c))
                if (r, c) not in activated:
                    activated[r, c] = np.zeros((k, k), dtype=bool)
                activated[r, c] |= payload
                events.append(dict(branch=branch, position=(r, c), color=color,
                                   incoming=(d-turn) % 4, turn=turn, depth=depth,
                                   before=before, after=payload.copy()))
                if not payload.any():
                    break
            elif color == bg:
                traffic[r*k:(r+1)*k, c*k:(c+1)*k] |= payload
                prints.append((branch, r, c, d, depth, payload.copy()))
            else:
                _require(color == source, "signal met an unkeyed tile")
                # A return through the emitter does not reset the signal.
    out = info['board'].copy()
    interior = out[1:-1, 1:-1]
    interior[traffic] = source
    for (r, c), mask in activated.items():
        color = int(tiles[r, c])
        target = info['mapping'][color] if map_colors else color
        if twice:
            target = info['mapping'].get(target, target)
        interior[r*k:(r+1)*k, c*k:(c+1)*k] = np.where(mask, target, bg)
    return out, dict(events=events, prints=prints, visits=visits,
                     traffic=traffic, activated=activated)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input's visible circuit and keys."""
    return _render(_parse(input_grid))[0]


def _asymmetric(mask):
    return all(not np.array_equal(mask, other)
               for other in [np.rot90(mask, t) for t in (1, 2, 3)]
               + [np.rot90(np.fliplr(mask), t) for t in range(4)])


def _mask(rng, k, dense=False):
    kind = int(rng.integers(0, 8))
    y, x = np.indices((k, k))
    if dense:
        result = rng.random((k, k)) < rng.uniform(.72, .94)
    elif kind == 0:
        result = rng.random((k, k)) < rng.uniform(.2, .9)
    elif kind == 1:
        result = (x+y) % 2 == 0
    elif kind == 2:
        result = x == y
    elif kind == 3:
        result = (x == 0) | (y == 0) | (x == k-1) | (y == k-1)
    elif kind == 4:
        result = (y == k//2) | (x == k//2)
    elif kind == 5:
        result = y % 2 == 0
    elif kind == 6:
        result = np.ones((k, k), dtype=bool)
    else:
        result = np.zeros((k, k), dtype=bool)
        result[int(rng.integers(k)), int(rng.integers(k))] = True
    if not result.any():
        result[int(rng.integers(k)), int(rng.integers(k))] = True
    return np.rot90(result, int(rng.integers(4))).copy()


def _scene(rng, h, w, colors, turns, min_depth, max_depth, teaching):
    """Grow relays on actual traced routes, with optional untouched relays.

    All coordinates are sampled on the board's tile lattice, not on a fixed
    radial or demonstration-index layout. Every insertion rechecks all routes.
    """
    for attempt in range(80):
        tiles = np.zeros((h, w), dtype=np.int8)
        if teaching:
            sr, sc = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        else:
            sr, sc = int(rng.integers(h)), int(rng.integers(w))
        tiles[sr, sc] = 1
        if teaching:
            target = int(rng.integers(4, min(7, h*w//2)+1))
        elif rng.random() < .25:
            target = int(rng.integers(1, 4))
        else:
            target = int(rng.integers(1, min(12, h*w//2)+1))
        paths, chains = _geometry(tiles, 0, 1, turns)
        for _ in range(target*18):
            if np.count_nonzero(tiles >= 2) >= target:
                break
            open_cells = [p for path in paths for p in path if tiles[p[0], p[1]] == 0]
            if not open_cells:
                break
            deep = [p for p in open_cells if p[3] == max(x[3] for x in open_cells)]
            candidates = deep if rng.random() < .72 else open_cells
            r, c, _, _ = candidates[int(rng.integers(len(candidates)))]
            color = int(rng.choice(colors))
            trial = tiles.copy()
            trial[r, c] = color
            try:
                new_paths, new_chains = _geometry(trial, 0, 1, turns)
            except ValueError:
                continue
            depth = max(map(len, new_chains))
            if depth > max_depth:
                continue
            tiles, paths, chains = trial, new_paths, new_chains
        depth = max(map(len, chains))
        # The two first teaching lessons use two distinct relay identities in
        # a successive route, so both the map and cumulative mask are visible.
        eligible = [chain for chain in chains if len(chain) >= min_depth
                    and (not teaching or chain[0][2] != chain[1][2])]
        if not eligible or depth < min_depth:
            continue
        hit_sides = {turns[p[2]] for chain in chains for p in chain}
        if teaching and hit_sides != {-1, 1}:
            continue
        # Add a genuine off-route relay in most scenes, independent of keys.
        reached = {(p[0], p[1]) for path in paths for p in path}
        unhit = [(r, c) for r in range(h) for c in range(w)
                 if tiles[r, c] == 0 and (r, c) not in reached]
        if unhit and (teaching or rng.random() < .6):
            rng.shuffle(unhit)
            for r, c in unhit[:int(rng.integers(1, min(3, len(unhit))+1))]:
                tiles[r, c] = int(rng.choice(colors))
        return tiles, eligible[int(rng.integers(len(eligible)))]
    raise RuntimeError("could not grow an acyclic teaching circuit")


def _mapping(rng, colors):
    mode = str(rng.choice(['cycle', 'chain', 'many', 'external', 'mixed']))
    order = list(map(int, rng.permutation(colors)))
    external = [x for x in range(2, 9) if x not in colors]
    if len(colors) > 1 and mode == 'cycle':
        return {c: order[(i+1) % len(order)] for i, c in enumerate(order)}
    if len(colors) > 1 and mode == 'chain':
        return {c: order[i+1] if i+1 < len(order) else int(rng.choice(external))
                for i, c in enumerate(order)}
    if mode == 'many':
        return {c: int(external[0]) for c in colors}
    if mode == 'external' or len(colors) == 1:
        return {c: int(rng.choice(external)) for c in colors}
    return {c: int(rng.choice([x for x in range(2, 9) if x != c])) for c in colors}


def _encode(rng, tiles, k, turns, masks, mapping, palette):
    h, w = tiles.shape
    width = w*k+2
    colors = list(map(int, rng.permutation(list(turns))))
    spare = width - (len(colors)*(k+5)+1)
    _require(spare >= 0, "key band does not fit")
    widths = np.full(len(colors), k+4, dtype=int)
    widths += rng.multinomial(spare, np.full(len(colors), 1/len(colors)))
    band = np.zeros((k+5, width), dtype=np.int8)
    band[[0, -1], :] = 9
    wall = 0
    for color, size in zip(colors, widths):
        right = wall+int(size)+1
        band[:, [wall, right]] = 9
        bar = wall+1 if turns[color] == -1 else right-1
        band[1:k+1, bar] = color
        band[1:k+1, wall+3:wall+3+k] = np.where(masks[color], color, 0)
        band[k+2, wall+3:wall+3+k] = mapping[color]
        wall = right
    board = np.full((h*k+2, width), 9, dtype=np.int8)
    board[1:-1, 1:-1] = np.repeat(np.repeat(tiles, k, axis=0), k, axis=1)
    return palette[np.vstack((band, board))].astype(np.int8)


def _candidate(rng, k, lesson, depth_shift=False, test=False):
    teaching = lesson in ('survive', 'extinguish')
    n = int(rng.integers(2 if teaching else 1, 4))
    colors = list(range(2, 2+n))
    turns = {c: int(rng.choice([-1, 1])) for c in colors}
    if teaching:
        shuffled = list(map(int, rng.permutation(colors)))
        turns[shuffled[0]], turns[shuffled[1]] = -1, 1
    hmax = (23-k)//k
    hmin = 4 if teaching or (depth_shift and test) else 3
    h = int(rng.integers(hmin, hmax+1))
    wmin = max(4, (n*(k+5)-1+k-1)//k)
    w = int(rng.integers(wmin, (28//k)+1))
    minimum = 3 if depth_shift and test else 2 if teaching else 1
    maximum = 2 if depth_shift and not test else int(rng.integers(max(3, minimum), 7))
    tiles, chain = _scene(rng, h, w, colors, turns, minimum, maximum, teaching)
    masks = {c: _mask(rng, k, dense=depth_shift and test) for c in colors}
    # Tune a selected two-relay route to teach intersection or extinction.
    # This chooses only visible masks, never hidden routing exceptions.
    if teaching:
        first, second = chain[0][2], chain[1][2]
        for _ in range(100):
            m = rng.random((k, k)) < rng.uniform(.4, .7)
            if 2 <= m.sum() < k*k and (k == 2 or _asymmetric(m)):
                break
        masks[first] = m
        incoming = np.rot90(m, -turns[second])
        if lesson == 'extinguish':
            masks[second] = ~incoming
        else:
            # Retain some incoming pixels, reject some, and include pixels that
            # cannot pass: differs from the old shape and from the new stencil.
            new = rng.random((k, k)) < .55
            active, inactive = np.argwhere(incoming), np.argwhere(~incoming)
            new[tuple(active[0])] = True
            new[tuple(active[-1])] = False
            new[tuple(inactive[0])] = True
            masks[second] = new
    mapping = _mapping(rng, colors)
    if lesson == 'survive':
        # Direct mapping into another keyed identity is visibly demonstrated.
        first, second = chain[0][2], chain[1][2]
        mapping[first], mapping[second] = second, first
    palette = rng.permutation(10)
    inp = _encode(rng, tiles, k, turns, masks, mapping, palette)
    info = _parse(inp)
    out, trace = _render(info)
    if teaching:
        _require(any(e['depth'] >= 2 for e in trace['events']), "missing live second encounter")
        _require(all(len({e['incoming'] for e in trace['events'] if e['turn'] == side}) >= 2
                     for side in (-1, 1)), "relative turns need multiple arrival headings")
        _require(not np.array_equal(out, _render(info, filtering='reset')[0]), "filtering is hidden")
        if lesson == 'survive':
            _require(any(e['depth'] >= 2 and 0 < e['after'].sum()
                         < info['stencils'][e['color']].sum() for e in trace['events']), "missing surviving intersection")
            _require(not np.array_equal(out, _render(info, rotate=False)[0]), "rotation is hidden")
        else:
            _require(any(not e['after'].any() for e in trace['events']), "missing extinction")
    if depth_shift and test:
        _require(max((e['depth'] for e in trace['events']), default=0) >= 3,
                 "test does not exercise its deeper route")
    _require(not np.array_equal(out, _render(info, straight=True)[0]), "turns are invisible")
    _require(not np.array_equal(out, _render(info, filtering='solid', map_colors=False)[0]),
             "stencil substitution is invisible")
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    depth_shift = bool(rng.integers(2))
    base = int(rng.choice([2, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = ('survive' if i == 0 else 'extinguish') if split == 'train' and i < 2 else 'free'
            if depth_shift:
                k = base
            elif split == 'test':
                k = 4
            else:
                k = (2+i % 2) if i < 2 else int(rng.choice([2, 3]))
            for attempt in range(240):
                try:
                    pair = _candidate(rng, k, lesson, depth_shift, split == 'test')
                except (ValueError, RuntimeError):
                    continue
                signature = _key(pair['input'])
                if signature not in seen:
                    puzzle[split].append(pair)
                    seen.add(signature)
                    break
            else:
                raise RuntimeError(f"exhausted {split} lesson construction")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, lessons = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _parse(inp)
            expected, trace = _render(info)
            _require(np.array_equal(out, expected), "incorrect filtered-beam output")
            _require(out.shape == info['board'].shape, "incorrect board crop")
            _require(np.all(out[[0, -1], :] == info['frame'])
                     and np.all(out[:, [0, -1]] == info['frame']), "frame changed")
            k, tiles = info['k'], info['tiles']
            for r, c in np.ndindex(tiles.shape):
                original = int(tiles[r, c])
                patch = out[1+r*k:1+(r+1)*k, 1+c*k:1+(c+1)*k]
                if original == info['source'] or (original in info['turns'] and (r, c) not in trace['visits']):
                    _require(np.all(patch == original), "emitter or unvisited relay changed")
                if original in info['turns'] and (r, c) in trace['visits']:
                    _require(np.array_equal(patch, np.where(trace['activated'][r, c],
                                                          info['mapping'][original], info['background'])),
                             "relay was not cleared and directly mapped")
            _require(not np.array_equal(out, _render(info, straight=True)[0]), "turns have no effect")
            _require(not np.array_equal(out, _render(info, filtering='solid', map_colors=False)[0]),
                     "stencils and substitution have no effect")
            if split == 'train':
                lessons.append((info, trace, out))
    events = [e for _, trace, _ in lessons for e in trace['events']]
    _require({e['turn'] for e in events} == {-1, 1}, "both turn meanings must be taught")
    _require(all(len({e['incoming'] for e in events if e['turn'] == side}) >= 2
                 for side in (-1, 1)), "relative turning is not demonstrated from multiple headings")
    _require(any(e['depth'] >= 2 and 0 < e['after'].sum()
                 < info['stencils'][e['color']].sum()
                 for info, trace, _ in lessons for e in trace['events']), "missing cumulative survival lesson")
    _require(any(not e['after'].any() for e in events), "missing extinction lesson")
    _require(any(not np.array_equal(out, _render(info, rotate=False)[0]) for info, _, out in lessons),
             "pattern rotation is not demonstrated")
    _require(any(not np.array_equal(out, _render(info, filtering='reset')[0]) for info, _, out in lessons),
             "intersection cannot be distinguished from resetting")
    _require(any(not np.array_equal(out, _render(info, twice=True)[0]) for info, _, out in lessons),
             "direct substitution cannot be distinguished from another remap")


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            q = _parse(pair['input'])
            _, t = _render(q)
            events = t['events']
            actual = set(map(int, q['tiles'].ravel())) & set(q['turns'])
            geometrical_hits = {(r, c) for chain in q['chains'] for r, c, _, _ in chain}
            masks = list(q['stencils'].values())
            records.append(dict(split=split, k=q['k'], shape=list(q['board'].shape),
                                tile_shape=list(q['tiles'].shape), keys=len(q['turns']),
                                relays=int(np.isin(q['tiles'], list(q['turns'])).sum()),
                                used_keys=len(actual), live_depth=max((e['depth'] for e in events), default=0),
                                geometric_depth=max(map(len, q['chains'])),
                                extinguished=sum(not e['after'].any() for e in events),
                                unvisited=int(np.isin(q['tiles'], list(q['turns'])).sum())-len(t['visits']),
                                off_axis_hits=sum(r != q['origin'][0] and c != q['origin'][1]
                                                  for r, c in geometrical_hits),
                                nonfull=sum(not m.all() for m in masks),
                                asymmetric=sum(_asymmetric(m) for m in masks),
                                mask_signatures=[m.tobytes().hex() for m in masks],
                                roles=[q['background'], q['frame'], q['source']],
                                turns=sorted(q['turns'].values()),
                                map_overlap=sum(v in q['turns'] for v in q['mapping'].values())))
    return records


def self_test():
    """Bounded schema, semantic, mutation, ablation, and 42-seed variation checks."""
    _worked_checks()
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    fingerprints = {tuple(_key(p['input']) for split in ('train', 'test') for p in q[split]) for q in puzzles}
    assert len(fingerprints) == 42
    assert all(_key(a[name]) == _key(b[name])
               for split in ('train', 'test')
               for a, b in zip(puzzles[0][split], generate_puzzle(0)[split])
               for name in ('input', 'output'))
    records = [r for q in puzzles for r in measure_puzzle(q)]
    assert {r['k'] for r in records} == {2, 3, 4}
    assert {r['keys'] for r in records} == {1, 2, 3}
    assert len({tuple(r['shape']) for r in records}) >= 20
    assert len({(tuple(r['tile_shape']), r['relays'], r['geometric_depth']) for r in records}) >= 80
    assert len({(r['k'], s) for r in records for s in r['mask_signatures']}) >= 80
    assert any(r['live_depth'] >= 3 for r in records)
    assert any(r['relays'] >= 8 for r in records)
    assert sum(r['unvisited'] > 0 for r in records) >= 30
    assert sum(r['off_axis_hits'] > 0 for r in records) >= 30
    for i in range(3):
        assert {r['roles'][i] for r in records} == set(range(10))
    # Legal counts are all exercised here, as well as by the host verifier.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            q = generate_puzzle(173+nt*10+nq, nt, nq)
            assert len(q['train']) == nt and len(q['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Recoloring every semantic role preserves the rule, including palette overlap.
    rng = np.random.default_rng(714)
    for q in puzzles[:4]:
        p = q['train'][0]
        perm = rng.permutation(10)
        assert np.array_equal(solve(perm[p['input']]), perm[p['output']])
        before = p['input'].copy()
        assert np.array_equal(solve(before), p['output'])
        assert np.array_equal(before, p['input']), "solve mutated its input"
        # Swapping complete key panels of unequal width is harmless: association
        # is by color. The source/stencil orientation inside a panel stays fixed.
        info = _parse(before)
        panels = sorted(info['panels'].values())
        slices = [before[:info['crop'], left:right].copy() for left, right in panels]
        changed = before.copy()
        changed[:info['crop']] = np.hstack(slices[::-1] + [before[:info['crop'], -1:]])
        assert np.array_equal(solve(changed), p['output'])
    return dict(seeds=42, pairs=len(records), board_shapes=len({tuple(r['shape']) for r in records}),
                circuit_profiles=len({(tuple(r['tile_shape']), r['relays'], r['geometric_depth']) for r in records}),
                masks=len({(r['k'], s) for r in records for s in r['mask_signatures']}))


def _worked_checks():
    """Hand-specified answers, rather than labels obtained with _render()."""
    rng = np.random.default_rng(812)
    k = 3
    a = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    b = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool)
    f = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool)
    z = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    tiles = np.zeros((5, 8), dtype=np.int8)
    tiles[3, 1], tiles[3, 4], tiles[1, 4], tiles[1, 6], tiles[4, 7] = 1, 2, 3, 4, 2
    inp = _encode(rng, tiles, k, {2: -1, 3: 1, 4: -1},
                  {2: a, 3: b, 4: z}, {2: 3, 3: 4, 4: 2}, np.arange(10))
    expected = np.full((17, 26), 9, dtype=np.int8)
    expected[1:-1, 1:-1] = 0

    def stamp(r, c, mask, color):
        expected[1+3*r:1+3*(r+1), 1+3*c:1+3*(c+1)] = np.where(mask, color, 0)

    full = np.ones((3, 3), dtype=bool)
    for r, c in ((3, 1), (2, 1), (1, 1), (0, 1), (4, 1), (3, 0), (3, 2), (3, 3)):
        stamp(r, c, full, 1)
    stamp(3, 4, a, 3)  # First collision copies the displayed stencil.
    stamp(2, 4, a, 1)  # A carries a holey signal to an off-axis relay.
    stamp(1, 4, f, 4)  # CW(a) AND b = f, using the direct 3 -> 4 mapping.
    stamp(1, 5, f, 1)
    stamp(4, 7, full, 2)  # An unvisited same-color relay stays solid.
    # CCW(f) AND z is empty: tile (1,6) clears and (1,7) stays empty.
    assert np.array_equal(solve(inp), expected), "worked three-relay answer disagrees"
    q = _parse(inp)
    assert not np.array_equal(_render(q, rotate=False)[0], expected)
    assert not np.array_equal(_render(q, filtering='reset')[0], expected)
    assert not np.array_equal(_render(q, twice=True)[0], expected)
    # Two patterned branches cross in an empty tile and continue independently.
    tiles = np.zeros((5, 7), dtype=np.int8)
    tiles[2, 2], tiles[2, 5], tiles[0, 2] = 1, 2, 3
    inp = _encode(rng, tiles, k, {2: -1, 3: 1}, {2: a, 3: b},
                  {2: 3, 3: 2}, np.arange(10))
    out = solve(inp)
    cross = out[1:4, 16:19]
    assert np.array_equal(cross, np.where(a | b, 1, 0)), "crossing must union active pixels"
    assert np.array_equal(out[1:4, 19:22], np.where(b, 1, 0)), "crossing changed a continuing branch"


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
