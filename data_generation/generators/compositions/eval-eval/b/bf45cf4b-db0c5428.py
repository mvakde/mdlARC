"""Beacon-addressed mosaics: spatially read addresses route half-turned stamps.

Only NumPy and the standard library are used. No parent code or data is loaded.
The public solver reads every convention from its one input.
"""

from __future__ import annotations

import copy
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
TILE_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    result = []
    while remaining:
        stack = [remaining.pop()]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _separated(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _anchors(shape):
    h, w = shape
    return tuple((r, c) for r in (0, h // 2, h - 1) for c in (0, w // 2, w - 1))


def _views(tile):
    # Identity, quarter-turns, and the four reflected views, with stable indices.
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _inward(shape, direction):
    h, w = shape
    dr, dc = direction
    return h // 2 - dr * (h // 2), w // 2 - dc * (w // 2)


def _read_candidate(a, bg, marker, frame_cells, beacon_cells):
    fr0, fr1, fc0, fc1 = fb = _box(frame_cells)
    fh, fw = fr1 - fr0 + 1, fc1 - fc0 + 1
    _need(fh >= 4 and fw >= 4, "frame too small")
    border = {(r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1)
              if r in (fr0, fr1) or c in (fc0, fc1)}
    _need(set(frame_cells) == border, "marker component must be a hollow frame")
    br, bc = beacon_cells[0]
    _need(not (fr0 <= br <= fr1 and fc0 <= bc <= fc1), "beacon must be outside the map")
    target = a[fr0 + 1:fr1, fc0 + 1:fc1].copy()
    _need(marker not in target, "marker cannot be an address")

    occupied = a != bg
    occupied[fr0:fr1 + 1, fc0:fc1 + 1] = False
    occupied[br, bc] = False
    pieces = _components(occupied)
    _need(2 <= len(pieces) <= 7, "expected 2..7 source tiles")
    records, tile_shape, dirs, bank_cells = [], None, set(), [(br, bc)]
    for cells in pieces:
        r0, r1, c0, c1 = box = _box(cells)
        shape = r1 - r0 + 1, c1 - c0 + 1
        _need(shape in TILE_SHAPES, "source tile dimensions must be 3 or 5 on each axis")
        _need(len(cells) == shape[0] * shape[1], "source tiles must be full rectangles")
        _need(tile_shape is None or shape == tile_shape, "all tiles in one input must share dimensions")
        tile_shape = shape
        tile = a[r0:r1 + 1, c0:c1 + 1].copy()
        _need(marker not in tile, "marker cannot occur in tiles")
        dy, dx = r0 + shape[0] // 2 - br, c0 + shape[1] // 2 - bc
        direction = int(np.sign(dy)), int(np.sign(dx))
        _need(direction in DIRECTIONS and direction not in dirs, "ambiguous compass direction")
        _need(not (dy and dx) or abs(dy) == abs(dx), "tile center must lie on a compass ray")
        _need(max(abs(dy), abs(dx)) >= max(shape) + 1, "tiles too close to beacon")
        _need(all(_separated(box, t['box']) for t in records), "tiles need blank separation")
        dirs.add(direction)
        bank_cells.extend(cells)
        records.append(dict(tile=tile, box=box, direction=direction,
                            radius=max(abs(dy), abs(dx))))
    _need(_separated(_box(bank_cells), fb), "bank and framed map need a blank separator")
    values = np.concatenate([t['tile'].ravel() for t in records])
    colors, counts = np.unique(values, return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    _need(int(max(counts)) > len(values) / 2, "tile bank needs a strict majority carrier")
    keys = []
    for t in records:
        _need(np.any(t['tile'] == carrier), "carrier must occur in every tile")
        t['key'] = int(t['tile'][_inward(tile_shape, t['direction'])])
        _need(t['key'] not in (bg, marker, carrier), "inward cell must be an address color")
        keys.append(t['key'])
    _need(len(set(keys)) == len(keys), "inward addresses must be distinct")
    _need(set(map(int, np.unique(target))) - {bg} == set(keys),
          "each source address must occur in the map, and every map address needs a source")
    _need(target.shape[0] * tile_shape[0] <= 30 and target.shape[1] * tile_shape[1] <= 30,
          "expanded map exceeds ARC bounds")
    records.sort(key=lambda t: DIRECTIONS.index(t['direction']))
    return dict(background=bg, marker=marker, carrier=carrier, target=target,
                tile_shape=tile_shape, tiles=records, frame=fb, beacon=(br, bc),
                bank_box=_box(bank_cells))


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _need(np.all(border == bg), "input must have a uniform background rim")
    candidates = []
    for marker in map(int, np.unique(a)):
        if marker == bg:
            continue
        parts = _components(a == marker)
        if len(parts) != 2:
            continue
        parts.sort(key=len)
        if len(parts[0]) != 1 or len(parts[1]) < 12:
            continue
        try:
            candidates.append(_read_candidate(a, bg, marker, parts[1], parts[0]))
        except ValueError:
            pass
    _need(len(candidates) == 1, "input must have exactly one frame/beacon interpretation")
    return candidates[0]


def _render(target, lookup, bg):
    sh, sw = next(iter(lookup.values())).shape
    result = np.full((target.shape[0] * sh, target.shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(target != bg):
        result[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = lookup[int(target[r, c])][::-1, ::-1]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read addresses from inward faces, then expand only the framed map."""
    info = _decode(input_grid)
    return _render(info['target'], {t['key']: t['tile'] for t in info['tiles']}, info['background'])


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _check_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, b = _grid(pair['input']), _grid(pair['output'])
    info = _decode(a)
    target, bg = info['target'], info['background']
    sh, sw = info['tile_shape']
    _need(b.shape == (target.shape[0] * sh, target.shape[1] * sw), "incorrect dimension products")
    lookup = {t['key']: t['tile'] for t in info['tiles']}
    # Scalar inverse coordinates, separate from the generator's block writes.
    for r in range(b.shape[0]):
        for c in range(b.shape[1]):
            address = int(target[r // sh, c // sw])
            expected = bg if address == bg else int(lookup[address][sh - 1 - r % sh, sw - 1 - c % sw])
            _need(int(b[r, c]) == expected, "incorrect address, half-turn, or background block")
    active = int(np.count_nonzero(target != bg))
    _need(np.count_nonzero(b != bg) == active * sh * sw, "incorrect occupied block count")
    _need(info['marker'] not in b, "frame/beacon leaked into output")
    return info


def _learn_candidates(pairs, infos):
    """Finite learnability audit: nine spatial sample anchors x eight D4 views.

    Per-direction candidates are filtered using demonstration outputs only.
    Color identity, seed, and sampled profiles play no part in the hypotheses.
    """
    candidates = {d: {(p, op) for p in range(9) for op in range(8)} for d in DIRECTIONS}
    observed = set()
    for pair, info in zip(pairs, infos):
        target, output = info['target'], np.asarray(pair['output'])
        sh, sw = info['tile_shape']
        for t in info['tiles']:
            d, tile = t['direction'], t['tile']
            observed.add(d)
            possible = set()
            for p, coord in enumerate(_anchors(tile.shape)):
                address = int(tile[coord])
                sites = np.argwhere(target == address)
                if address == info['background'] or not len(sites):
                    continue
                for op, view in enumerate(_views(tile)):
                    if view.shape == (sh, sw) and all(np.array_equal(
                            output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw], view)
                            for r, c in sites):
                        possible.add((p, op))
            candidates[d].intersection_update(possible)
    _need(observed == set(DIRECTIONS), "training must demonstrate all compass directions")
    _need(all(candidates.values()), "contradictory demonstrations")
    return candidates


def _max_reuse(infos):
    return max(int(np.count_nonzero(i['target'] == t['key'])) for i in infos for t in i['tiles'])


def _blank_edge(info):
    m, bg = info['target'], info['background']
    return any(np.all(side == bg) for side in (m[0], m[-1], m[:, 0], m[:, -1]))


def validate_generated_puzzle(puzzle):
    """Validate labels, visible roles, training evidence, and a real test shift."""
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid pair counts")
    infos, seen = {}, set()
    for split in ('train', 'test'):
        infos[split] = []
        for pair in puzzle[split]:
            info = _check_pair(pair)
            k = _key(_grid(pair['input']))
            _need(k not in seen, "duplicate input")
            seen.add(k)
            infos[split].append(info)
    train, tests = infos['train'], infos['test']
    _need(any(_blank_edge(i) for i in train), "training must show uncropped blank map edges")
    _need(any(np.any(i['target'] == i['background']) for i in train), "training must show inactive blocks")
    _need(any(np.count_nonzero(t['tile'] == t['key']) > 1 for i in train for t in i['tiles']),
          "training must show that an address is spatial, not a unique color")
    _need(any(any(not np.array_equal(i['tiles'][0]['tile'], t['tile']) for t in i['tiles']) for i in train),
          "single-stamp binary-mask shortcut must fail")
    _need(any(not np.array_equal(t['tile'], t['tile'][::-1, ::-1]) for i in train for t in i['tiles']),
          "unchanged-copy shortcut must fail")
    candidates = _learn_candidates(puzzle['train'], train)
    for info in tests:
        for t in info['tiles']:
            anchors, views = _anchors(t['tile'].shape), _views(t['tile'])
            for p, op in candidates[t['direction']]:
                _need(int(t['tile'][anchors[p]]) == t['key'] and
                      np.array_equal(views[op], t['tile'][::-1, ::-1]),
                      "demonstrations leave a query-relevant sampling/rotation ambiguity")
    # Three sampler regimes, expressed entirely through visible measurements.
    extent = all(any(q['target'].shape[axis] > max(t['target'].shape[axis] for t in train)
                     for axis in (0, 1)) for q in tests)
    reuse = all(_max_reuse([q]) > _max_reuse(train) for q in tests)
    aspect = any(all((i['target'].shape[0] - i['target'].shape[1]) * sign > 0 for i in train)
                 and all((i['target'].shape[0] - i['target'].shape[1]) * sign < 0 for i in tests)
                 for sign in (-1, 1))
    _need(extent or reuse or aspect, "test must require a visible structural generalization")


def _witness_tile(rng, shape, carrier, key, colors, direction):
    inward = _inward(shape, direction)
    aux = int(rng.choice([c for c in colors if c != key]))
    positions = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c) != inward]
    rng.shuffle(positions)
    for pos in positions:
        tile = np.full(shape, carrier, dtype=np.int8)
        tile[inward], tile[pos] = key, aux
        valid = [v for v in _views(tile) if v.shape == shape]
        if len({v.tobytes() for v in valid}) != len(valid):
            continue
        base = tile.copy()
        extras = int(rng.integers(0, min(5, tile.size // 4)))
        pool = [c for c in colors if c != key]
        for p in positions[:extras]:
            tile[p] = int(rng.choice(pool))
        valid = [v for v in _views(tile) if v.shape == shape]
        if len({v.tobytes() for v in valid}) != len(valid):
            tile = base
        return tile
    raise RuntimeError("could not construct a spatial/orientation witness")


def _random_tile(rng, shape, carrier, key, colors, direction, force_repeat=False):
    tile = np.full(shape, carrier, dtype=np.int8)
    inward = _inward(shape, direction)
    tile[inward] = key
    budget = tile.size // 2 - 1  # Leaves a visible strict carrier majority.
    mode = int(rng.integers(4))
    positions = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if (r, c) != inward]
    rng.shuffle(positions)
    if mode == 0:  # An optional half-turn-symmetric motif.
        tile[shape[0] - 1 - inward[0], shape[1] - 1 - inward[1]] = key
        for r, c in positions[:int(rng.integers(0, budget // 2 + 1))]:
            if (r, c) == inward or (shape[0] - 1 - r, shape[1] - 1 - c) == inward:
                continue
            trial = tile.copy()
            trial[r, c] = trial[shape[0] - 1 - r, shape[1] - 1 - c] = int(rng.choice(colors))
            if np.count_nonzero(trial != carrier) <= budget + 1:
                tile = trial
    else:
        count = int(rng.integers(0, budget + 1))
        if mode == 1:  # Prefer a line through the inward cell.
            axis = int(rng.integers(2))
            positions.sort(key=lambda p: p[axis] != inward[axis])
        elif mode == 2:  # Prefer a compact patch around it.
            positions.sort(key=lambda p: abs(p[0] - inward[0]) + abs(p[1] - inward[1]))
        for p in positions[:count]:
            tile[p] = key if mode in (1, 2) and rng.random() < .7 else int(rng.choice(colors))
    if force_repeat:
        # A competing unique noncarrier color defeats the unique-color heuristic.
        tile = np.full(shape, carrier, dtype=np.int8)
        tile[inward] = tile[positions[0]] = key
        tile[positions[1]] = int(rng.choice([c for c in colors if c != key]))
    return tile


def _make_map(rng, shape, bg, keys, blank_edge=False, touches_sides=False, repeat_floor=2):
    h, w = shape
    eligible = [(r, c) for r in range(h) for c in range(w)]
    if blank_edge:
        side = int(rng.integers(4))
        eligible = [p for p in eligible if (p[0] != 0 if side == 0 else p[0] != h - 1 if side == 1
                                            else p[1] != 0 if side == 2 else p[1] != w - 1)]
    minimum = len(keys) + repeat_floor - 1
    maximum = len(eligible) if blank_edge else len(eligible) - 1
    _need(minimum <= maximum, "map has insufficient address capacity")
    count = int(rng.integers(minimum, maximum + 1))
    rng.shuffle(eligible)
    if touches_sides:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        eligible = corners + [p for p in eligible if p not in corners]
        count = max(count, 4)
    sites = eligible[:count]
    repeated_key = int(rng.choice(keys))
    labels = list(keys) + [repeated_key] * (repeat_floor - 1)
    labels += [int(rng.choice(keys)) for _ in range(count - len(labels))]
    rng.shuffle(labels)
    target = np.full(shape, bg, dtype=np.int8)
    for p, color in zip(sites, labels):
        target[p] = color
    return target


def _layout(rng, bank, framed, bg):
    bh, bw = bank.shape
    fh, fw = framed.shape
    options = []
    for horizontal in (False, True):
        for gap in range(1, 6):
            h = max(bh, fh) + 2 if horizontal else bh + fh + gap + 2
            w = bw + fw + gap + 2 if horizontal else max(bw, fw) + 2
            if h <= 30 and w <= 30:
                options.append((horizontal, gap, h, w))
    _need(bool(options), "objects cannot fit a 30x30 input")
    horizontal, gap, h, w = options[int(rng.integers(len(options)))]
    padding = 30 if rng.random() < .25 else 5
    h += int(rng.integers(0, min(30 - h, padding) + 1))
    w += int(rng.integers(0, min(30 - w, padding) + 1))
    reverse = bool(rng.integers(2))
    if horizontal:
        start = int(rng.integers(1, w - bw - fw - gap))
        bc, fc = (start + fw + gap, start) if reverse else (start, start + bw + gap)
        br, fr = int(rng.integers(1, h - bh)), int(rng.integers(1, h - fh))
    else:
        start = int(rng.integers(1, h - bh - fh - gap))
        br, fr = (start + fh + gap, start) if reverse else (start, start + bh + gap)
        bc, fc = int(rng.integers(1, w - bw)), int(rng.integers(1, w - fw))
    a = np.full((h, w), bg, dtype=np.int8)
    a[br:br + bh, bc:bc + bw] = bank
    a[fr:fr + fh, fc:fc + fw] = framed
    return a


def _new_pair(rng, palette, tile_shape, directions, witnesses, repeated_direction,
              shift, train, blank_edge=False, train_reuse=0, aspect_sign=1):
    bg, marker, carrier = map(int, palette[:3])
    k = len(directions)
    keys = list(map(int, rng.permutation(palette[3:])[:k]))
    detail_colors = list(map(int, palette[3:3 + int(rng.integers(max(2, k), 8))]))
    detail_colors = list(dict.fromkeys(keys + detail_colors))
    tiles = []
    used_views = set()
    sh, sw = tile_shape
    for d, address in zip(directions, keys):
        # If two sources are D4-equivalent, a wrong sample address paired with
        # a different turn can imitate delivery of the other source. Exclude
        # that sampling coincidence so each witness identifies its own object.
        for _ in range(60):
            tile = (_witness_tile(rng, tile_shape, carrier, address, detail_colors, d) if d in witnesses else
                    _random_tile(rng, tile_shape, carrier, address, detail_colors, d, d == repeated_direction))
            views = {v.tobytes() for v in _views(tile) if v.shape == tile_shape}
            if views.isdisjoint(used_views):
                used_views.update(views)
                break
        else:
            raise RuntimeError('could not make distinguishable source motifs')
        upper = 6 if max(tile_shape) == 3 else 7
        radius = int(rng.integers(max(tile_shape) + 1, upper + 1))
        tiles.append((d, address, tile, radius))
    r0 = min([0] + [d[0] * radius - sh // 2 for d, _, _, radius in tiles])
    r1 = max([0] + [d[0] * radius + sh // 2 for d, _, _, radius in tiles])
    c0 = min([0] + [d[1] * radius - sw // 2 for d, _, _, radius in tiles])
    c1 = max([0] + [d[1] * radius + sw // 2 for d, _, _, radius in tiles])
    bank = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=np.int8)
    bank[-r0, -c0] = marker
    for d, _, tile, radius in tiles:
        r, c = d[0] * radius - sh // 2 - r0, d[1] * radius - sw // 2 - c0
        bank[r:r + sh, c:c + sw] = tile

    repeat_floor = train_reuse + 1 if not train and shift == 'multiplicity' else 2
    shapes = []
    for h in range(2, (4 if train else 30 // sh) + 1):
        for w in range(2, (4 if train else 30 // sw) + 1):
            capacity = h * w - (min(h, w) if blank_edge else 1)
            if capacity < k + repeat_floor - 1:
                continue
            if shift == 'extent' and not train and max(h, w) < 5:
                continue
            if shift == 'aspect' and ((h - w) * aspect_sign <= 0 if train else (h - w) * aspect_sign >= 0):
                continue
            shapes.append((h, w))
    _need(bool(shapes), "no feasible map shape")
    shape = shapes[int(rng.integers(len(shapes)))]
    # Choose an edge whose removal leaves enough room; _make_map picks uniformly
    # among edges, so both edge lengths must be feasible for this witness.
    if blank_edge and shape[0] * shape[1] - max(shape) < k + repeat_floor - 1:
        shape = (4, 3) if shift == 'aspect' and aspect_sign == 1 else (3, 4) if shift == 'aspect' else (4, 4)
    target = _make_map(rng, shape, bg, keys, blank_edge, touches_sides=train and not blank_edge,
                       repeat_floor=repeat_floor)
    framed = np.full((target.shape[0] + 2, target.shape[1] + 2), marker, dtype=np.int8)
    framed[1:-1, 1:-1] = target
    a = _layout(rng, bank, framed, bg)
    return dict(input=a, output=_render(target, {address: tile for _, address, tile, _ in tiles}, bg))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['extent', 'multiplicity', 'aspect']))
    aspect_sign = int(rng.choice([-1, 1]))
    stable_palette = bool(rng.random() < .22)
    base_palette = rng.permutation(10)
    first_shape = TILE_SHAPES[int(rng.integers(4))]
    second_shape = tuple(8 - n for n in first_shape)
    k0 = int(rng.integers(4, 8))
    first_indices = list(map(int, rng.choice(8, size=k0, replace=False)))
    missing = [j for j in range(8) if j not in first_indices]
    k1 = int(rng.integers(max(4, len(missing) + 1), 8))
    common = list(map(int, rng.choice(first_indices, size=k1 - len(missing), replace=False)))
    plans = [([DIRECTIONS[j] for j in first_indices], set(DIRECTIONS[j] for j in first_indices), None),
             ([DIRECTIONS[j] for j in missing + common], set(DIRECTIONS[j] for j in missing), DIRECTIONS[common[0]])]
    puzzle, seen, train_infos = dict(train=[], test=[]), set(), []
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            train = split == 'train'
            if train and index < 2:
                directions, witnesses, repeated = plans[index]
                tile_shape = first_shape if index == 0 else second_shape
            else:
                indices = rng.choice(8, size=int(rng.integers(2, 8)), replace=False)
                directions, witnesses, repeated = [DIRECTIONS[int(j)] for j in indices], set(), None
                tile_shape = TILE_SHAPES[int(rng.integers(4))]
            for _ in range(60):
                palette = base_palette if stable_palette else rng.permutation(10)
                pair = _new_pair(rng, palette, tile_shape, directions, witnesses, repeated,
                                 shift, train, blank_edge=train and index == 1,
                                 train_reuse=_max_reuse(train_infos) if not train else 0,
                                 aspect_sign=aspect_sign)
                try:
                    info = _decode(pair['input'])
                except ValueError:
                    continue
                if _key(pair['input']) in seen:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                if train:
                    train_infos.append(info)
                break
            else:
                raise RuntimeError('unable to generate a distinct readable pair')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _canonical(a, bg):
    roles = {bg: 0}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Bounded 42-seed semantic/diversity audit plus count and corruption checks."""
    # Hand-authored input and literal label, independent of the sampler/render
    # path. It also exercises a radius (8) outside the sampler's 4..7 range.
    hand = np.zeros((23, 25), dtype=np.int8)
    hand[9, 9] = 9
    hand[2:5, 8:11] = [[1, 1, 1], [1, 1, 3], [1, 2, 1]]
    hand[8:11, 16:19] = [[1, 1, 4], [3, 1, 1], [1, 1, 1]]
    hand[17:21, 18:23] = 9
    hand[18:20, 19:22] = [[2, 3, 2], [0, 2, 0]]
    literal = np.array([[int(c) for c in row] for row in
                        ('121111121', '311113311', '111411111',
                         '000121000', '000311000', '000111000')], dtype=np.int8)
    _need(np.array_equal(solve(hand), literal), 'hand-authored routing regression failed')
    shapes, outputs, structures, maps, palettes, counts, directions = set(), set(), set(), set(), set(), set(), set()
    tile_shapes, backgrounds, signatures = set(), set(), set()
    repeated = symmetric = irregular_radii = blank_bands = rectangular = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                _need(np.array_equal(solve(np.array(pair['input'])), pair['output']), 'solver disagreement')
                shapes.add(pair['input'].shape)
                outputs.add(pair['output'].shape)
                structures.add(_canonical(pair['input'], info['background']))
                maps.add(_canonical(info['target'], info['background']))
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                counts.add(len(info['tiles']))
                tile_shapes.add(info['tile_shape'])
                backgrounds.add(info['background'])
                directions.update(t['direction'] for t in info['tiles'])
                repeated += any(np.count_nonzero(t['tile'] == t['key']) > 1 for t in info['tiles'])
                symmetric += any(np.array_equal(t['tile'], t['tile'][::-1, ::-1]) for t in info['tiles'])
                irregular_radii += len({t['radius'] for t in info['tiles']}) > 1
                blank_bands += _blank_edge(info)
                rectangular += info['tile_shape'][0] != info['tile_shape'][1]
    _need(len(signatures) == 42 and len(structures) >= 180, 'structural diversity collapsed')
    _need(len(shapes) >= 35 and len(outputs) >= 25 and len(maps) >= 120, 'geometry diversity collapsed')
    _need(tile_shapes == set(TILE_SHAPES) and counts == set(range(2, 8)), 'tile/count range collapsed')
    _need(backgrounds == set(range(10)) and directions == set(DIRECTIONS), 'color/direction range collapsed')
    _need(min(repeated, symmetric, irregular_radii, blank_bands, rectangular) >= 20, 'motif/layout variation missing')
    a = generate_puzzle(9182, 6, 2)
    _need(_signature(a) == _signature(generate_puzzle(9182, 6, 2)), 'seed is not reproducible')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(100 + n * 5 + m, n, m)
            _need(len(p['train']) == n and len(p['test']) == m, 'count request ignored')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError('illegal seed accepted')
    # Labels, addresses, and geometry each have rejection tests.
    for corruption in ('label', 'address', 'beacon'):
        damaged = copy.deepcopy(a)
        pair = damaged['test'][0]
        info = _decode(pair['input'])
        if corruption == 'label':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif corruption == 'address':
            t = info['tiles'][0]
            r, c = _inward(info['tile_shape'], t['direction'])
            pair['input'][t['box'][0] + r, t['box'][2] + c] = info['carrier']
        else:
            pair['input'][info['beacon']] = info['background']
        try:
            validate_generated_puzzle(damaged)
        except ValueError:
            continue
        raise AssertionError('corruption accepted')
    report = dict(seeds=42, pairs=210, handcrafted_cases=1, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                  output_shapes=len(outputs), color_normalized_inputs=len(structures),
                  color_normalized_maps=len(maps), palettes=len(palettes),
                  source_counts=sorted(counts), tile_shapes=sorted(tile_shapes),
                  repeated_address_pairs=int(repeated), symmetric_tile_pairs=int(symmetric),
                  unequal_radius_pairs=int(irregular_radii), blank_edge_pairs=int(blank_bands),
                  rectangular_tile_pairs=int(rectangular))
    return report


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
