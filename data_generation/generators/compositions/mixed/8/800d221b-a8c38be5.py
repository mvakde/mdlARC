"""Addressed ballot tiles: assembly creates the wiring on which votes travel.

Only NumPy and the standard library are used. No evidence files are loaded.
The decoder works on input cells; construction records are used only to draw
the independently constructed expected output. See task_spec.md for the rule.
"""

from itertools import product
import hashlib
import json

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
_SIDES = ((0, 1), (1, 2), (2, 1), (1, 0))
_CENTER = (1, 1)
_L = ((0, 0), (0, 1), (1, 0))
_EDGE = ((0, 0), (0, 1), (0, 2), (1, 1))
_BALLOT = ((1, 2), (2, 1), (2, 2))
_OPTIONAL = ((1, 1), (0, 2), (2, 0))
_U = ((1, 0), (2, 0), (2, 1), (2, 2), (1, 2))


def _turn(p, turns):
    """Generation/geometry coordinates only: extracted tiles never rotate."""
    r, c = p
    for _ in range(turns):
        r, c = c, 2 - r
    return r, c


_CORNER_MASKS = tuple(frozenset(_turn(p, i) for p in _L) for i in range(4))
_SIDE_MASKS = tuple(frozenset(_turn(p, i) for p in _EDGE) for i in range(4))
_MANDATORY = tuple(frozenset(_turn(p, i) for p in _BALLOT) for i in range(4))
_AVAILABLE = tuple(frozenset(_turn(p, i) for p in _BALLOT + _OPTIONAL)
                   for i in range(4))
_SIDE_PATHS = tuple(tuple(_turn(p, i) for p in _U) for i in range(4))
# Each nonempty U subset has components touching just one neighboring corner.
# Tuple entries are (indices in U, number of ports owned by each endpoint).
_STENCILS = (
    ((1,), (1, 0)), ((0, 1), (1, 0)),
    ((1, 2), (2, 0)), ((0, 1, 2), (2, 0)),
    ((3,), (0, 1)), ((3, 4), (0, 1)),
    ((2, 3), (0, 2)), ((2, 3, 4), (0, 2)),
    ((1, 3), (1, 1)), ((0, 1, 3), (1, 1)),
    ((1, 3, 4), (1, 1)), ((0, 1, 3, 4), (1, 1)),
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             bool(np.all((a >= 0) & (a <= 9))), "integer colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if 0 <= q[0] < h and 0 <= q[1] < w and q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _mask(tile, color):
    return frozenset(map(tuple, np.argwhere(tile == color)))


def _place(assembly, slot, tile):
    r, c = 3 * slot[0], 3 * slot[1]
    assembly[r:r + 3, c:c + 3] = tile


def _absolute(slot, cells):
    return {(3 * slot[0] + r, 3 * slot[1] + c) for r, c in cells}


def _decode(input_grid):
    """Recover tile addresses and semantic colors without labels or a seed."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(int(counts[bg]) > int(np.max(np.delete(counts, bg))),
             "background must be uniquely most frequent")
    components = _components(inp != bg)
    _require(len(components) == 9, "need nine separated tiles")
    extracted = []
    for cells in components:
        r, c = min(p[0] for p in cells), min(p[1] for p in cells)
        _require(len(cells) == 9 and max(p[0] for p in cells) == r + 2
                 and max(p[1] for p in cells) == c + 2,
                 "every tile must be a filled 3x3 square")
        extracted.append((inp[r:r + 3, c:c + 3], (r, c)))
    plains = [(t, p) for t, p in extracted if len(np.unique(t)) == 1]
    _require(len(plains) == 1, "need one plain scaffold collector")
    center, center_pos = plains[0]
    scaffold = int(center[0, 0])
    sides, corner_tiles = {}, []
    carriers, side_accents = set(), set()
    for tile, pos in extracted:
        if pos == center_pos:
            continue
        if not np.any(tile == scaffold):
            corner_tiles.append((tile, pos))
            continue
        candidates = []
        for color in map(int, np.unique(tile)):
            if color == scaffold:
                continue
            mask = _mask(tile, color)
            for i, expected in enumerate(_SIDE_MASKS):
                if mask != expected:
                    continue
                other = set(map(int, np.unique(tile[tile != color])))
                if len(other) == 2 and scaffold in other:
                    candidates.append((i, color, next(iter(other - {scaffold}))))
        _require(len(candidates) == 1, "side address must be unique")
        i, accent, carrier = candidates[0]
        _require(i not in sides, "duplicate side address")
        sides[i] = (tile, pos, accent)
        carriers.add(carrier)
        side_accents.add(accent)
    _require(len(sides) == 4 and len(corner_tiles) == 4 and len(carriers) == 1,
             "need four sides, four corners and one shared carrier")
    carrier = carriers.pop()
    _require(len({bg, scaffold, carrier}) == 3 and
             not side_accents & {bg, scaffold, carrier}, "overlapping roles")
    choices = []
    for tile, pos in corner_tiles:
        candidates = []
        for accent in map(int, np.unique(tile)):
            if accent in (carrier, scaffold, bg):
                continue
            motif = _mask(tile, accent)
            for i, expected in enumerate(_CORNER_MASKS):
                if motif != expected:
                    continue
                signal_cells = frozenset(map(tuple, np.argwhere(
                    (tile != accent) & (tile != carrier))))
                if not _MANDATORY[i] <= signal_cells <= _AVAILABLE[i]:
                    continue
                signals = tuple(sorted({int(tile[p]) for p in signal_cells}))
                if len(signals) != 2 or set(signals) & {bg, scaffold, carrier}:
                    continue
                n0 = sum(int(tile[p]) == signals[0] for p in signal_cells)
                n1 = len(signal_cells) - n0
                if n0 == n1:
                    continue
                majority = signals[int(n1 > n0)]
                candidates.append((i, tile, pos, accent, signals,
                                   signal_cells, majority, min(n0, n1)))
        _require(bool(candidates), "corner has no ballot/address interpretation")
        choices.append(candidates)
    interpretations = []
    for selection in product(*choices):
        if len({x[0] for x in selection}) != 4:
            continue
        if len({x[4] for x in selection}) != 1:
            continue
        signals = selection[0][4]
        accents = side_accents | {x[3] for x in selection}
        if accents & set(signals) or len({x[6] for x in selection}) != 2:
            continue
        assembly = np.empty((9, 9), dtype=np.int8)
        _place(assembly, _CENTER, center)
        positions = {_CENTER: center_pos}
        for i, (tile, pos, _) in sides.items():
            _place(assembly, _SIDES[i], tile)
            positions[_SIDES[i]] = pos
        panels = [None] * 4
        majorities, minorities = [None] * 4, [None] * 4
        for i, tile, pos, _, _, cells, majority, minority in selection:
            _place(assembly, _CORNERS[i], tile)
            positions[_CORNERS[i]] = pos
            panels[i] = _absolute(_CORNERS[i], cells)
            majorities[i], minorities[i] = majority, minority
        try:
            routing = _route_analysis(assembly, scaffold, signals, panels, majorities)
        except ValueError:
            continue
        interpretations.append(dict(assembly=assembly, positions=positions,
            background=bg, carrier=carrier, scaffold=scaffold, signals=signals,
            accents=accents, panels=panels, majorities=majorities,
            minorities=minorities, **routing))
    _require(len(interpretations) == 1, "need one complete geometric interpretation")
    return interpretations[0]


def _route_analysis(assembly, scaffold, signals, panels, majorities):
    """Trace actual assembled scaffold components; never look up stencils."""
    panel_ids = {p: i for i, cells in enumerate(panels) for p in cells}
    route_mask = assembly == scaffold
    route_mask[3:6, 3:6] = False
    routes = _components(route_mask)
    ports = ({(2, c) for c in range(3, 6)} |
             {(6, c) for c in range(3, 6)} |
             {(r, 2) for r in range(3, 6)} |
             {(r, 6) for r in range(3, 6)})
    weights, owners, labels, contacts = [0] * 4, [], [], []
    for cells in routes:
        adjacent = set()
        contact_colors = set()
        for r, c in cells:
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in panel_ids:
                    adjacent.add(panel_ids[q])
                    contact_colors.add(int(assembly[q]))
        _require(len(adjacent) == 1, "route must meet exactly one ballot panel")
        owner = adjacent.pop()
        port_count = len(set(cells) & ports)
        _require(port_count > 0, "every scaffold component must enter collector")
        weights[owner] += port_count
        owners.append(owner)
        labels.append(majorities[owner])
        contacts.append(contact_colors)
    _require(all(weights), "every corner ballot must feed a collector port")
    votes = [sum(w for w, label in zip(weights, majorities) if label == s)
             for s in signals]
    _require(min(votes) > 0 and votes[0] != votes[1], "port vote must be strict")
    return dict(routes=routes, owners=owners, labels=labels, contacts=contacts,
                weights=weights, votes=votes, winner=signals[int(votes[1] > votes[0])])


def _render(info):
    out = info['assembly'].copy()
    for cells, label in zip(info['routes'], info['labels']):
        for p in cells:
            out[p] = label
    out[4, 4] = info['winner']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the 9x9 assembled, propagated and voted grid from the input alone."""
    return _render(_decode(input_grid))


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10),
                                 (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not lo <= int(value) <= hi:
            raise ValueError(f'{name} outside its legal range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def _weights(stencils):
    weights = [0] * 4
    for i, stencil in enumerate(stencils):
        left, right = _STENCILS[stencil][1]
        weights[i] += left
        weights[(i + 1) % 4] += right
    return weights


def _topology(rng, witness, shift, split):
    if witness:
        heavy = int(rng.integers(4))
        stencils = [0] * 4
        stencils[heavy] = 2  # Two ports; only the minority corner pixel touches.
        stencils[(heavy - 1) % 4] = int(rng.choice((6, 7)))
        if rng.integers(2):
            stencils[(heavy + 1) % 4] = int(rng.integers(8, 12))
            stencils[(heavy + 2) % 4] = int(rng.choice((4, 5)))
        else:
            stencils[(heavy + 1) % 4] = int(rng.choice((0, 1)))
            stencils[(heavy + 2) % 4] = int(rng.integers(8, 12))
        return stencils, {heavy}
    # This finite rejection search involves four small stencils, not paths.
    # Feasible choices exist for every supported regime.
    for _ in range(256):
        pool = (2, 3, 6, 7, 8, 9, 10, 11) if split == 'test' and shift == 1 else range(12)
        stencils = [int(rng.choice(pool)) for _ in range(4)]
        weights = _weights(stencils)
        if not all(weights) or (split == 'train' and shift == 1 and sum(weights) == 8):
            continue
        options = [{i for i in range(4) if bits >> i & 1} for bits in range(1, 15)]
        options = [ids for ids in options if sum(weights[i] for i in ids) * 2 > sum(weights)
                   and len(ids) >= 2]
        if options:
            return stencils, options[int(rng.integers(len(options)))]
    raise RuntimeError('bounded topology search exhausted')


def _scatter(rng, tiles, bg, bounds):
    h, w = map(int, rng.integers(bounds[0], bounds[1] + 1, size=2))
    if rng.random() < .15:
        w = h
    positions = None
    # Try free independent placement, then a guaranteed separated district draw.
    for _ in range(24):
        placed = []
        for _ in range(9):
            for _ in range(60):
                r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
                if all(abs(r - x) >= 4 or abs(c - y) >= 4 for x, y in placed):
                    placed.append((r, c))
                    break
            else:
                break
        if len(placed) == 9:
            positions = placed
            break
    if positions is None:
        row_sizes = 4 + rng.multinomial(h - 12, (1/3, 1/3, 1/3))
        col_sizes = 4 + rng.multinomial(w - 12, (1/3, 1/3, 1/3))
        row_starts = (0, int(row_sizes[0]), int(sum(row_sizes[:2])))
        col_starts = (0, int(col_sizes[0]), int(sum(col_sizes[:2])))
        positions = [(row_starts[i] + int(rng.integers(row_sizes[i] - 3)),
                      col_starts[j] + int(rng.integers(col_sizes[j] - 3)))
                     for i in range(3) for j in range(3)]
    inp = np.full((h, w), bg, dtype=np.int8)
    for index, tile_index in enumerate(rng.permutation(9)):
        r, c = positions[index]
        inp[r:r + 3, c:c + 3] = tiles[int(tile_index)]
    return inp


def _make_pair(rng, roles, winner_index, witness, shared_accent, shift, split):
    bg, carrier, scaffold, s0, s1 = roles
    signals = sorted((s0, s1))
    winner, loser = signals[winner_index], signals[1 - winner_index]
    stencils, winning_corners = _topology(rng, witness, shift, split)
    accent_pool = [c for c in range(10) if c not in roles]
    naccents = 1 if shared_accent else int(rng.integers(2, 6))
    chosen = list(map(int, rng.choice(accent_pool, naccents, replace=False)))
    assignments = chosen + [int(rng.choice(chosen)) for _ in range(8 - naccents)]
    rng.shuffle(assignments)
    assembly = np.full((9, 9), carrier, dtype=np.int8)
    _place(assembly, _CENTER, np.full((3, 3), scaffold, dtype=np.int8))
    out = assembly.copy()
    tiles = [np.full((3, 3), scaffold, dtype=np.int8)]
    majorities = [winner if i in winning_corners else loser for i in range(4)]
    # Corner ballots: three mandatory inward cells and any optional extensions.
    for i, slot in enumerate(_CORNERS):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        for p in _CORNER_MASKS[i]:
            tile[p] = assignments[i]
        if witness and i in winning_corners:
            extras = int(rng.integers(2))
        elif shift == 2:
            extras = int(rng.integers(2, 4)) if split == 'test' else 1
        else:
            extras = int(rng.integers(1, 4)) if witness else int(rng.integers(4))
        optional_indices = rng.permutation(3)[:extras]
        cells = sorted(_MANDATORY[i] | {_turn(_OPTIONAL[int(j)], i) for j in optional_indices})
        nminor = int(rng.integers(1, (len(cells) - 1) // 2 + 1))
        if witness and i in winning_corners:
            minority_cells = {_turn((2, 2), i)}
        else:
            minority_cells = {cells[int(j)] for j in rng.permutation(len(cells))[:nminor]}
        for p in cells:
            tile[p] = (loser if i in winning_corners else winner) if p in minority_cells else majorities[i]
        _place(assembly, slot, tile)
        _place(out, slot, tile)
        tiles.append(tile)
    # Side records label expected output directly from their two endpoint owners.
    for i, slot in enumerate(_SIDES):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        for p in _SIDE_MASKS[i]:
            tile[p] = assignments[4 + i]
        indices, weights = _STENCILS[stencils[i]]
        for j in indices:
            tile[_SIDE_PATHS[i][j]] = scaffold
        colored = tile.copy()
        for j in indices:
            owner = i if j < 2 or (j == 2 and weights[0] == 2) else (i + 1) % 4
            colored[_SIDE_PATHS[i][j]] = majorities[owner]
        _place(assembly, slot, tile)
        _place(out, slot, colored)
        tiles.append(tile)
    out[4, 4] = winner
    bounds = (24, 30) if shift == 0 and split == 'test' else ((13, 20) if shift == 0 else (13, 30))
    inp = _scatter(rng, tiles, bg, bounds)
    # Cross-check construction labels against topology recovered from input cells.
    info = _decode(inp)
    _require(np.array_equal(info['assembly'], assembly) and
             np.array_equal(_render(info), out), 'construction/reconstruction disagreement')
    return {'input': inp, 'output': out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))
    stable_palette = bool(rng.integers(2))
    base_roles = list(map(int, rng.choice(10, 5, replace=False)))
    witness_slot = int(rng.integers(num_train))
    winner_indices = list(map(int, rng.integers(2, size=num_train)))
    if len(set(winner_indices)) == 1:
        winner_indices[int(rng.integers(num_train))] ^= 1
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = split == 'train' and i == witness_slot
            shared = witness or (split == 'test' and rng.random() < .55)
            roles = base_roles if stable_palette else list(map(int, rng.choice(10, 5, replace=False)))
            winner_index = winner_indices[i] if split == 'train' else int(rng.integers(2))
            for _ in range(16):
                pair = _make_pair(rng, roles, winner_index, witness, shared, shift, split)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(info):
    winner = info['winner']
    loser = next(s for s in info['signals'] if s != winner)
    a = info['assembly']
    panel_margin = sum(1 if s == winner else -1 for s in info['majorities'])
    raw_margin = int(np.sum(a == winner)) - int(np.sum(a == loser))
    wrong_contact = any(contact == {loser if label == winner else winner}
                        for contact, label in zip(info['contacts'], info['labels']))
    return dict(weights=tuple(sorted(info['weights'])),
                ports=sum(info['weights']), margin=abs(info['votes'][0] - info['votes'][1]),
                panel_margin=panel_margin, raw_margin=raw_margin,
                wrong_contact=wrong_contact,
                areas=tuple(len(cells) for cells in info['panels']),
                route_cells=sum(map(len, info['routes'])), accents=len(info['accents']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            info = _decode(inp)
            _require(out.shape == (9, 9) and np.array_equal(out, _render(info)), 'wrong output')
            # All non-scaffold payload cells survive assembly; the collector ring survives.
            mask = info['assembly'] != info['scaffold']
            _require(np.array_equal(out[mask], info['assembly'][mask]), 'payload changed')
            ring = out[3:6, 3:6].copy()
            ring[1, 1] = info['scaffold']
            _require(bool(np.all(ring == info['scaffold'])), 'collector border changed')
            _require(4 <= sum(info['weights']) <= 8, 'bad occupied port count')
            _require(not np.array_equal(out, info['assembly']), 'parent A has no effect')
            if split == 'train':
                train_info.append(info)
    measures = [_metrics(info) for info in train_info]
    _require(any(m['panel_margin'] < 0 and m['raw_margin'] < 0 and m['wrong_contact']
                 and m['weights'] == (1, 1, 1, 4) for m in measures),
             'training must defeat panel-count, raw-cell and contact-pixel shortcuts')
    _require({info['signals'].index(info['winner']) for info in train_info} == {0, 1},
             'training must demonstrate both signal orderings winning')
    _require(any(m['accents'] == 1 for m in measures) and
             any(m['accents'] > 1 for m in measures), 'training must show address-color reuse')


def self_test():
    """42 streamed seeds, bounded count/mutation tests and causal ablations."""
    shapes, palettes, topologies, areas, margins, fingerprints = set(), set(), set(), set(), set(), set()
    accent_counts, route_sizes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        h = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                m = _metrics(info)
                shapes.add(pair['input'].shape)
                palettes.add((info['background'], info['carrier'], info['scaffold'], *info['signals']))
                topologies.add(tuple(info['weights']))
                areas.update(m['areas'])
                margins.add(m['margin'])
                accent_counts.add(m['accents'])
                route_sizes.add(m['route_cells'])
                h.update(repr(_key(pair['input'])).encode())
        fingerprints.add(h.digest())
        # Exchange real ballot payloads between geometric addresses. Total cell
        # counts, local vote strengths and panel-label counts stay unchanged;
        # which ballot meets the heavy side routes changes the collector result.
        for pair in puzzle['train']:
            info = _decode(pair['input'])
            if _metrics(info)['panel_margin'] >= 0:
                continue
            heavy = int(np.argmax(info['weights']))
            light = (heavy + 1) % 4
            changed = pair['input'].copy()
            hr, hc = info['positions'][_CORNERS[heavy]]
            lr, lc = info['positions'][_CORNERS[light]]
            for canonical in _BALLOT + _OPTIONAL:
                ar, ac = _turn(canonical, heavy)
                br, bc = _turn(canonical, light)
                changed[hr + ar, hc + ac] = pair['input'][lr + br, lc + bc]
                changed[lr + br, lc + bc] = pair['input'][hr + ar, hc + ac]
            counter = _decode(changed)
            assert counter['winner'] != info['winner']
            assert np.array_equal(np.bincount(changed.ravel(), minlength=10),
                                  np.bincount(pair['input'].ravel(), minlength=10))
            assert solve(changed)[4, 4] != pair['output'][4, 4]
            assert counter['majorities'].count(info['winner']) == 1
            # The saved construction has no panel/scaffold contacts before
            # assembly: all nine tile components are spatially separated.
            inp = pair['input']
            bg = info['background']
            assert len(_components(inp != bg)) == 9
            break
    assert len(fingerprints) == 42
    assert len(shapes) >= 50 and len(palettes) >= 35
    assert len(topologies) >= 25 and areas == {3, 4, 5, 6}
    assert len(margins) >= 4 and accent_counts == {1, 2, 3, 4, 5}
    assert len(route_sizes) >= 7
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4000 + 10 * a + b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    p, q = generate_puzzle(2718, 2, 1), generate_puzzle(2718, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
               for k in ('input', 'output'))
    # Source order and literal palette are irrelevant to the input-only rule.
    original = p['test'][0]
    info = _decode(original['input'])
    shuffled = original['input'].copy()
    positions = list(info['positions'].values())
    for i, (r, c) in enumerate(positions):
        sr, sc = positions[(i + 1) % 9]
        shuffled[r:r + 3, c:c + 3] = original['input'][sr:sr + 3, sc:sc + 3]
    assert np.array_equal(solve(shuffled), original['output'])
    permutation = np.array((7, 2, 8, 1, 6, 0, 9, 4, 3, 5), dtype=np.int8)
    assert np.array_equal(solve(permutation[original['input']]), permutation[original['output']])
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    pair = p['test'][0]
    bad = {s: [{'input': x['input'].copy(), 'output': x['output'].copy()} for x in p[s]] for s in p}
    bad['test'][0]['output'][4, 4] = _decode(pair['input'])['scaffold']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('wrong collector accepted')
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes),
            'palettes': len(palettes), 'ordered_weight_profiles': len(topologies),
            'panel_areas': sorted(areas), 'global_margins': sorted(margins),
            'accent_counts': sorted(accent_counts), 'route_cell_counts': sorted(route_sizes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
