"""A tip-led stencil printer with keyed microtiles and transparent ink layers.

Only NumPy and the standard library are used. No parent generator is imported.
The decoder obtains every command, including the initial orientation, from input.
"""
from __future__ import annotations

import copy
import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # clockwise: U, R, D, L
COUNTERFACTUALS = (
    "tips_only", "fixed_footprint", "fixed_tile", "reverse_order",
    "opaque_holes", "last_only", "solid_tiles", "no_key", "along_segments",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _rotate(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _footprint(offsets, tip, turns):
    return {(tip[0] + r, tip[1] + c) for r, c in
            (_rotate(p, turns) for p in offsets)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _heading(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "route leg must be orthogonal and nonzero")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _identify_head(cells):
    """Find the unique reflection axis and projecting end, from actual pixels."""
    _require(len(cells) >= 3, "printhead too small")
    choices = []
    for tip in sorted(cells):
        for d in range(4):
            canonical = {_rotate((r-tip[0], c-tip[1]), 1-d) for r, c in cells}
            if (all(c <= 0 for r, c in canonical)
                    and {p for p in canonical if p[1] == 0} == {(0, 0)}
                    and any(r != 0 for r, c in canonical)
                    and {(-r, c) for r, c in canonical} == canonical):
                choices.append((tip, d))
    _require(len(choices) == 1, "head has no unique geometric tip and facing")
    tip, direction = choices[0]
    # Offsets in the visible initial orientation; no mask vocabulary lookup.
    return tip, direction, tuple(sorted((r-tip[0], c-tip[1]) for r, c in cells))


def _order_route(start, sites):
    remaining = set(sites)
    route, directions = [start], []
    while remaining:
        current = route[-1]
        rays = {}
        for q in remaining:
            if q[0] == current[0] or q[1] == current[1]:
                d = _heading(current, q)
                rays.setdefault(d, []).append(q)
        _require(len(rays) == 1, "route must have exactly one occupied ray at every step")
        d, candidates = next(iter(rays.items()))
        q = min(candidates, key=lambda p: abs(p[0]-current[0])+abs(p[1]-current[1]))
        route.append(q)
        directions.append(d)
        remaining.remove(q)
    return route, directions


def _parse(input_grid):
    """Read a framed macro board and a right-hand stack of keyed stamp cards."""
    g = _grid(input_grid)
    sep = int(g[0, 0])
    # The board's top border ends at the one-column background gutter.
    ends = np.flatnonzero(g[0] != sep)
    _require(len(ends) > 0, "missing board/catalog gutter")
    board_width = int(ends[0])
    bg = int(g[0, board_width])
    card_col = board_width + 1
    s = g.shape[1] - card_col - 2
    _require(2 <= s <= 4 and 8 <= board_width <= 17, "invalid tile or board width")
    _require(bool(np.all(g[:, board_width] == bg)), "gutter must be background")
    _require(bg != sep, "structural color must differ from background")
    breaks = np.flatnonzero(g[:, 0] != sep)
    board_height = int(breaks[0]) if len(breaks) else g.shape[0]
    h, w = board_height - 2, board_width - 2
    _require(6 <= h <= 15 and max(h*s, w*s) <= 30, "macro board cannot expand in bounds")
    board_frame = g[:board_height, :board_width]
    _require(bool(np.all(board_frame[[0, -1], :] == sep)
                  and np.all(board_frame[:, [0, -1]] == sep)), "broken board frame")
    _require(bool(np.all(g[board_height:, :board_width] == bg)), "noise below board")
    board = board_frame[1:-1, 1:-1]

    right = g[:, card_col:]
    rows = np.flatnonzero(np.any(right != bg, axis=1))
    _require(len(rows) > 0, "missing stamp cards")
    catalog_height = int(rows[-1]) + 1
    _require((catalog_height+1) % (s+5) == 0, "invalid card stack height")
    k = (catalog_height+1) // (s+5)
    _require(2 <= k <= 4, "library must contain two to four identities")
    templates, keys, order = {}, {}, []
    for j in range(k):
        r0 = j * (s+5)
        card = right[r0:r0+s+4]
        _require(card.shape == (s+4, s+2), "incomplete card")
        _require(bool(np.all(card[[0, s+1, s+3], :] == sep)
                      and np.all(card[:, [0, -1]] == sep)), "broken card frame/divider")
        motif = card[1:s+1, 1:s+1]
        colors = np.unique(motif[motif != bg])
        _require(len(colors) == 1, "template must have one visible source identity")
        label = int(colors[0])
        _require(label != sep and label not in templates, "duplicate or structural template identity")
        keyrow = card[s+2, 1:s+1]
        target = int(keyrow[s//2])
        _require(target != bg, "key must be visible")
        expected = np.full(s, bg, dtype=np.int8)
        expected[s//2] = target
        _require(np.array_equal(keyrow, expected), "key row must have one centered swatch")
        templates[label], keys[label] = motif == label, target
        order.append(label)
        if j+1 < k:
            _require(bool(np.all(right[r0+s+4] == bg)), "card gutter must be background")
    _require(bool(np.all(right[catalog_height:] == bg)), "noise below catalog")
    _require(g.shape[0] == max(board_height, catalog_height), "unencoded outer padding")

    other = set(map(int, np.unique(board))) - {bg, *templates}
    _require(len(other) == 1, "board must contain one separate printhead color")
    ink = next(iter(other))
    _require(ink != sep, "printhead may not use frame color")
    cells = {(int(r), int(c)) for r, c in zip(*np.nonzero(board == ink))}
    tip, initial, offsets = _identify_head(cells)
    sites = {(int(r), int(c)): int(board[r, c]) for r, c in
             zip(*np.nonzero(np.isin(board, list(templates))))}
    _require(len(sites) >= 2 and set(sites.values()) == set(templates), "all templates need waypoint use")
    locations = list(sites)
    for i, q in enumerate(locations):
        _require(all(max(abs(q[0]-r), abs(q[1]-c)) > 1 for r, c in cells),
                 "waypoint touches the initial head")
        _require(all(max(abs(q[0]-p[0]), abs(q[1]-p[1])) > 1 for p in locations[i+1:]),
                 "waypoints must be visibly separated singletons")
    route, directions = _order_route(tip, sites)
    _require(directions[0] == initial, "first leg must follow the visible tip")
    footprints = [_footprint(offsets, p, d-initial) for p, d in zip(route[1:], directions)]
    _require(all(_inside(f, (h, w)) for f in footprints), "arrival footprint crosses board boundary")
    return dict(bg=bg, sep=sep, ink=ink, shape=(h, w), s=s, k=k,
                templates=templates, keys=keys, order=tuple(order), offsets=offsets,
                initial=initial, route=route, directions=directions, sites=sites,
                footprints=footprints)


def _render(p, alternative=None):
    """Blockwise input-only reconstruction, also used for shortcut audits."""
    s, bg = p['s'], p['bg']
    output = np.full(tuple(d*s for d in p['shape']), bg, dtype=np.int8)
    events = list(zip(p['route'][1:], p['directions']))
    if alternative == 'reverse_order':
        events.reverse()
    elif alternative == 'last_only':
        events = events[-1:]
    elif alternative == 'along_segments':
        events = []
        for a, b, d in zip(p['route'], p['route'][1:], p['directions']):
            dr, dc = DIRS[d]
            length = abs(b[0]-a[0])+abs(b[1]-a[1])
            for distance in range(1, length+1):
                events.append(((a[0]+dr*distance, a[1]+dc*distance), d, p['sites'][b]))
    for event in events:
        tip, d = event[:2]
        label = p['sites'][tip] if len(event) == 2 else event[2]
        turns = (d-p['initial']) % 4
        cells = (_footprint(p['offsets'], tip, 0 if alternative == 'fixed_footprint' else turns)
                 if alternative != 'tips_only' else {tip})
        motif = np.rot90(p['templates'][label], -turns if alternative != 'fixed_tile' else 0)
        if alternative == 'solid_tiles':
            motif = np.ones((s, s), dtype=bool)
        color = label if alternative == 'no_key' else p['keys'][label]
        for r, c in cells:
            # Valid rule events are always in bounds. Shortcut segment/frozen
            # footprints may cross an edge; compare their in-board paint only.
            if not (0 <= r < p['shape'][0] and 0 <= c < p['shape'][1]):
                continue
            tile = output[r*s:(r+1)*s, c*s:(c+1)*s]
            if alternative == 'opaque_holes':
                tile[:] = bg
            tile[motif] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and paint using only this input; no examples, seed, or caches."""
    return _render(_parse(input_grid))


def _arrow(span, extension, solid, tails):
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span+1):
        rows = range(-depth, depth+1) if solid else (-depth, depth)
        cells.update((r, 1-extension-depth) for r in rows)
    for tail in range(1, tails+1):
        cells.update((r, 1-extension-span-tail) for r in (-span, span))
    return tuple(sorted(cells))


def _sample_route(rng, shape, canonical, count):
    """Grow fresh irregular walks, excluding competing rays at every old tip."""
    for _ in range(60):
        start = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        initial = int(rng.integers(4))
        first_cells = _footprint(canonical, start, initial-1)
        if not _inside(first_cells, shape):
            continue
        points, directions, prints = [start], [], []
        straight_bias = float(rng.uniform(.5, 2.5))
        while len(directions) < count:
            options, weights = [], []
            current = points[-1]
            for d, (dr, dc) in enumerate(DIRS):
                if not directions and d != initial:
                    continue
                for distance in range(2, max(shape)):
                    q = (current[0]+dr*distance, current[1]+dc*distance)
                    footprint = _footprint(canonical, q, d-1)
                    if not _inside(footprint, shape):
                        continue
                    if any(max(abs(q[0]-r), abs(q[1]-c)) <= 1 for r, c in first_cells):
                        continue
                    if any(max(abs(q[0]-r), abs(q[1]-c)) <= 1 for r, c in points[1:]):
                        continue
                    valid = True
                    for old, nxt in zip(points[:-1], points[1:]):
                        if old[0] == q[0] or old[1] == q[1]:
                            # A later collinear site must be on that same ray,
                            # farther than the already selected next site.
                            if (q == old or _heading(old, q) != _heading(old, nxt)
                                    or abs(q[0]-old[0])+abs(q[1]-old[1]) <=
                                    abs(nxt[0]-old[0])+abs(nxt[1]-old[1])):
                                valid = False
                                break
                    if valid:
                        options.append((q, d, footprint))
                        overlap = any(bool(footprint & f) for f in prints)
                        weights.append((straight_bias if directions and d == directions[-1] else 1.)
                                       * (3. if overlap else 1.))
            if not options:
                break
            weights = np.asarray(weights)
            q, d, f = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            points.append(q)
            directions.append(d)
            prints.append(f)
        if len(directions) == count:
            return points, directions, initial
    return None


def _motif(rng, s):
    mode = int(rng.integers(7))
    mask = np.zeros((s, s), dtype=bool)
    if mode == 0:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2:
        mask[int(rng.integers(s)), :] = True
    elif mode == 3:
        mask = np.eye(s, dtype=bool)
    elif mode == 4:
        rr, cc = np.indices((s, s))
        mask = (rr+cc) % 2 == int(rng.integers(2))
    else:
        mask = rng.random((s, s)) < float(rng.uniform(.15, .8))
    if not mask.any():
        mask.flat[int(rng.integers(s*s))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _encode(shape, bg, sep, ink, canonical, route, initial, templates, keys, sites, order):
    h, w = shape
    s = next(iter(templates.values())).shape[0]
    k = len(order)
    height = max(h+2, k*(s+5)-1)
    result = np.full((height, w+s+5), bg, dtype=np.int8)
    result[:h+2, :w+2] = sep
    board = result[1:h+1, 1:w+1]
    board[:] = bg
    for r, c in _footprint(canonical, route[0], initial-1):
        board[r, c] = ink
    for (r, c), label in sites.items():
        board[r, c] = label
    col = w+3
    for j, label in enumerate(order):
        r0 = j*(s+5)
        card = result[r0:r0+s+4, col:]
        card[:] = sep
        card[1:s+1, 1:s+1] = np.where(templates[label], label, bg)
        card[s+2, 1:s+1] = bg
        card[s+2, 1+s//2] = keys[label]
    return result


def _construct_output(shape, bg, canonical, route, directions, initial, s, templates, keys, sites):
    """Generation uses full-canvas Kronecker layers; solver uses individual tiles."""
    output = np.full(tuple(d*s for d in shape), bg, dtype=np.int8)
    for tip, d in zip(route[1:], directions):
        macro = np.zeros(shape, dtype=bool)
        for r, c in _footprint(canonical, tip, d-1):
            macro[r, c] = True
        label = sites[tip]
        micro = np.rot90(templates[label], -(d-initial))
        output[np.kron(macro, micro)] = keys[label]
    return output


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _shortcut_differences(p, output):
    return {name: int(np.count_nonzero(output != _render(p, name))) for name in COUNTERFACTUALS}


def _is_witness(p, output):
    targets = set(p['keys'].values())
    encounter_order = tuple(dict.fromkeys(p['sites'][q] for q in p['route'][1:]))
    return (len(targets) == p['k']
            and all(p['keys'][label] != label for label in p['templates'])
            and targets <= set(map(int, np.unique(output)))
            and encounter_order != p['order']
            and all(_shortcut_differences(p, output).values()))


def _make_pair(rng, s, k, count, need_witness=False, excluded_mask=None, long_route=False):
    for attempt in range(240):
        cap = 30//s
        low = 9 if long_route else 7 if need_witness else 6
        shape = tuple(int(x) for x in rng.integers(low, cap+1, size=2))
        span = 2 if min(shape) >= 10 and rng.random() < .25 and not need_witness else 1
        extension = int(rng.integers(1, 3))
        tails = int(rng.integers(2))
        if need_witness and extension == 1:
            tails = 1  # Some depth is needed for overlapping arrival footprints.
        canonical = _arrow(span, extension, bool(rng.integers(2)), tails)
        if canonical == excluded_mask:
            continue
        sampled = _sample_route(rng, shape, canonical, count)
        if sampled is None:
            continue
        route, directions, initial = sampled
        if need_witness:
            prints = [_footprint(canonical, q, d-1) for q, d in zip(route[1:], directions)]
            if len(set(directions)) == 1 or not any(a & b for i, a in enumerate(prints) for b in prints[i+1:]):
                continue
        palette = [int(x) for x in rng.choice(10, 3+k, replace=False)]
        bg, sep, ink, *labels = palette
        templates = {label: _motif(rng, s) for label in labels}
        if not need_witness and rng.random() < .15:
            templates[labels[-1]] = templates[labels[0]].copy()
        target_pool = [c for c in range(10) if c != bg]
        if need_witness:
            targets = [int(c) for c in rng.choice(target_pool, k, replace=False)]
            if any(a == b for a, b in zip(targets, labels)):
                continue
        else:
            targets = [int(c) for c in rng.choice(target_pool, k)]
        keys = dict(zip(labels, targets))
        # Mandatory uses, then uneven repetition. Identity order is independent
        # of route order, card order, shape, and target color.
        uses = labels.copy()
        if count > k:
            probs = rng.dirichlet(np.full(k, .6))
            uses += [int(c) for c in rng.choice(labels, count-k, p=probs)]
        rng.shuffle(uses)
        sites = dict(zip(route[1:], uses))
        order = [int(c) for c in rng.permutation(labels)]
        if need_witness and order == list(dict.fromkeys(uses)):
            order = order[1:]+order[:1]
        inp = _encode(shape, bg, sep, ink, canonical, route, initial, templates, keys, sites, order)
        output = _construct_output(shape, bg, canonical, route, directions, initial,
                                   s, templates, keys, sites)
        if need_witness:
            parsed = _parse(inp)
            if not _is_witness(parsed, output):
                continue
        return {'input': inp, 'output': output}, canonical
    raise RuntimeError("bounded construction could not find a teaching example")


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # route length, intermediate tile scale, library count
    shared_s = int(rng.choice([2, 3])) if regime == 0 else int(rng.choice([2, 3, 4]))
    puzzle, seen, first_mask = {'train': [], 'test': []}, set(), None
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            witness = split == 'train' and i == 0
            if regime == 1:
                s = (2 if i % 2 == 0 else 4) if split == 'train' else 3
            else:
                s = shared_s
            kmax = min(4, 31//(s+5))
            if regime == 2:
                k = 2 if split == 'train' else int(rng.integers(3, kmax+1))
            else:
                k = 2 if witness else int(rng.integers(2, kmax+1))
            if regime == 0:
                count = int(rng.integers(max(k, 3), 5)) if split == 'train' else int(rng.integers(5, 8))
            else:
                count = int(rng.integers(max(k, 3), 5))
            for _ in range(12):
                pair, mask = _make_pair(rng, s, k, count, need_witness=witness,
                                       excluded_mask=first_mask if split == 'train' and i == 1 else None,
                                       long_route=regime == 0 and split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct-pair construction exhausted')
            if witness:
                first_mask = mask
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pixel_audit(p, output):
    """Check chronological ownership at pixel coordinates, without tile writes."""
    s, bg = p['s'], p['bg']
    owners = {}
    for tip, direction, footprint in zip(p['route'][1:], p['directions'], p['footprints']):
        label = p['sites'][tip]
        turns = (direction-p['initial']) % 4
        for u, v in zip(*np.nonzero(p['templates'][label])):
            u, v = int(u), int(v)
            for _ in range(turns):
                u, v = v, s-1-u
            for r, c in footprint:
                owners[(r*s+u, c*s+v)] = p['keys'][label]
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            _require(int(output[r, c]) == owners.get((r, c), bg), "wrong pixel ownership/paint order")


def validate_generated_puzzle(puzzle) -> None:
    """Check schema, recoverability, paint invariants, and demonstration evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, parsed = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            p = _parse(inp)
            _require(out.shape == tuple(s*p['s'] for s in p['shape']), "wrong expansion factor")
            _require(set(map(int, np.unique(out))) <= {p['bg'], *p['keys'].values()}, "unkeyed output ink")
            _pixel_audit(p, out)
            _require(np.array_equal(solve(inp), out), "input-only solver disagrees")
            parsed[split].append(p)
    train = parsed['train']
    normalized_masks = {tuple(sorted(_rotate(x, 1-p['initial']) for x in p['offsets'])) for p in train}
    _require(len(normalized_masks) >= 2, "training must demonstrate actual head geometry preservation")
    _require(any(_is_witness(p, np.asarray(pair['output']))
                 for p, pair in zip(train, puzzle['train'])), "missing integrated training witness")
    _require(len({_key(pair['input']) for pair in puzzle['train']}) >= 2, "insufficient demonstrations")
    # At least one genuine, observable shift; these checks do not consult seed.
    tests = parsed['test']
    shifted = (min(len(p['directions']) for p in tests) > max(len(p['directions']) for p in train)
               or all(p['s'] not in {q['s'] for q in train} for p in tests)
               or min(p['k'] for p in tests) > max(p['k'] for p in train))
    _require(shifted, "missing documented train/test shift")


def measure_puzzle(puzzle):
    """Descriptive structural metrics recovered from grids, for review."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            p = _parse(pair['input'])
            distances = [abs(a[0]-b[0])+abs(a[1]-b[1]) for a, b in zip(p['route'], p['route'][1:])]
            ds = p['directions']
            turn_senses = {(b-a) % 4 for a, b in zip(ds, ds[1:]) if a != b}
            counts = [sum(x == label for x in p['sites'].values()) for label in p['order']]
            records.append(dict(split=split, board=list(p['shape']), tile_size=p['s'], library_count=p['k'],
                                area=len(p['offsets']), initial=p['initial'], final=ds[-1],
                                waypoints=len(ds), distances=distances,
                                mixed_turns=1 in turn_senses and 3 in turn_senses,
                                overlaps=sum(bool(a & b) for i, a in enumerate(p['footprints'])
                                             for b in p['footprints'][i+1:]), uses=counts,
                                palette=[p['bg'], p['sep'], p['ink'], *p['order']],
                                template_areas=[int(p['templates'][x].sum()) for x in p['order']],
                                shortcuts=_shortcut_differences(p, np.asarray(pair['output']))))
    return records


def self_test():
    """Bounded 42-seed sweep, counterfactual checks, and malformed-input checks."""
    records, signatures, masks, motifs, shapes, route_shapes = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple((_key(pair['input']), _key(pair['output']))
                             for split in ('train', 'test') for pair in puzzle[split]))
        records.extend(measure_puzzle(puzzle))
        # Semantic color roles and identity matching must survive changes that
        # alter encoding order or literal colors without changing the program.
        pair = puzzle['train'][0]
        inp, out = pair['input'], pair['output']
        p = _parse(inp)
        permutation = np.random.default_rng(seed+100).permutation(10)
        _require(np.array_equal(solve(permutation[inp]), permutation[out]), "palette-role invariance failed")
        changed = inp.copy()
        col, pitch, height = p['shape'][1]+3, p['s']+5, p['s']+4
        for j in range(p['k']):
            old = p['k']-1-j
            changed[j*pitch:j*pitch+height, col:] = inp[old*pitch:old*pitch+height, col:]
        _require(np.array_equal(solve(changed), out), "catalog order used as identity")
        a, b = p['order'][:2]
        changed = inp.copy()
        changed[p['s']+2, col+1+p['s']//2] = p['keys'][b]
        changed[pitch+p['s']+2, col+1+p['s']//2] = p['keys'][a]
        expected = out.copy()
        expected[out == p['keys'][a]] = p['keys'][b]
        expected[out == p['keys'][b]] = p['keys'][a]
        _require(np.array_equal(solve(changed), expected), "visible key swatches ignored")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                masks.add(tuple(sorted(_rotate(x, 1-p['initial']) for x in p['offsets'])))
                motifs.update((p['s'], m.tobytes()) for m in p['templates'].values())
                shapes.add((pair['input'].shape, pair['output'].shape))
                # Remove translation and initial rotation from route, not colors.
                origin = p['route'][0]
                route_shapes.add(tuple(_rotate((r-origin[0], c-origin[1]), 1-p['initial'])
                                       for r, c in p['route']))
    _require(len(signatures) == 42, "seed diversity failed")
    _require(len(masks) >= 7 and len(motifs) >= 100 and len(shapes) >= 50
             and len(route_shapes) >= 100, "structural diversity failed")
    _require({r['tile_size'] for r in records} == {2, 3, 4}, "tile scale coverage failed")
    _require({r['library_count'] for r in records} == {2, 3, 4}, "library count coverage failed")
    for split in ('train', 'test'):
        rs = [r for r in records if r['split'] == split]
        _require({r['initial'] for r in rs} == set(range(4)) and
                 {r['final'] for r in rs} == set(range(4)), "direction coverage failed")
        _require(any(r['mixed_turns'] for r in rs), "missing mixed turning senses")
        _require(any(r['overlaps'] for r in rs) and any(len(set(r['distances'])) > 1 for r in rs),
                 "missing overlapping prints or unequal legs")
    first = generate_puzzle(0)
    second = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for split in first for a, b in
                 zip(first[split], second[split]) for k in ('input', 'output')), "seed nondeterminism")
    # Visible head geometry outside the sampling vocabulary: a longer shaft
    # with paired rear cells. The input-only solver may not substitute by area.
    canonical = ((0,0), (0,-1), (0,-2), (-1,-3), (1,-3), (-1,-4), (1,-4))
    shape, route, ds, initial = (10,10), [(2,4), (2,7), (6,7)], [1,2], 1
    templates = {1: np.array([[1,0], [1,1]], dtype=bool),
                 2: np.array([[0,1], [1,0]], dtype=bool)}
    keys, sites = {1:3, 2:4}, {(2,7):1, (6,7):2}
    inp = _encode(shape, 0, 8, 9, canonical, route, initial, templates, keys, sites, [2,1])
    out = _construct_output(shape, 0, canonical, route, ds, initial, 2, templates, keys, sites)
    _require(np.array_equal(solve(inp), out), "unsampled visible head rejected")
    # Every explicit count combination, including the two-demo teaching limit.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(310+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = copy.deepcopy(first)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Explicitly reject a branch even if one ray continues straight ahead.
    try:
        _order_route((3, 3), {(3, 5): 1, (5, 3): 2})
    except ValueError:
        pass
    else:
        raise AssertionError('branching route accepted')
    corrupted = first['train'][0]['input'].copy()
    corrupted[0, 0] = (int(corrupted[0, 0])+1) % 10
    try:
        solve(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('broken frame accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), head_geometries=len(masks),
                template_bitmaps=len(motifs), shape_pairs=len(shapes), normalized_routes=len(route_shapes))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
