"""Legend-directed cargo queues: visible commands, exact-fit admission, stable packing.

Only standard-library modules and NumPy are used. No evidence, output labels,
seed, training examples, or construction records are consulted by solve().
Diagnostics stream bounded puzzles in memory; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "bad grid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}


def _is_rectangle(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _decode_card(a, cells):
    if not _is_rectangle(cells):
        return None
    r0, r1, c0, c1 = _bbox(cells)
    if min(r1 - r0, c1 - c0) < 3:
        return None
    sub = a[r0:r1 + 1, c0:c1 + 1]
    carrier = int(sub[0, 0])
    if any(np.any(edge != carrier) for edge in (sub[0], sub[-1], sub[:, 0], sub[:, -1])):
        return None
    others = set(map(int, np.unique(sub))) - {carrier}
    if len(others) != 2:
        return None
    readings = []
    for tip in sorted(others):
        ink = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(sub == tip)))
        trails = set(map(tuple, np.argwhere(sub == ink)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRS:
            used, lengths = set(), []
            for rr, cc in sorted(tips):
                stroke = []
                rr, cc = rr - dr, cc - dc
                while (rr, cc) in trails:
                    stroke.append((rr, cc))
                    rr, cc = rr - dr, cc - dc
                if len(stroke) < 2 or not used.isdisjoint(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                readings.append((ink, tip, (dr, dc), tuple(lengths)))
    _require(len(readings) == 1, "card has no unique trail-to-tip vector")
    ink, tip, direction, lengths = readings[0]
    return dict(carrier=carrier, ink=ink, tip=tip, direction=direction,
                lengths=lengths, cells=cells, shape=sub.shape, bbox=(r0, r1, c0, c1))


def _parse_receiver(cells):
    readings = []
    for outward in DIRS:
        perp = -outward[1], outward[0]
        kmin = min(r * outward[0] + c * outward[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum(k > 0 and qq == q for k, qq in local) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected.update((k, q) for q, n in enumerate(lengths) for k in range(1, n + 1))
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if len(runs) != 3 or any(runs[j][1] >= runs[j + 1][0] for j in (0, 1)):
            continue
        qa, qb, center = runs[1]
        if qa + qb != width:
            continue
        readings.append(dict(out=outward, perp=perp, kmin=kmin, qmin=qmin,
                             q0=qa, span=qb - qa, stop=center + 1,
                             width=width, runs=tuple(runs),
                             lengths=tuple(run[2] for run in runs), cells=cells))
    _require(len(readings) == 1, "receiver must have one base and three positive prongs")
    return readings[0]


def _world(fork, k, q):
    k, q = k + fork['kmin'], q + fork['qmin']
    return (k * fork['out'][0] + q * fork['perp'][0],
            k * fork['out'][1] + q * fork['perp'][1])


def _analyze(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background is not unique")
    bg = int(values[counts.argmax()])
    cards = []
    for cells in _components(a != bg):
        card = _decode_card(a, cells)
        if card is not None:
            cards.append(card)
    _require(cards, "no cards")
    by_color = {c['carrier']: c for c in cards}
    _require(len(by_color) == len(cards), "duplicate carrier card")
    inks, tips = {c['ink'] for c in cards}, {c['tip'] for c in cards}
    _require(len(inks) == len(tips) == 1, "inconsistent ink or tips")
    ink, tip = next(iter(inks)), next(iter(tips))
    _require(not ({bg, ink, tip} & set(by_color)) and len({bg, ink, tip}) == 3,
             "conflicting roles")
    card_cells = set().union(*(c['cells'] for c in cards))
    remaining = a.copy()
    _paint(remaining, card_cells, bg)
    tokens, forks = [], []
    for color in map(int, np.unique(remaining)):
        if color == bg:
            continue
        for cells in _components(remaining == color):
            if color in by_color:
                _require(_is_rectangle(cells), "cargo must be a solid rectangle")
                tokens.append(dict(cells=cells, color=color, direction=by_color[color]['direction']))
            else:
                _require(color not in (ink, tip), "ink or tip outside a card")
                fork = _parse_receiver(cells)
                fork['color'] = color
                forks.append(fork)
    _require(tokens and forks, "missing cargo or receivers")
    _require(set(by_color) == {t['color'] for t in tokens}, "card without cargo")
    fixed = card_cells | set().union(*(f['cells'] for f in forks))
    return dict(grid=a, background=bg, ink=ink, tip=tip, cards=cards,
                tokens=tokens, forks=forks, fixed=fixed)


def _candidate(info, token, fork, use_arrows=True, match_span=True):
    if use_arrows and token['direction'] != tuple(-x for x in fork['out']):
        return None
    local = {(r * fork['out'][0] + c * fork['out'][1] - fork['kmin'],
              r * fork['perp'][0] + c * fork['perp'][1] - fork['qmin'])
             for r, c in token['cells']}
    ks, qs = {k for k, _ in local}, {q for _, q in local}
    wanted = set(range(fork['q0'], fork['q0'] + fork['span']))
    if match_span:
        if qs != wanted:
            return None
    elif min(qs) + max(qs) != 2 * fork['q0'] + fork['span'] - 1:
        return None
    k0, k1 = min(ks), max(ks)
    if k0 < fork['stop']:
        return None
    corridor = {_world(fork, k, q) for k in range(fork['stop'], k1 + 1) for q in qs}
    h, w = info['grid'].shape
    if not corridor.isdisjoint(info['fixed']) or any(not (0 <= r < h and 0 <= c < w) for r, c in corridor):
        return None
    return dict(k0=k0, k1=k1, qs=qs, length=k1 - k0 + 1,
                distance=k0 - fork['stop'])


def _prediction(info, use_arrows=True, match_span=True, pack=True, strict=True):
    groups = [[] for _ in info['forks']]
    rejected = []
    for ti, token in enumerate(info['tokens']):
        options = []
        for fi, fork in enumerate(info['forks']):
            candidate = _candidate(info, token, fork, use_arrows, match_span)
            if candidate is not None:
                options.append((candidate['distance'], fi, candidate))
        if not options:
            rejected.append(ti)
            continue
        options.sort(key=lambda x: (x[0], x[1]))
        if strict:
            _require(len(options) == 1 or options[0][0] != options[1][0], "equal-distance receiver tie")
        # Direction-removal diagnostics can introduce ties absent from the
        # intended rule. Their sorted receiver index is a diagnostic tie-break.
        _, fi, candidate = options[0]
        groups[fi].append(dict(candidate, ti=ti, fi=fi))
    finals, swept, moves, occupied = [], set(), [], set()
    for fi, group in enumerate(groups):
        fork = info['forks'][fi]
        group.sort(key=lambda m: m['k0'])
        _require(len({m['k0'] for m in group}) == len(group), "queue depth tie")
        cursor = fork['stop']
        for m in group:
            start = cursor if pack else fork['stop']
            cells = {_world(fork, k, q) for k in range(start, start + m['length']) for q in m['qs']}
            route = {_world(fork, k, q) for k in range(start, m['k1'] + 1) for q in m['qs']}
            if strict:
                _require(start <= m['k0'], "queue overflow would reverse motion")
                _require(cells.isdisjoint(occupied), "colliding final cargo")
                _require(cells.isdisjoint(info['fixed']), "final cargo hits a fixed object")
                _require(all(0 <= r < info['grid'].shape[0] and 0 <= c < info['grid'].shape[1]
                             for r, c in cells), "final cargo outside canvas")
            occupied.update(cells)
            swept.update(route)
            finals.append((cells, info['tokens'][m['ti']]['color']))
            moves.append(dict(m, final=cells, route=route, actual_distance=m['k0'] - start))
            cursor += m['length']
    output = info['grid'].copy()
    for token in info['tokens']:
        _paint(output, token['cells'], info['background'])
    _paint(output, swept - occupied, info['ink'])
    for cells, color in finals:
        _paint(output, cells, color)
    return output, dict(groups=groups, moves=moves, rejected=rejected,
                        final=occupied, swept=swept)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the full rule from visible card geometry, cargo and receivers."""
    return _prediction(_analyze(input_grid))[0]


def _card(rng, color, ink, tip, direction):
    n = int(rng.choice([2, 2, 3]))
    length = int(rng.integers(2, 4))
    offsets = list(map(int, rng.integers(0, 2, size=n)))
    card = np.full((2 * n + 1, length + 3 + max(offsets)), color, np.int8)
    for j, offset in enumerate(offsets):
        card[1 + 2 * j, 1 + offset:1 + offset + length] = ink
        card[1 + 2 * j, 1 + offset + length] = tip
    # E -> N -> W -> S under counterclockwise array rotations.
    k = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}[direction]
    return np.rot90(card, k).copy()


def _fork_spec(rng, span):
    side = int(rng.integers(1, 3))
    gap = int(rng.integers(1, 3))
    tail = int(rng.integers(0, 2))
    # Keep enough room for seven-cell card banks on a <=30-cell canvas.
    while span + 2 * (side + gap + tail) > 13:
        tail = 0
        if span + 2 * (side + gap) > 13:
            gap = 1
    lengths = tuple(map(int, rng.integers(1, 5, size=3)))
    return dict(span=span, side=side, gap=gap, tail=tail,
                width=span + 2 * (side + gap + tail), lengths=lengths)


def _fork_cells(spec, base, lane, sign):
    pad = spec['side'] + spec['gap'] + spec['tail']
    r0 = lane - pad
    width = spec['width']
    cells = {(r0 + q, base) for q in range(width)}
    runs = ((spec['tail'], spec['tail'] + spec['side']),
            (pad, pad + spec['span']),
            (width - spec['tail'] - spec['side'], width - spec['tail']))
    for (qa, qb), length in zip(runs, spec['lengths']):
        cells.update((r0 + q, base + sign * k)
                     for q in range(qa, qb) for k in range(1, length + 1))
    return cells


def _place_cards(rng, shape, band, cards):
    # Reserve the full traffic band, then pack independently ordered cards in
    # either free bank. Thirty-two bounded tries have a guaranteed fitting tail.
    h, w = shape
    for attempt in range(33):
        occupied = set(band)
        placed = []
        for index in map(int, rng.permutation(len(cards))):
            card = cards[index]
            ch, cw = card.shape
            blocked = _expand(occupied)
            choices = [(r, c) for r in range(h - ch + 1) for c in range(w - cw + 1)
                       if _rect(r, c, ch, cw).isdisjoint(blocked)]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            occupied.update(_rect(r, c, ch, cw))
            placed.append((card, r, c))
        if len(placed) == len(cards):
            return placed
    # Each bank is at least eight cells high; every card side is <=7.
    # Two cards plus a one-cell gap fit in the >=20-cell width.
    result, col = [], 0
    for index, card in enumerate(cards):
        if index < 2:
            result.append((card, 0, col))
            col += card.shape[1] + 1
        else:
            result.append((card, h - card.shape[0], 0))
    return result


def _construct_pair(rng, ink, tip, mode, rotation):
    dual = mode != 'single'
    ncolors = 3 if mode in ('single', 'counterflow', 'query') else int(rng.integers(2, 4))
    span = int(rng.integers(1, 4))
    left = _fork_spec(rng, span)
    if mode == 'single':
        center = left['lengths'][1]
        guards = [n for n in range(1, 5) if n != center]
        left['lengths'] = (int(rng.choice(guards)), center, int(rng.choice(guards)))
    right = _fork_spec(rng, span) if dual else None
    max_width = max(left['width'], right['width'] if dual else 0, span + 2)
    # Equal parity of fork widths makes their centered transverse lane exact.
    h = int(rng.integers(max_width + 16, 31))
    band0 = int(rng.integers(8, h - max_width - 7))
    lane = band0 + (max_width - span) // 2
    lm, rm = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    ml = max(left['lengths'])
    mr = max(right['lengths']) if dual else 0
    colors = [int(x) for x in rng.permutation([c for c in range(10) if c not in (ink, tip)])]
    bg = colors[0]
    cargo_colors = colors[1:1 + ncolors]
    fork_colors = colors[1 + ncolors:3 + ncolors]
    _require(len(fork_colors) == 2, "insufficient palette")
    if rng.random() < .35:
        fork_colors[1] = fork_colors[0]
    directions = [(0, -1)] * ncolors
    if dual:
        directions[-1] = (0, 1)
    else:
        directions[-1] = [(0, 1), (1, 0), (-1, 0)][int(rng.integers(3))]
    # Cargo records here are construction-only; the parser recovers fresh ones.
    if mode == 'query':
        nleft, nright = int(rng.integers(2, 4)), 2
    elif dual:
        nleft, nright = 2, int(rng.integers(1, 3))
    else:
        nleft, nright = 2, 0
    records = []
    for j in range(nleft):
        ci = j % (ncolors - 1)
        records.append(dict(role=ci, target='left', wide=False))
    for _ in range(nright):
        records.append(dict(role=ncolors - 1, target='right', wide=False))
    if not dual:
        records.append(dict(role=ncolors - 1, target=None, wide=False))
    records.append(dict(role=int(rng.integers(ncolors - (not dual))), target=None, wide=True))
    # Make a right-commanded item start on the left side, defeating nearest-dock
    # selection. Other records are freely interleaved, including the width decoy.
    order = list(map(int, rng.permutation(len(records))))
    if dual:
        ri = next(j for j in order if records[j]['target'] == 'right')
        order.remove(ri)
        order.insert(0, ri)
        # At least one left-commanded item starts to its right.
    records = [records[j] for j in order]
    if mode in ('single', 'counterflow'):
        left_slots = [j for j, t in enumerate(records) if t['target'] == 'left']
        roles = sorted((records[j]['role'] for j in left_slots),
                       key=lambda ci: cargo_colors[ci], reverse=mode == 'single')
        for j, role in zip(left_slots, roles):
            records[j]['role'] = role
    n = len(records)
    # Maximum length budget includes both structures, two-cell detachment at
    # each end, and mandatory inter-item gaps. No rejected construction payloads.
    factor = lm + ml + (rm + mr if dual else 0) + (4 if dual else 3)
    budget = 30 - factor - (n - 1)
    _require(budget >= n, "cargo count cannot fit")
    maxlen = 3 if mode == 'query' else 2
    lengths = []
    for j in range(n):
        cap = min(maxlen, budget - (n - j - 1))
        length = int(rng.integers(1, cap + 1))
        lengths.append(length)
        budget -= length
    if mode in ('single', 'counterflow'):
        # Contrasting demonstrations rule out ordering by color or item size.
        # Both choices fit: training has <=5 slots, each at most length two.
        for j, length in zip(left_slots, (2, 1) if mode == 'single' else (1, 2)):
            lengths[j] = length
    extent = sum(lengths) + n - 1
    w = int(rng.integers(max(20, factor + extent), 31))
    leftbase = lm
    rightbase = w - 1 - rm
    start = leftbase + ml + 2
    last = rightbase - mr - 2 if dual else w - 2
    extra = last - start + 1 - extent
    _require(extra >= 0, "initial cargo does not fit")
    gaps = [1] * (n - 1)
    for j in range(n - 1):
        add = int(rng.integers(0, min(2, extra) + 1))
        gaps[j] += add
        extra -= add
    # Keep the first counterflow item visibly nearer the wrong (left) dock.
    # Remaining slack stays at the far end; it never encodes a solving rule.
    start += int(rng.integers(min(extra, 2) + 1)) if dual else int(rng.integers(extra + 1))
    a = np.full((h, w), bg, np.int8)
    fixed = _fork_cells(left, leftbase, lane, 1)
    _paint(a, fixed, fork_colors[0])
    if dual:
        rcells = _fork_cells(right, rightbase, lane, -1)
        _require(fixed.isdisjoint(rcells), "receiver overlap")
        _paint(a, rcells, fork_colors[1])
        fixed |= rcells
    x = start
    for j, (record, length) in enumerate(zip(records, lengths)):
        height = span + 2 if record['wide'] else span
        row = lane - 1 if record['wide'] else lane
        record.update(c=x, length=length, cells=_rect(row, x, height, length),
                      color=cargo_colors[record['role']])
        _require(record['cells'].isdisjoint(fixed), "cargo touches receiver")
        _paint(a, record['cells'], record['color'])
        x += length + (gaps[j] if j < n - 1 else 0)
    cards = [_card(rng, color, ink, tip, direction)
             for color, direction in zip(cargo_colors, directions)]
    band = _rect(band0, 0, max_width, w)
    for card, r, c in _place_cards(rng, (h, w), band, cards):
        a[r:r + card.shape[0], c:c + card.shape[1]] = card
    # Construct labels without solve(), the parser, or candidate checking.
    b = a.copy()
    for record in records:
        _paint(b, record['cells'], bg)
    destinations, all_sweeps, final_union = [], set(), set()
    for target in ('left', 'right'):
        group = [t for t in records if t['target'] == target]
        group.sort(key=lambda t: t['c'], reverse=target == 'right')
        cursor = leftbase + left['lengths'][1] + 1 if target == 'left' else (
            rightbase - right['lengths'][1] - 1 if dual else 0)
        for record in group:
            length = record['length']
            col = cursor if target == 'left' else cursor - length + 1
            final = _rect(lane, col, span, length)
            lo = min(col, record['c'])
            hi = max(col + length, record['c'] + length)
            all_sweeps |= _rect(lane, lo, span, hi - lo)
            _require(final_union.isdisjoint(final), "constructed final overlap")
            final_union |= final
            destinations.append((final, record['color']))
            cursor += length if target == 'left' else -length
    _paint(b, all_sweeps - final_union, ink)
    for cells, color in destinations:
        _paint(b, cells, color)
    # Rotate all visible semantics together; mirror is also a valid symmetry.
    a, b = np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()
    if rng.random() < .5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    return dict(input=a, output=b)


def _valid_int(value, low, high):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def _counts(num_train, num_test):
    if not _valid_int(num_train, 2, 10) or not _valid_int(num_test, 1, 4):
        raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _fingerprint(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    ink, tip = map(int, rng.choice(10, 2, replace=False))
    first_rotation = int(rng.integers(4))
    # Different axis orientations are guaranteed even if all other samples align.
    rotations = [first_rotation, (first_rotation + 1) % 4]
    plans = [('single', rotations[0]), ('counterflow', rotations[1])]
    plans += [('single' if rng.random() < .4 else 'ordinary', int(rng.integers(4)))
              for _ in range(nt - 2)]
    plans = [plans[int(j)] for j in rng.permutation(nt)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, sequence in (('train', plans), ('test', [('query', int(rng.integers(4))) for _ in range(nq)])):
        for mode, rotation in sequence:
            for _ in range(8):
                pair = _construct_pair(rng, ink, tip, mode, rotation)
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    break
            _require(fp not in seen, 'bounded uniqueness retries exhausted')
            seen.add(fp)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual_differs(info, output, **kwargs):
    # Ablations can overwrite colliding final footprints, so disable only the
    # collision assertions. Always compare actual arrays; exceptions are bugs.
    return not np.array_equal(_prediction(info, strict=False, **kwargs)[0], output)


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, inks, tips, train_directions = set(), set(), set(), set()
    witness_a = witness_b = witness_pack = False
    single = counterflow = central_stop = False
    length_orders, color_orders, training_lengths = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'shape changed')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _analyze(a)
            predicted, packing = _prediction(info)
            _require(np.array_equal(predicted, b), 'output does not follow cargo rule')
            _require(not np.array_equal(a, b), 'identity example')
            _require(all(b[p] == a[p] for p in info['fixed']), 'fixed geometry changed')
            inks.add(info['ink'])
            tips.add(info['tip'])
            tokens, forks = info['tokens'], info['forks']
            _require(2 <= len(info['cards']) <= 3, 'expected two or three cargo colors')
            _require(1 <= len(forks) <= 2, 'sampler canal receiver count')
            # Require separation based on actual input components, including keys.
            objects = [t['cells'] for t in tokens] + [f['cells'] for f in forks] + [c['cells'] for c in info['cards']]
            occupied = set()
            for cells in objects:
                _require(cells.isdisjoint(_expand(occupied)), 'input objects touch')
                occupied |= cells
            moves = packing['moves']
            _require(moves and packing['rejected'], 'missing admission or rejection')
            _require(max(map(len, packing['groups'])) >= 2, 'no queue packing')
            _require(all(m['actual_distance'] > 0 for m in moves), 'sampled cargo did not move')
            # Per-color output area equals admitted area; no rejected cargo survives.
            expected_area = Counter()
            for m in moves:
                t = tokens[m['ti']]
                expected_area[t['color']] += len(t['cells'])
                _require(len(m['final']) == len(t['cells']), 'cargo resized')
                _require(_is_rectangle(m['final']), 'cargo shape changed')
            for card in info['cards']:
                color = card['carrier']
                outside_cards = sum(int(b[p] == color) for p in np.ndindex(b.shape)
                                    if not any(p in c['cells'] for c in info['cards']))
                _require(outside_cards == expected_area[color], 'admitted color/area not conserved')
            # Trail checks exclude cards, whose ordinary strokes already use ink.
            outside = set(map(tuple, np.argwhere(b == info['ink']))) - set().union(*(c['cells'] for c in info['cards']))
            _require(outside == packing['swept'] - packing['final'], 'incorrect swept ink')
            # Explicit centered width mismatch, commanded toward an existing dock.
            width_decoys = [ti for ti in packing['rejected']
                            if any(_candidate(info, tokens[ti], f, match_span=False) is not None for f in forks)]
            _require(width_decoys, 'no width-matching counterexample')
            if split == 'train':
                train_directions.update(c['direction'] for c in info['cards'])
                training_lengths.update(m['length'] for m in moves)
                central_stop |= any(f['lengths'][1] not in (f['lengths'][0], f['lengths'][2]) for f in forks)
                for group in packing['groups']:
                    for near, far in zip(group, group[1:]):
                        length_orders.add(int(np.sign(far['length'] - near['length'])))
                        color_orders.add(int(np.sign(tokens[far['ti']]['color'] - tokens[near['ti']]['color'])))
                witness_a |= _counterfactual_differs(info, b, use_arrows=False)
                witness_b |= _counterfactual_differs(info, b, match_span=False)
                witness_pack |= _counterfactual_differs(info, b, pack=False)
                if len(forks) == 1:
                    # A rejected exact-fit cargo becomes admissible with direction removed.
                    directional_reject = any(_candidate(info, tokens[ti], forks[0], use_arrows=False) is not None
                                             for ti in packing['rejected'])
                    mixed = len({tokens[m['ti']]['color'] for m in moves}) >= 2
                    single |= directional_reject and mixed
                if len(forks) == 2:
                    for m in moves:
                        t = tokens[m['ti']]
                        others = [_candidate(info, t, f, use_arrows=False) for j, f in enumerate(forks) if j != m['fi']]
                        if any(c is not None and c['distance'] < m['distance'] for c in others):
                            counterflow = True
                _require(all(m['length'] <= 2 for m in moves), 'training length outside documented range')
            else:
                _require(len(forks) == 2 and all(len(g) >= 2 for g in packing['groups']), 'query multiplicity shift')
                _require(all(m['length'] <= 3 for m in moves), 'query length outside range')
    _require(len(inks) == len(tips) == 1, 'puzzle ink/tip conventions changed')
    _require(single and counterflow, 'missing guaranteed demonstrations')
    _require(central_stop and training_lengths == {1, 2}, 'missing stop/rigid-length evidence')
    _require({-1, 1} <= length_orders and {-1, 1} <= color_orders, 'queue order shortcuts not distinguished')
    _require(len(train_directions) >= 2, 'missing direction variation')
    _require(witness_a and witness_b and witness_pack, 'parent/packing ablation has no training witness')


def self_test():
    """Bounded streaming checks over 42 seeds; returns compact measurements."""
    shapes, normalized, hashes, cards, geometries = set(), set(), set(), set(), set()
    directions, inks, spans, cargo_lengths, widths, spoke_lengths = set(), set(), set(), set(), set(), set()
    counts, mixed_orders, overlaps = Counter(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_hash(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                _, packing = _prediction(info)
                a = info['grid']
                shapes.add(a.shape)
                role_ids = {}
                canonical = bytes(role_ids.setdefault(int(c), len(role_ids)) for c in a.ravel())
                normalized.add(hashlib.sha256(bytes(a.shape) + canonical).hexdigest())
                inks.add(info['ink'])
                directions.update(c['direction'] for c in info['cards'])
                cards.update(c['shape'] for c in info['cards'])
                counts[(split, len(info['forks']), len(info['cards']))] += 1
                for fork in info['forks']:
                    spans.add(fork['span'])
                    widths.add(fork['width'])
                    spoke_lengths.update(fork['lengths'])
                    geometries.add((fork['width'], fork['runs']))
                for group in packing['groups']:
                    if len(group) > 1:
                        mixed_orders.add(tuple(info['tokens'][m['ti']]['color'] for m in group))
                cargo_lengths.update(m['length'] for m in packing['moves'])
                routes = [m['route'] for m in packing['moves']]
                overlaps += int(any(not x.isdisjoint(y) for j, x in enumerate(routes) for y in routes[j + 1:]))
        # Input-only solver is equivariant, with no reliance on canonical layout.
        if seed < 4:
            pair = puzzle['test'][0]
            for k in range(4):
                for mirror in (False, True):
                    a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                    if mirror:
                        a, b = np.fliplr(a), np.fliplr(b)
                    _require(np.array_equal(solve(a), b), 'rotation/reflection failure')
    _require(len(hashes) == 42, 'seed diversity')
    _require(len(shapes) >= 30 and len(normalized) >= 200, 'structural diversity')
    _require(directions == set(DIRS) and len(inks) >= 8, 'direction/palette diversity')
    _require(spans == {1, 2, 3} and cargo_lengths == {1, 2, 3}, 'cargo scale diversity')
    _require(len(widths) >= 7 and len(geometries) >= 100 and len(cards) >= 6, 'geometry diversity')
    _require(spoke_lengths == {1, 2, 3, 4} and len(mixed_orders) >= 40 and overlaps >= 42,
             'queue and route diversity')
    # All 30 legal explicit combinations; one puzzle retained at a time.
    legal_counts = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1701 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'count contract')
            legal_counts += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _require(_puzzle_hash(generate_puzzle(12345, 6, 3)) == _puzzle_hash(generate_puzzle(12345, 6, 3)),
             'seed nondeterminism')
    bad = generate_puzzle(99, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('semantic corruption accepted')
    try:
        _parse_receiver(_rect(0, 0, 7, 1) | _rect(0, 1, 1, 3) | _rect(6, 1, 1, 3))
    except AssertionError:
        pass
    else:
        raise AssertionError('missing central prong accepted')
    return dict(seeds=42, distinct_puzzles=len(hashes), shapes=len(shapes),
                normalized_structures=len(normalized), card_shapes=len(cards),
                receiver_geometries=len(geometries), fork_widths=sorted(map(int, widths)),
                cargo_lengths=sorted(map(int, cargo_lengths)), ink_colors=sorted(inks),
                mixed_queue_orders=len(mixed_orders), pairs_with_overlapping_routes=overlaps,
                legal_count_combinations=legal_counts,
                semantic_verification='authored generator checks; no independent solver')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
