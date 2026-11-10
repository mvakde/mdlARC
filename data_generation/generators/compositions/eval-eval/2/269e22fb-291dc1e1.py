"""Bookended ribbons: exact crop addresses define spans in a keyed chart.

All solving information is visible in one input. No parent code or stored
reference bitmap is used. Only Python's standard library and NumPy are needed.
"""
from __future__ import annotations

import copy
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _d4(a, i):
    return np.rot90(np.fliplr(a) if i >= 4 else a, i % 4).copy()


def _crop_class(a):
    """Dihedral/color orbit of a full-thickness binary crop.

    For nonsquares, transpose first if needed, then the four axis reflections
    cover all transformations having height two. Squares also need transpose.
    Taking complements permits either bijection of the two color roles.
    """
    a = np.asarray(a, dtype=np.int8)
    if a.shape[0] != 2:
        if a.shape[1] != 2:
            raise ValueError("Fragments must contain exactly two thickness cells")
        a = a.T
    if a.shape[1] < 2:
        raise ValueError("Fragments must be at least two cells long")
    variants = [a, a[::-1], a[:, ::-1], a[::-1, ::-1]]
    if a.shape[1] == 2:
        variants += [v.T for v in variants[:]]
    return a.shape[1], min(v.tobytes() for b in variants for v in (b, 1-b))


def _address_index(strips):
    """Index EVERY proper interval, not merely the sampled crop bank."""
    index = {}
    colors = sorted(set(int(c) for b in strips for c in b.flat))
    if len(colors) != 2:
        raise ValueError("The reference chart must use exactly two payload colors")
    for lane, strip in enumerate(strips):
        binary = (strip == colors[1]).astype(np.int8)
        length = strip.shape[1]
        for width in range(2, length):
            for start in range(length-width+1):
                patch = binary[:, start:start+width]
                if patch.min() == patch.max():
                    continue
                index.setdefault(_crop_class(patch), set()).add((lane, start, start+width))
    return index


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    cells.append(point)
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)+1
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)+1
        if len(cells) != (r1-r0)*(c1-c0):
            raise ValueError("Every ribbon and loose fragment must be a full rectangle")
        result.append((int(r0), int(r1), int(c0), int(c1)))
    return sorted(result)


def _find_chart(a, bg):
    candidates = []
    for marker in map(int, np.unique(a)):
        if marker == bg:
            continue
        rr, cc = np.nonzero(a == marker)
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()+1), int(cc.min()), int(cc.max()+1)
        if min(r1-r0, c1-c0) < 6:
            continue
        physical = a[r0:r1, c0:c1]
        for rotation in range(4):
            chart = np.rot90(physical, rotation).copy()
            H, W = chart.shape
            for bottom in (False, True):
                fast_row = H-1 if bottom else 0
                other_row = 0 if bottom else H-1
                origin = int(chart[fast_row, 0])
                if origin in (bg, marker) or np.count_nonzero(a == origin) != 1:
                    continue
                expected = np.where(np.arange(1, W-1) % 2, marker, bg)
                if (np.all(chart[other_row] == marker)
                        and np.all(chart[1:-1, 0] == marker)
                        and np.all(chart[1:-1, -1] == marker)
                        and chart[fast_row, -1] == marker
                        and np.array_equal(chart[fast_row, 1:-1], expected)
                        and not np.any(chart[1:-1, 1:-1] == marker)):
                    candidates.append((chart, (r0, r1, c0, c1), rotation, bottom, marker, origin))
    if len(candidates) != 1:
        raise ValueError("Expected one chart with a dotted fast edge and a unique corner bead")
    return candidates[0]


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("The outside border must be clear background")
    chart, box, rotation, bottom, marker, origin = _find_chart(a, bg)
    interior = chart[1:-1, 1:-1]
    if not (np.all(interior[0] == bg) and np.all(interior[-1] == bg)
            and np.all(interior[:, 0] == bg) and np.all(interior[:, -1] == bg)):
        raise ValueError("Reference ribbons need a background margin inside their frame")
    boxes = _components(interior != bg)
    if not boxes or any(r1-r0 != 2 or c1-c0 < 3 for r0, r1, c0, c1 in boxes):
        raise ValueError("Reference lanes contain one two-cell-thick ribbon each")
    if any(b[0]-a0[1] < 1 for a0, b in zip(boxes, boxes[1:])):
        raise ValueError("Reference lane bands must be separated")
    boxes.sort(key=lambda b: b[0], reverse=bottom)
    strips = [interior[r0:r1, c0:c1].copy() for r0, r1, c0, c1 in boxes]
    colors = set(int(c) for b in strips for c in b.flat)
    if len(colors) != 2 or colors & {bg, marker, origin}:
        raise ValueError("Reference roles and marker roles must be distinct")
    if any(len(np.unique(b)) != 2 for b in strips):
        raise ValueError("Each reference ribbon must show both binary roles")
    index = _address_index(strips)
    mask = a != bg
    r0, r1, c0, c1 = box
    mask[r0:r1, c0:c1] = False
    ticket_boxes = _components(mask)
    if not ticket_boxes:
        raise ValueError("At least one loose fragment is required")
    tickets = []
    for tbox in ticket_boxes:
        u0, u1, v0, v1 = tbox
        ticket = a[u0:u1, v0:v1]
        palette = tuple(map(int, np.unique(ticket)))
        if len(palette) != 2 or set(palette) & {bg, marker, origin}:
            raise ValueError("Loose fragments must use two non-marker, non-background colors")
        # Segmentation and chart identity must not depend on an output label.
        neighborhood = a[u0-1:u1+1, v0-1:v1+1].copy()
        neighborhood[1:-1, 1:-1] = bg
        if np.any(neighborhood != bg):
            raise ValueError("Loose fragments must be surrounded by background")
        binary = (ticket == palette[1]).astype(np.int8)
        addresses = index.get(_crop_class(binary), set())
        if len(addresses) != 1:
            raise ValueError("A fragment must identify exactly one reference interval")
        lane, start, end = next(iter(addresses))
        tickets.append(dict(lane=lane, start=start, end=end, box=tbox, palette=palette,
                            shape=ticket.shape, binary=binary))
    spans, bounds, lanes = [], [], []
    for lane, strip in enumerate(strips):
        members = [t for t in tickets if t['lane'] == lane]
        if members:
            start, end = min(t['start'] for t in members), max(t['end'] for t in members)
            spans.append(strip[:, start:end].copy())
            bounds.append((start, end))
            lanes.append(lane)
    if len({b.shape[1] % 2 for b in spans}) != 1:
        raise ValueError("All selected spans must share length parity for exact centering")
    return dict(background=bg, marker=marker, origin=origin, chart=chart, chart_box=box,
                rotation=rotation, bottom=bottom, strips=strips, tickets=tickets,
                spans=spans, bounds=bounds, lanes=lanes, colors=colors)


def _stack(blocks, background):
    width = max(b.shape[1] for b in blocks)
    if 2*len(blocks) > 30 or width > 30 or len({b.shape[1] % 2 for b in blocks}) != 1:
        raise ValueError("The centered output must fit the ARC bounds with exact padding")
    out = np.full((2*len(blocks), width), background, dtype=np.int8)
    for i, block in enumerate(blocks):
        pad = (width-block.shape[1])//2
        out[2*i:2*i+2, pad:pad+block.shape[1]] = block
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recognize addresses and assemble the answer using this input alone."""
    d = _read(input_grid)
    return _stack(d['spans'], d['background'])


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in (a[::-1], a[:, ::-1], a[::-1, ::-1]))


def _runs(rng, length):
    row = np.empty(length, dtype=np.int8)
    pos, value = 0, int(rng.integers(2))
    while pos < length:
        end = min(length, pos+int(rng.integers(1, 5)))
        row[pos:end] = value
        pos, value = end, 1-value
    return row


def _ribbon(rng, length):
    mode = int(rng.integers(4))
    first = _runs(rng, length)
    if mode == 0:
        second = _runs(rng, length)
    elif mode == 1:
        second = np.roll(first, int(rng.integers(1, 4)))
        if rng.integers(2):
            second = 1-second
    elif mode == 2:
        motif = rng.integers(0, 2, size=(2, int(rng.integers(2, 6))), dtype=np.int8)
        result = np.tile(motif, (1, length))[:, :length]
        start = int(rng.integers(length-2))
        result[int(rng.integers(2)), start:start+int(rng.integers(1, 4))] ^= 1
        return result
    else:
        second = np.full(length, int(rng.integers(2)), dtype=np.int8)
        for _ in range(int(rng.integers(2, 5))):
            start = int(rng.integers(length-1))
            second[start:min(length, start+int(rng.integers(1, 5)))] ^= 1
    return np.vstack((first, second)).astype(np.int8)


def _banks(strips, parity):
    index = _address_index(strips)
    banks = [[] for _ in strips]
    for addresses in index.values():
        if len(addresses) != 1:
            continue
        lane, start, end = next(iter(addresses))
        width = end-start
        # Even starts are a construction device, not an address disambiguator.
        if start % 2 == 0 and width % 2 == parity and 3 <= width <= 8:
            banks[lane].append((start, end))
    return [sorted(bank) for bank in banks]


def _bookends(strips, banks, mode):
    options = []
    for lane, bank in enumerate(banks):
        for a in bank:
            for b in bank:
                if a[0] >= b[0]:
                    continue
                separated = b[0] > a[1]
                overlapping = b[0] < a[1]
                end = max(a[1], b[1])
                block = strips[lane][:, a[0]:end]
                if ((separated if mode == 'bridge' else overlapping)
                        and end-a[0] < strips[lane].shape[1] and _asymmetric(block)
                        and _asymmetric(strips[lane][:, a[0]:a[1]])
                        and _asymmetric(strips[lane][:, b[0]:b[1]])):
                    options.append((lane, a, b))
    return options


def _make_reference(rng, parity, mode, is_test=False):
    n = int(rng.integers(3, 5) if mode else rng.integers(2 if is_test else 1, 6))
    for _ in range(100):
        lengths = rng.integers(14 if mode else 10, 21, size=n)
        strips = [_ribbon(rng, int(length)) for length in lengths]
        if any(len(np.unique(b)) != 2 for b in strips):
            continue
        banks = _banks(strips, parity)
        if any(not bank for bank in banks):
            continue
        if mode:
            options = _bookends(strips, banks, mode)
            options = [o for o in options if any(
                lane != o[0] and _asymmetric(strips[lane][:, s:e])
                and e-s != max(o[1][1], o[2][1])-o[1][0]
                for lane, bank in enumerate(banks) for s, e in bank)]
            if not options:
                continue
            lane, a, b = options[int(rng.integers(len(options)))]
            queries = [(lane, *a), (lane, *b)]
            others = [(j, s, e) for j, bank in enumerate(banks) for s, e in bank
                      if j != lane and _asymmetric(strips[j][:, s:e])
                      and e-s != max(a[1], b[1])-a[0]]
            queries.append(others[int(rng.integers(len(others)))])
            if mode == 'overlap':
                queries.append(queries[int(rng.integers(2))])
        else:
            if is_test:
                # Queries combine established factors: several hidden lanes,
                # each with multiple addresses, with shorter identifying crops.
                banks = [[a for a in bank if a[1]-a[0] <= 6] or bank for bank in banks]
                selected_count = int(rng.integers(2, min(4, n)+1))
            else:
                selected_count = int(rng.integers(1, n+1))
            selected = list(map(int, rng.choice(n, selected_count, replace=False)))
            queries = []
            for j, lane in enumerate(selected):
                count = int(rng.integers(2 if is_test else 1, 4))
                if is_test:
                    count = min(count, 9-len(queries)-2*(len(selected)-j-1))
                for _ in range(count):
                    s, e = banks[lane][int(rng.integers(len(banks[lane])))]
                    queries.append((lane, s, e))
            if len(queries) > 9:
                queries = queries[:9]
            if len(queries) < 9 and rng.random() < .25:
                queries.append(queries[int(rng.integers(len(queries)))])
        return strips, queries
    raise RuntimeError("Bounded reference construction exhausted")


def _draw_chart(rng, strips, roles, bottom):
    bg, marker, origin, zero, one = roles
    n = len(strips)
    slow_gaps = rng.integers(1, 3 if n < 5 else 2, size=n-1).tolist()
    near, far = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    H = 2+near+far+2*n+sum(slow_gaps)
    starts = rng.integers(1, 3, size=n).tolist()
    W = 2+max(u+b.shape[1] for u, b in zip(starts, strips))+int(rng.integers(1, 3))
    chart = np.full((H, W), bg, dtype=np.int8)
    chart[0] = chart[-1] = marker
    chart[:, 0] = chart[:, -1] = marker
    fast_row = H-1 if bottom else 0
    chart[fast_row, 0] = origin
    chart[fast_row, 1:-1] = np.where(np.arange(1, W-1) % 2, marker, bg)
    cursor = 1+near
    for i, (strip, start) in enumerate(zip(strips, starts)):
        row = H-cursor-2 if bottom else cursor
        chart[row:row+2, 1+start:1+start+strip.shape[1]] = np.where(strip, one, zero)
        cursor += 2+(slow_gaps[i] if i < n-1 else 0)
    return chart


def _scatter(rng, chart, tickets, bg):
    """Place objects with full background moats; no fixed panel layout."""
    h, w = chart.shape
    for _ in range(40):
        H = int(rng.integers(max(24, h+7), 31))
        W = int(rng.integers(max(24, w+2), 31))
        if rng.random() < .65:
            r = int(rng.choice([1, H-h-1]))
            c = int(rng.choice([1, W-w-1]))
        else:
            r, c = int(rng.integers(1, H-h)), int(rng.integers(1, W-w))
        a = np.full((H, W), bg, dtype=np.int8)
        occupied = np.zeros((H, W), dtype=bool)
        a[r:r+h, c:c+w] = chart
        occupied[r:r+h, c:c+w] = True
        order = sorted(range(len(tickets)), key=lambda i: (-max(tickets[i].shape), -tickets[i].size))
        success = True
        for i in order:
            ticket = tickets[i]
            th, tw = ticket.shape
            positions = [(u, v) for u in range(1, H-th) for v in range(1, W-tw)]
            found = False
            for j in rng.permutation(len(positions)):
                u, v = positions[int(j)]
                if not np.any(occupied[u-1:u+th+1, v-1:v+tw+1]):
                    a[u:u+th, v:v+tw] = ticket
                    occupied[u:u+th, v:v+tw] = True
                    found = True
                    break
            if not found:
                success = False
                break
        if success:
            return a
    raise RuntimeError("Bounded loose-fragment placement exhausted")


def _make_pair(rng, configuration, mode, roles, is_test=False):
    q, bottom = configuration
    parity = int(rng.integers(2))
    strips, queries = _make_reference(rng, parity, mode, is_test)
    bg, marker, origin, zero, one = roles
    chart = _draw_chart(rng, strips, roles, bottom)
    pool = [c for c in range(10) if c not in (bg, marker, origin)]
    tickets = []
    rng.shuffle(queries)
    for i, (lane, start, end) in enumerate(queries):
        patch = strips[lane][:, start:end]
        if mode and i == 0:
            colors = (one, zero)
        elif mode and i == 1:
            palettes = [(a, b) for a in pool for b in pool
                        if a != b and {a, b} != {zero, one}]
            colors = palettes[int(rng.integers(len(palettes)))]
        else:
            colors = tuple(map(int, rng.choice(pool, 2, replace=False)))
        orientation = (1 if i == 0 else 4 if i == 1 else int(rng.integers(8))) if mode else int(rng.integers(8))
        tickets.append(_d4(np.where(patch, colors[1], colors[0]).astype(np.int8), orientation))
    inp = np.rot90(_scatter(rng, chart, tickets, bg), q).copy()
    # Construct the label directly from the sampled reference addresses, not
    # by calling solve. Rotation of the whole input preserves these blocks.
    selected = []
    for lane, strip in enumerate(strips):
        addresses = [(s, e) for j, s, e in queries if j == lane]
        if addresses:
            start, end = min(s for s, _ in addresses), max(e for _, e in addresses)
            selected.append(np.where(strip[:, start:end], one, zero).astype(np.int8))
    return dict(input=inp, output=_stack(selected, bg))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first = (int(rng.integers(4)), bool(rng.integers(2)))
    second = ((first[0]+1) % 4, not first[1])
    configurations = [(q, b) for q in range(4) for b in (False, True)]
    train_configs = [first, second]+[configurations[int(rng.integers(8))] for _ in range(nt-2)]
    held = [c for c in configurations if c not in train_configs]
    test_pool = held if held and rng.random() < .7 else train_configs
    test_configs = [test_pool[int(rng.integers(len(test_pool)))] for _ in range(nq)]
    stable_palette = bool(rng.random() < .25)
    base_roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, configs in (('train', train_configs), ('test', test_configs)):
        for i, config in enumerate(configs):
            mode = ('bridge', 'overlap')[i] if split == 'train' and i < 2 else None
            for _ in range(30):
                roles = base_roles if stable_palette or rng.random() < .3 else tuple(map(int, rng.choice(10, 5, replace=False)))
                try:
                    pair = _make_pair(rng, config, mode, roles, split == 'test')
                except RuntimeError:
                    continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded pair construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    grouped = [[t for t in d['tickets'] if t['lane'] == lane] for lane in d['lanes']]
    bridge = False
    for group, (start, end) in zip(grouped, d['bounds']):
        covered = np.zeros(end-start, dtype=bool)
        for t in group:
            covered[t['start']-start:t['end']-start] = True
        bridge |= not bool(np.all(covered))
    overlap = any(a is not b and max(a['start'], b['start']) < min(a['end'], b['end'])
                  and (a['start'], a['end']) != (b['start'], b['end'])
                  for group in grouped for a in group for b in group)
    duplicate = len({(t['lane'], t['start'], t['end']) for t in d['tickets']}) < len(d['tickets'])
    return dict(configuration=(d['rotation'], d['bottom']), ribbons=len(d['strips']),
                selected=len(d['spans']), fragments=len(d['tickets']),
                lengths=tuple(b.shape[1] for b in d['spans']), bridge=bridge,
                overlap=overlap, duplicate=duplicate, unused=len(d['strips']) > len(d['spans']),
                proper=any(e-s < d['strips'][lane].shape[1] for lane, (s, e) in zip(d['lanes'], d['bounds'])),
                asymmetric=any(_asymmetric(b) for b in d['spans']),
                novel_palette=any(set(t['palette']) != d['colors'] for t in d['tickets']))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("Puzzle must contain exactly train and test")
    if any(not isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("Splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("Every pair must contain exactly input and output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if _key(inp) in seen:
                raise ValueError("Inputs must be distinct within a puzzle")
            seen.add(_key(inp))
            d = _read(inp)
            expected = _stack(d['spans'], d['background'])
            if not np.array_equal(out, expected):
                raise ValueError("Output disagrees with the crop-address interval-hull rule")
            if not set(map(int, out.flat)) <= d['colors'] | {d['background']}:
                raise ValueError("Only reference payload and padding colors may be output")
            # Cell-level span invariant: addressed source windows are contained
            # with their exact source contents; no cells outside the hull enter.
            for lane, (s, e), span in zip(d['lanes'], d['bounds'], d['spans']):
                assert np.array_equal(span, d['strips'][lane][:, s:e])
                for t in d['tickets']:
                    if t['lane'] == lane:
                        assert s <= t['start'] < t['end'] <= e
            if split == 'train':
                training.append((d, _features(d)))
    features = [f for _, f in training]
    required = ('bridge', 'overlap', 'duplicate', 'unused', 'proper', 'asymmetric', 'novel_palette')
    if any(not any(f[name] for f in features) for name in required):
        raise ValueError("Training lacks a required rule witness")
    if len({d['rotation'] % 2 for d, _ in training}) != 2 or len({d['bottom'] for d, _ in training}) != 2:
        raise ValueError("Training must expose both axes and both lane directions")
    if not any(len(set(f['lengths'])) > 1 for f in features):
        raise ValueError("Training must demonstrate unequal-span centering")
    # Explicit parent-ablation checks. Ignoring A's interval grouping changes
    # block count, while ignoring B's order/rotation changes payload placement.
    if not any(len(d['tickets']) > len(d['spans']) for d, _ in training):
        raise ValueError("Ignoring crop-address grouping must change a demonstration")
    for alternative in ('reverse_lanes', 'reflect_thickness', 'reflect_fast'):
        changed = False
        for d, _ in training:
            spans = d['spans']
            alt = (spans[::-1] if alternative == 'reverse_lanes' else
                   [b[::-1] for b in spans] if alternative == 'reflect_thickness' else
                   [b[:, ::-1] for b in spans])
            changed |= not np.array_equal(_stack(alt, d['background']), _stack(spans, d['background']))
        if not changed:
            raise ValueError("Training does not distinguish a key/rotation shortcut")


def self_test():
    """Bounded 42-seed diversity, determinism, ablation, and corruption checks."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    signature = lambda p: tuple(_key(_grid(pair[role])) for split in ('train', 'test')
                                for pair in p[split] for role in ('input', 'output'))
    assert len({signature(p) for p in corpus}) == 42
    assert signature(corpus[0]) == signature(generate_puzzle(0))
    references, configurations, shapes, lengths, populations, fragment_shapes, palettes = set(), set(), set(), set(), set(), set(), set()
    bridge = overlap = duplicates = singleton = familiar = unfamiliar = 0
    for p in corpus:
        configs = {_features(_read(pair['input']))['configuration'] for pair in p['train']}
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _read(pair['input'])
                f = _features(d)
                assert np.array_equal(solve(pair['input']), pair['output'])
                # A proper whole-scene rotation must leave the normalized answer unchanged.
                for q in range(1, 4):
                    assert np.array_equal(solve(np.rot90(pair['input'], q)), pair['output'])
                references.add(tuple(_crop_class((b == max(d['colors'])).astype(np.int8)) for b in d['strips']))
                configurations.add(f['configuration'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                lengths.update(f['lengths'])
                populations.add((f['ribbons'], f['selected'], f['fragments']))
                fragment_shapes.update(t['shape'] for t in d['tickets'])
                palettes.add(tuple(sorted(d['colors'])))
                bridge += f['bridge']; overlap += f['overlap']; duplicates += f['duplicate']
                singleton += f['fragments'] == 1
                if split == 'test':
                    familiar += f['configuration'] in configs
                    unfamiliar += f['configuration'] not in configs
    assert len(references) >= 180, "Reference content must change, not only palettes"
    assert len(shapes) >= 80 and configurations == {(q, b) for q in range(4) for b in (False, True)}
    assert len(lengths) >= 12 and len(populations) >= 25 and len(fragment_shapes) >= 10 and len(palettes) >= 25
    assert bridge >= 42 and overlap >= 42 and duplicates >= 42 and singleton >= 1
    assert familiar and unfamiliar
    # All legal requested counts, including the two-demonstration boundary.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(800+10*nt+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = copy.deepcopy(corpus[0])
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("A corrupted output was accepted")
    # Removing a repeated address is idempotent; removing one disjoint
    # bookend changes the hull. These exercise the input-only rule directly.
    witnessed_duplicate = witnessed_bookend = False
    for pair in corpus[0]['train']:
        d = _read(pair['input'])
        addresses = [(t['lane'], t['start'], t['end']) for t in d['tickets']]
        for i, t in enumerate(d['tickets']):
            if addresses.count(addresses[i]) > 1 and not witnessed_duplicate:
                changed = pair['input'].copy()
                r0, r1, c0, c1 = t['box']
                changed[r0:r1, c0:c1] = d['background']
                assert np.array_equal(solve(changed), pair['output'])
                witnessed_duplicate = True
        if _features(d)['bridge'] and not witnessed_bookend:
            for t in d['tickets']:
                group = [x for x in d['tickets'] if x['lane'] == t['lane']]
                if len(group) == 2 and max(x['start'] for x in group) > min(x['end'] for x in group):
                    changed = pair['input'].copy()
                    r0, r1, c0, c1 = t['box']
                    changed[r0:r1, c0:c1] = d['background']
                    assert not np.array_equal(solve(changed), pair['output'])
                    witnessed_bookend = True
                    break
    assert witnessed_duplicate and witnessed_bookend
    # Ambiguity is about addresses, not arbitrary latent orientation uniqueness.
    strip = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.int8)
    index = _address_index([strip, strip.copy()])
    assert len(index[_crop_class(strip[:, :3])]) > 1
    rng = np.random.default_rng(42)
    ambiguous_chart = _draw_chart(rng, [strip, strip.copy()], (8, 1, 2, 3, 4), False)
    ambiguous_ticket = np.where(strip[:, :3], 4, 3).astype(np.int8)
    ambiguous_input = _scatter(rng, ambiguous_chart, [ambiguous_ticket], 8)
    try:
        solve(ambiguous_input)
    except ValueError:
        pass
    else:
        raise AssertionError("The solver accepted conflicting reference addresses")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, rotation and corruption checks')
