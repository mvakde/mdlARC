"""Broken-arrow jigsaw: assembly completes simultaneous color instructions.

Only NumPy and the standard library are used. All prediction state comes from
the query pixels. Constructor labels are checked against an exhaustive decoder.
"""
from collections import Counter
from itertools import permutations
from math import isqrt

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CYCLE = (0, 1, 3, 2)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    _need(grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape), "invalid grid dimensions")
    _need(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((grid >= 0) & (grid <= 9))), "invalid color")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    pending.append(q)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _key(array):
    return array.shape, array.astype(np.int8).tobytes()


def _parse(grid):
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    background = int(np.bincount(border, minlength=10).argmax())
    _need(np.count_nonzero(border == background) > len(border) // 2, "ambiguous background")
    components = _components(grid != background)
    _need(len(components) == 4, "four separated pieces required")
    ids = np.full(grid.shape, -1, dtype=int)
    for i, cells in enumerate(components):
        ids[cells[:, 0], cells[:, 1]] = i
    for i, cells in enumerate(components):
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                        _need(ids[rr, cc] in (-1, i), "pieces need a background-cell gap")
    keys = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            patch = grid[r:r+2, c:c+2]
            if len(np.unique(patch)) == 4 and background not in patch:
                owner = int(ids[r, c])
                if owner >= 0 and np.all(ids[r:r+2, c:c+2] == owner):
                    keys.append((owner, r, c, patch.copy()))
    _need(len(keys) == 1, "one unambiguous four-color key required")
    anchor, kr, kc, legend = keys[0]
    order = [anchor] + [i for i in range(4) if i != anchor]
    pieces, carriers = [], []
    for i in order:
        cells = components[i]
        low, high = cells.min(axis=0), cells.max(axis=0) + 1
        array = np.full(tuple(high - low), -1, dtype=np.int8)
        local = cells - low
        array[local[:, 0], local[:, 1]] = grid[cells[:, 0], cells[:, 1]]
        values = [int(grid[r, c]) for r, c in cells
                  if not (i == anchor and kr <= r < kr+2 and kc <= c < kc+2)]
        counts = Counter(values).most_common()
        _need(counts and (len(counts) == 1 or counts[0][1] > counts[1][1]), "carrier must be uniquely modal")
        carrier = counts[0][0]
        _need(carrier not in legend, "carriers and legend must be distinct")
        pieces.append(array)
        carriers.append(carrier)
    key_location = kr - int(components[anchor][:, 0].min()), kc - int(components[anchor][:, 1].min())
    extra_colors = set()
    for i, array in enumerate(pieces):
        extra = (array >= 0) & (array != carriers[i])
        if i == 0:
            r, c = key_location
            extra[r:r+2, c:c+2] = False
        extra_colors.update(map(int, array[extra]))
    possible_inks = extra_colors - set(map(int, legend.flat)) - set(carriers) - {background}
    _need(possible_inks, "missing arrow ink")
    area = sum(np.count_nonzero(p >= 0) for p in pieces)
    side = isqrt(area)
    _need(side * side == area and side <= 30, "piece area must be square")
    return dict(pieces=pieces, carriers=carriers, legend=legend, key_location=key_location,
                background=background, inks=sorted(possible_inks), side=side)


def _placements(piece, side, corner, anchor=False):
    results, seen = [], set()
    for rotation in ((0,) if anchor else range(4)):
        array = np.rot90(piece, rotation).copy()
        h, w = array.shape
        if h > side or w > side:
            continue
        r = 0 if corner < 2 else side - h
        c = 0 if corner % 2 == 0 else side - w
        rows, cols = np.nonzero(array >= 0)
        rr, cc = rows + r, cols + c
        mask = sum(1 << (int(y) * side + int(x)) for y, x in zip(rr, cc))
        corners = (0, side-1, side*(side-1), side*side-1)
        hits = [bool(mask & (1 << x)) for x in corners]
        if sum(hits) != 1 or not hits[corner]:
            continue
        signature = mask, _key(array)
        if signature not in seen:
            seen.add(signature)
            results.append(dict(mask=mask, rows=rr, cols=cc, values=array[rows, cols], rotation=rotation))
    return results


def _read_arrows(square, owner, carriers, key_location, ink, legend):
    """Exact-cover all noncarrier, nonkey cells by seam-crossing T motifs."""
    side = len(square)
    marked = square != np.asarray(carriers)[owner]
    r, c = key_location
    marked[r:r+2, c:c+2] = False
    points = set(map(tuple, np.argwhere(marked)))
    if not points or len(points) % 4:
        return []
    candidates = []
    for center in sorted(points):
        r, c = center
        source = int(owner[center])
        for dr, dc in DIRECTIONS:
            pr, pc = -dc, dr
            arms = ((r+pr, c+pc), (r-pr, c-pc))
            tip = r+dr, c+dc
            write = {center, *arms, tip}
            if len(write) != 4 or not write <= points:
                continue
            if not all(int(square[q]) == ink for q in (*arms, tip)):
                continue
            if not all(int(owner[q]) == source for q in arms):
                continue
            target = int(owner[tip])
            if target == source:
                continue
            explicit = int(square[center]) != ink
            payload = int(square[center]) if explicit else int(legend.flat[source])
            if explicit and payload in carriers:
                continue
            candidates.append(dict(center=center, direction=(dr, dc), source=source, target=target,
                                   explicit=explicit, payload=payload, write=write))
    by_point = {p: [i for i, a in enumerate(candidates) if p in a['write']] for p in points}
    results = []

    def cover(remaining, chosen, assignments):
        if len(results) >= 2:
            return
        if not remaining:
            results.append(chosen)
            return
        p = min(remaining, key=lambda q: (sum(candidates[i]['write'] <= remaining for i in by_point[q]), q))
        for i in by_point[p]:
            arrow = candidates[i]
            if not arrow['write'] <= remaining:
                continue
            target, payload = arrow['target'], arrow['payload']
            if target in assignments and assignments[target] != payload:
                continue
            cover(remaining - arrow['write'], chosen + [arrow], {**assignments, target: payload})

    cover(points, [], {})
    return results


def _decode(input_grid):
    grid = _grid(input_grid)
    parsed = _parse(grid)
    pieces, side = parsed['pieces'], parsed['side']
    table = {(piece, corner): _placements(array, side, corner, piece == 0)
             for piece, array in enumerate(pieces) for corner in range(4)
             if (piece == 0) == (corner == 0)}
    full = (1 << (side * side)) - 1
    valid, geometric = {}, set()
    for rest in permutations((1, 2, 3)):
        order = (0, *rest)

        def search(corner, occupied, chosen):
            if corner == 4:
                if occupied != full:
                    return
                geometric.add(tuple((order[j], chosen[j]['mask']) for j in range(4)))
                square = np.empty((side, side), dtype=np.int8)
                owner = np.empty((side, side), dtype=np.int8)
                for j, placement in enumerate(chosen):
                    rr, cc = placement['rows'], placement['cols']
                    square[rr, cc] = placement['values']
                    owner[rr, cc] = j
                carriers = [parsed['carriers'][p] for p in order]
                for ink in parsed['inks']:
                    interpretations = _read_arrows(square, owner, carriers, parsed['key_location'], ink, parsed['legend'])
                    for arrows in interpretations:
                        signature = _key(owner), _key(square), ink
                        # Distinct motif covers count as distinct interpretations.
                        cover_signature = tuple(sorted((a['center'], a['direction']) for a in arrows))
                        signature += (cover_signature,)
                        assignments = {a['target']: a['payload'] for a in arrows}
                        original = parsed['legend'].reshape(-1)
                        final = np.array([assignments.get(i, int(original[i])) for i in range(4)], dtype=np.int8)
                        valid[signature] = dict(parsed, owner=owner, square=square, arrows=arrows,
                                                ink=ink, assignments=assignments, final=final,
                                                output=final[owner], order=order,
                                                rotations=tuple(p['rotation'] for p in chosen))
                return
            for placement in table[order[corner], corner]:
                if not occupied & placement['mask']:
                    search(corner+1, occupied | placement['mask'], chosen + [placement])

        search(0, 0, [])
    _need(len(valid) == 1, "decorated square assembly must be unique")
    result = next(iter(valid.values()))
    result['shape_only_placements'] = len(geometric)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete broken arrows while packing, then transfer original key colors."""
    return _decode(input_grid)['output'].copy()


def _partition(rng, side, style):
    if style == 0:
        vertical = np.full(side, side // 2)
        horizontal = np.full(side, side // 2)
    else:
        def path():
            value = int(rng.integers(3, side-2))
            result = np.full(side, value)
            if style >= 2:
                for start in range(0, side, int(rng.integers(2, 5))):
                    value = int(np.clip(value + rng.integers(-2, 3), 3, side-3))
                    result[start:start+3] = value
            return result
        vertical, horizontal = path(), path()
    rr, cc = np.indices((side, side))
    owner = (2 * (rr >= horizontal[cc]) + (cc >= vertical[rr])).astype(np.int8)
    if style == 3:
        # Small tabs/notches across either wandering seam, preserving corners.
        for _ in range(side):
            r, c = map(int, rng.integers(1, side-1, size=2))
            values = [owner[r+dr, c+dc] for dr, dc in DIRECTIONS]
            if len(set(map(int, values))) > 1:
                owner[r, c] = rng.choice(values)
    for label in range(4):
        if len(_components(owner == label)) != 1 or np.count_nonzero(owner == label) < 9:
            return None
    if tuple(owner[r, c] for r, c in ((0, 0), (0, side-1), (side-1, 0), (side-1, side-1))) != (0, 1, 2, 3):
        return None
    return owner


def _ports(owner):
    result = []
    side = len(owner)
    for r in range(side):
        for c in range(side):
            source = int(owner[r, c])
            for dr, dc in DIRECTIONS:
                pr, pc = -dc, dr
                arms = ((r+pr, c+pc), (r-pr, c-pc))
                tip = r+dr, c+dc
                write = {(r, c), *arms, tip}
                if not all(0 <= y < side and 0 <= x < side for y, x in write):
                    continue
                target = int(owner[tip])
                if source != target and all(owner[q] == source for q in arms):
                    result.append(dict(center=(r, c), direction=(dr, dc), source=source, target=target, write=write))
    return result


def _annotations(rng, owner, legend, ink, payload_color, requests, count, allow_duplicate):
    ports = _ports(owner)
    rng.shuffle(ports)
    locations = [(r, c) for r in range(len(owner)-1) for c in range(len(owner)-1)
                 if np.all(owner[r:r+2, c:c+2] == 0)]
    rng.shuffle(locations)
    for kr, kc in locations:
        key_cells = {(kr+i, kc+j) for i in range(2) for j in range(2)}
        blocked = {(r+dr, c+dc) for r, c in key_cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        chosen, assignments = [], {}
        for source, target, explicit in requests:
            options = [p for p in ports if p['source'] == source and p['target'] == target and not p['write'] & blocked]
            if not options:
                break
            port = options[0]
            payload = payload_color if explicit else int(legend[source])
            chosen.append(dict(port, explicit=explicit, payload=payload))
            assignments[target] = payload
            blocked.update((r+dr, c+dc) for r, c in port['write'] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        if len(chosen) != len(requests):
            continue
        for port in ports:
            if len(chosen) >= count:
                break
            if port['write'] & blocked:
                continue
            source, target = port['source'], port['target']
            if target in assignments and not allow_duplicate:
                continue
            explicit = bool(rng.random() < .35)
            if target in assignments:
                payload = assignments[target]
                explicit = payload != int(legend[source])
            elif explicit:
                choices = [int(c) for c in [*legend, payload_color] if c not in (legend[source], legend[target])]
                payload = int(rng.choice(choices))
            else:
                payload = int(legend[source])
            chosen.append(dict(port, explicit=explicit, payload=payload))
            assignments[target] = payload
            blocked.update((r+dr, c+dc) for r, c in port['write'] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        if len(chosen) == count:
            return (kr, kc), chosen, assignments
    return None


def _scatter(rng, pieces, background):
    """Random ordered shelves, with both axes and every grouping available."""
    options = []
    for transpose in (False, True):
        dims = [p.shape[::-1] if transpose else p.shape for p in pieces]
        for order in permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, i in enumerate(order):
                    group.append(i)
                    if j == 3 or (cuts >> j) & 1:
                        groups.append(group)
                        group = []
                gap = int(rng.integers(1, 4))
                widths = [sum(dims[i][1] for i in g) + gap*(len(g)-1) for g in groups]
                heights = [max(dims[i][0] for i in g) for g in groups]
                h, w = sum(heights) + gap*(len(groups)-1), max(widths)
                if h <= 28 and w <= 28:
                    options.append((transpose, groups, gap, heights, widths, h, w))
    _need(options, "pieces cannot fit the canvas")
    transpose, groups, gap, heights, widths, h, w = options[int(rng.integers(len(options)))]
    # A visible blank perimeter is a convention, not an anti-copy edge artifact.
    before_r = int(rng.integers(1, min(4, 30-h)))
    after_r = int(rng.integers(1, min(4, 31-h-before_r)))
    before_c = int(rng.integers(1, min(4, 30-w)))
    after_c = int(rng.integers(1, min(4, 31-w-before_c)))
    canvas = np.full((h+before_r+after_r, w+before_c+after_c), background, dtype=np.int8)
    row = before_r
    for group, height, width in zip(groups, heights, widths):
        col = before_c + int(rng.integers(w-width+1))
        for i in group:
            array = pieces[i].T if transpose else pieces[i]
            r = row + int(rng.integers(height-array.shape[0]+1))
            cells = array >= 0
            patch = canvas[r:r+array.shape[0], col:col+array.shape[1]]
            patch[cells] = array[cells]
            col += array.shape[1] + gap
        row += height + gap
    return canvas.T.copy() if transpose else canvas


def _make_pair(rng, side, style, requests, count, rotations, allow_duplicate=True):
    owner = _partition(rng, side, style)
    if owner is None:
        return None
    palette = rng.permutation(10).tolist()
    background, ink = palette[:2]
    legend = np.asarray(palette[2:6], dtype=np.int8)
    carrier_pool = palette[6:6+int(rng.integers(1, 4))]
    carriers = [int(rng.choice(carrier_pool)) for _ in range(4)]
    payload_color = palette[9]
    plan = _annotations(rng, owner, legend, ink, payload_color, requests, count, allow_duplicate)
    if plan is None:
        return None
    key_location, arrows, assignments = plan
    assembled = np.asarray(carriers, dtype=np.int8)[owner]
    for arrow in arrows:
        for q in arrow['write']:
            assembled[q] = ink
        if arrow['explicit']:
            assembled[arrow['center']] = arrow['payload']
    kr, kc = key_location
    assembled[kr:kr+2, kc:kc+2] = legend.reshape(2, 2)
    pieces = []
    for i in range(4):
        cells = np.argwhere(owner == i)
        low, high = cells.min(axis=0), cells.max(axis=0)+1
        array = np.full(tuple(high-low), -1, dtype=np.int8)
        local = cells-low
        array[local[:, 0], local[:, 1]] = assembled[cells[:, 0], cells[:, 1]]
        pieces.append(np.rot90(array, rotations[i]).copy())
    inp = _scatter(rng, pieces, background)
    final = np.asarray([assignments.get(i, int(legend[i])) for i in range(4)], dtype=np.int8)
    output = final[owner]
    try:
        info = _decode(inp)
    except ValueError:
        return None
    if not np.array_equal(info['output'], output) or not np.array_equal(info['owner'], owner):
        return None
    return dict(input=inp, output=output), info


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError(name + ' outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    cycle = list(CYCLE if rng.random() < .5 else CYCLE[::-1])
    start = int(rng.integers(4))
    cycle = cycle[start:] + cycle[:start]
    puzzle, seen = dict(train=[], test=[]), set()
    taught_duplicate = False
    for index in range(int(num_train + num_test)):
        train = index < num_train
        if index == 0:
            requests = [(cycle[0], cycle[1], False), (cycle[1], cycle[2], False)]
            count, style = 2, 0
            side = int(rng.choice([8, 10, 12]))
            rotations = [0, 1, 2, 3]
        elif index == 1:
            requests = [(cycle[2], cycle[3], True), (cycle[3], cycle[0], False)]
            count, style = 2, int(rng.integers(2, 4))
            side = int(rng.integers(10, 13))
            rotations = [0, *map(int, rng.integers(0, 4, size=3))]
        else:
            requests = []
            count = int(rng.integers(1, 5) if train else rng.integers(3, 7 if taught_duplicate else 5))
            style = int(rng.integers(4))
            bounds = ((8, 13) if train else (11, 14)) if regime == 0 else ((10, 14) if train else (8, 12)) if regime == 1 else (9, 14)
            side = int(rng.integers(*bounds))
            rotations = [0, *map(int, rng.integers(0, 4, size=3))]
        for attempt in range(160):
            result = _make_pair(rng, side, style, requests, count, rotations,
                                allow_duplicate=train or taught_duplicate)
            if result is None:
                # Retain the intended mechanism, relaxing only sampled geometry.
                if attempt > 35 and index != 0:
                    side = min(13, side+1)
                    style = int(rng.integers(4))
                continue
            pair, info = result
            if _key(pair['input']) in seen:
                continue
            if index == 0 and info['shape_only_placements'] <= 1:
                continue
            seen.add(_key(pair['input']))
            puzzle['train' if train else 'test'].append(pair)
            if train:
                taught_duplicate |= _features(info)['duplicate_incoming']
            break
        else:
            raise RuntimeError('bounded broken-arrow construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    arrows = info['arrows']
    implicit = [a for a in arrows if not a['explicit']]
    original = info['legend'].reshape(-1)
    chain = any(a['target'] == b['source'] and a['payload'] != b['payload']
                for a in implicit for b in implicit if a is not b)
    # Counterfactuals omit one inherited mechanism and compare full arrays.
    without_transfer = original[info['owner']]
    carrier_by_corner = np.asarray([info['carriers'][i] for i in info['order']])
    wrong_final = carrier_by_corner.copy()
    for a in arrows:
        wrong_final[a['target']] = a['payload'] if a['explicit'] else carrier_by_corner[a['source']]
    areas, fills = [], []
    for i in range(4):
        coords = np.argwhere(info['owner'] == i)
        extent = coords.max(axis=0)-coords.min(axis=0)+1
        areas.append(len(coords))
        fills.append(len(coords)/int(np.prod(extent)))
    return dict(side=info['side'], arrows=len(arrows), explicit=sum(a['explicit'] for a in arrows),
                chain=chain, directions={a['direction'] for a in arrows},
                fanout=len({a['source'] for a in arrows}) < len({(a['source'], a['target']) for a in arrows}),
                duplicate_incoming=len(info['assignments']) < len(arrows),
                changed=int(np.count_nonzero(info['final'] != original)),
                untouched=set(range(4))-set(info['assignments']),
                areas=tuple(areas), min_fill=min(fills), carrier_count=len(set(info['carriers'])),
                shape_only_placements=info['shape_only_placements'],
                omit_a_diff=not np.array_equal(without_transfer, info['output']),
                omit_key_diff=not np.array_equal(wrong_final[info['owner']], info['output']))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
          and len(puzzle['train']) > len(puzzle['test']), 'illegal pair counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            inp, output = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _need(np.array_equal(output, info['output']), 'wrong assembly or simultaneous transfer')
            f = _features(info)
            _need(f['omit_a_diff'], 'transfer contribution is inert')
            features[split].append(f)
    train = features['train']
    _need(any(f['chain'] for f in train), 'original-color simultaneity not demonstrated')
    _need(any(f['explicit'] for f in train), 'literal junction payload not demonstrated')
    _need(any(f['shape_only_placements'] > 1 for f in train), 'fragment docking not demonstrated')
    _need(any(f['omit_key_diff'] for f in train), 'corner key contribution is inert')
    _need(set.union(*(f['untouched'] for f in train)) == set(range(4)), 'not all original key roles exposed')
    _need(not any(f['duplicate_incoming'] for f in features['test'])
          or any(f['duplicate_incoming'] for f in train), 'multiple incoming arrows lack a demonstration')


def self_test():
    """Bounded 42-seed audit, counterfactuals, determinism, corruption and counts."""
    records, geometry, canvases, palettes, graphs, seen = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(info)
                records.append(f)
                geometry.add(_key(info['owner']))
                canvases.add(pair['input'].shape)
                palettes.add(tuple(map(int, info['legend'].flat)))
                graphs.add(tuple(sorted((a['source'], a['target'], a['explicit']) for a in info['arrows'])))
                signature = _key(pair['input'])
                _need(signature not in seen, 'duplicate input across audited seeds')
                seen.add(signature)
                _need(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only solver failure')
        if seed < 2:
            other = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], other[s])
                      for k in ('input', 'output')), 'nondeterministic seed')
    _need(len(geometry) >= 70 and len(canvases) >= 50 and len(graphs) >= 25, 'structural variation collapsed')
    _need(len(palettes) >= 100, 'legend variation collapsed')
    _need({f['arrows'] for f in records} >= {1, 2, 3, 4, 5, 6}, 'missing instruction counts')
    _need(set.union(*(f['directions'] for f in records)) == set(DIRECTIONS), 'missing directions')
    _need(any(f['fanout'] for f in records) and any(f['duplicate_incoming'] for f in records), 'missing graph variation')
    _need({f['carrier_count'] for f in records} >= {1, 2, 3}, 'missing shared/distinct carriers')
    _need(any(f['min_fill'] < .7 for f in records), 'missing interlocking geometry')
    _need(any(f['changed'] == 4 for f in records), 'missing all-target transfers')
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+10*tr+te, tr, te)
            _need(len(p['train']) == tr and len(p['test']) == te, 'explicit counts ignored')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(0)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(records), 'partitions': len(geometry),
            'canvases': len(canvases), 'keys': len(palettes), 'graphs': len(graphs)}
