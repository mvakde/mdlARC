import hashlib

import numpy as np


def _is_integer_argument(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    seed_key: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_integer_argument(num_train):
            raise ValueError("num_train must be an integer or None")
        num_train = int(num_train)
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        if not _is_integer_argument(num_test):
            raise ValueError("num_test must be an integer or None")
        num_test = int(num_test)
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed_key % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_key // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_key // 7) % span
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + (seed_key // 11) % maximum_test

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("pair counts must be legal and num_train > num_test")

    return int(num_train), int(num_test)


def _color_components(
    grid: np.ndarray,
    color: int,
) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) != color:
                continue

            seen[row, col] = True
            stack = [(row, col)]
            component: list[tuple[int, int]] = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if seen[next_row, next_col]:
                        continue
                    if int(grid[next_row, next_col]) != color:
                        continue

                    seen[next_row, next_col] = True
                    stack.append((next_row, next_col))

            components.append(component)

    return components


def _find_exact_swatches(grid: np.ndarray, excluded=()) -> list[dict]:
    swatches: list[dict] = []

    for color_value in np.unique(grid):
        color = int(color_value)
        if color in excluded:
            continue
        for component in _color_components(grid, color):
            if len(component) != 4:
                continue

            rows = sorted({row for row, _ in component})
            cols = sorted({col for _, col in component})
            if len(rows) != 2 or len(cols) != 2:
                continue
            if rows[1] != rows[0] + 1 or cols[1] != cols[0] + 1:
                continue

            expected = {
                (rows[0], cols[0]),
                (rows[0], cols[1]),
                (rows[1], cols[0]),
                (rows[1], cols[1]),
            }
            if set(component) != expected:
                continue

            swatches.append(
                {
                    "row": rows[0],
                    "col": cols[0],
                    "color": color,
                    "cells": tuple(sorted(component)),
                }
            )

    return swatches


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    result = np.rot90(grid, -code % 4)
    return (result[:, ::-1] if code >= 4 else result).copy()


def _canonical_point(point, shape, code):
    r, c = point
    h, w = shape
    k = code % 4
    r, c = ((r, c), (c, h - 1 - r), (h - 1 - r, w - 1 - c),
            (w - 1 - c, r))[k]
    cw = h if k % 2 else w
    return int(r), int(cw - 1 - c if code >= 4 else c)


def _inside(shape, point):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _trace_route(mask, start):
    """Production tracer: maximal runs; leaving the canvas is terminal."""
    current, direction = start, 0
    path, collisions, legs = [], [], []
    directions = ((0, 1), (-1, 0))
    for _ in range(sum(mask.shape)):
        dr, dc = directions[direction]
        r, c = current
        leg = []
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r + dr, c + dc
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        # Do not turn around an out-of-bounds collision.
        if not _inside(mask.shape, (r, c)):
            end = "edge"
            break
        collisions.append((r, c))
        direction = 1 - direction
        dr, dc = directions[direction]
        current = (leg[-1][0] + dr, leg[-1][1] + dc)
        if not _inside(mask.shape, current) or not mask[current]:
            end = "blocked"
            break
    else:
        raise AssertionError("nonterminating route")
    return dict(path=tuple(path), collisions=tuple(collisions),
                legs=tuple(legs), end=end)


def _walk_cells(mask, start):
    """Independent cell-step validator, not a call to the production tracer."""
    r, c = start
    vertical = False
    path, collisions, legs = [], [], [0]
    while True:
        assert mask[r, c] and (r, c) not in path
        path.append((r, c))
        legs[-1] += 1
        ahead = (r - 1, c) if vertical else (r, c + 1)
        if not _inside(mask.shape, ahead):
            end = "edge"
            break
        if mask[ahead]:
            r, c = ahead
            continue
        collisions.append(ahead)
        turn = (r, c + 1) if vertical else (r - 1, c)
        if not _inside(mask.shape, turn) or not mask[turn]:
            end = "blocked"
            break
        vertical = not vertical
        legs.append(0)
        r, c = turn
    return dict(path=tuple(path), collisions=tuple(collisions),
                legs=tuple(legs), end=end)


def _decode(grid):
    """Recover roles and semantic directions solely from visible input cues."""
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 5
    terrain_colors = set(int(x) for x in colors[np.argsort(counts)[-2:]])
    swatches = _find_exact_swatches(grid, terrain_colors)
    assert 3 <= len(swatches) <= 7
    annotation_cells = {p for s in swatches for p in s['cells']}
    starts = [tuple(map(int, p)) for p in np.argwhere(
        ~np.isin(grid, list(terrain_colors))) if tuple(p) not in annotation_cells]
    assert 1 <= len(starts) <= 4
    assert len({int(grid[p]) for p in starts}) == 1
    first = int(grid[starts[0]])
    # Swatch surrounds identify the erased terrain; area rank has no semantic role.
    rings = set()
    for swatch in swatches:
        for r, c in swatch['cells']:
            for p in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if _inside(grid.shape, p) and p not in annotation_cells:
                    rings.add(int(grid[p]))
    assert len(rings) == 1 and rings <= terrain_colors
    background = rings.pop()
    route_color = (terrain_colors - {background}).pop()
    restored = grid.copy()
    for p in annotation_cells:
        restored[p] = background
    for p in starts:
        restored[p] = route_color
    assert set(map(int, np.unique(restored))) == terrain_colors
    candidates = []
    for code in range(8):
        canonical = _unorient(restored, code)
        mask = canonical == route_color
        cs = tuple(sorted(_canonical_point(p, grid.shape, code) for p in starts))
        transformed = []
        for s in swatches:
            cells = tuple(_canonical_point(p, grid.shape, code) for p in s['cells'])
            transformed.append(dict(row=min(p[0] for p in cells),
                                    col=min(p[1] for p in cells),
                                    color=s['color'], cells=cells))
        for marker in transformed:
            legend = sorted([s for s in transformed if s is not marker],
                            key=lambda s: s['col'])
            if len({s['row'] for s in legend}) != 1:
                continue
            if any(b['col'] - a['col'] < 3 for a, b in zip(legend, legend[1:])):
                continue
            sequence = tuple(s['color'] for s in legend)
            if first != sequence[0] or marker['color'] in sequence:
                continue
            if not all(r >= legend[0]['row'] + 4 for r, c in cs):
                continue
            if not all((c == 0 or not mask[r,c-1]) and c+1 < mask.shape[1]
                       and mask[r,c+1] for r,c in cs):
                continue
            if any(abs(r-s) + abs(c-t) <= 1 for i,(r,c) in enumerate(cs)
                   for s,t in cs[i+1:]):
                continue
            candidates.append(dict(orientation=code, terrain=canonical, mask=mask,
                                   starts=cs, sequence=sequence, marker=marker,
                                   legend=legend, background=background,
                                   route_color=route_color))
    assert len(candidates) == 1, "ambiguous legend or travel orientation"
    info = candidates[0]
    annotation_colors = set(info['sequence']) | {info['marker']['color']}
    for color in annotation_colors:
        expected = 4 * sum(s['color'] == color for s in swatches)
        expected += len(starts) if color == first else 0
        assert np.count_nonzero(grid == color) == expected
    return info


def _analyze_input(grid):
    info = _decode(grid)
    mask = info['mask']
    records = [_walk_cells(mask, start) for start in info['starts']]
    occupied = set()
    annotations = {p for s in info['legend'] + [info['marker']] for p in s['cells']}
    for record in records:
        footprint = set(record['path']) | set(record['collisions'])
        assert not footprint & occupied, "routes must not merge or share markers"
        assert not footprint & annotations, "route collides with an annotation"
        occupied |= footprint
    output = info['terrain'].copy()
    sequence = info['sequence']
    for record in records:
        for i, p in enumerate(record['path']):
            output[p] = sequence[i % len(sequence)]
        for p in record['collisions']:
            output[p] = info['marker']['color']
    info['records'] = records
    info['expected_output'] = _orient(output, info['orientation'])
    return info


def _equality_pattern(sequence):
    seen = {}
    return tuple(seen.setdefault(x, len(seen)) for x in sequence)


def _coverage(info):
    """Observe which plausible incorrect programs training actually rules out."""
    word = info['sequence']
    unique = tuple(dict.fromkeys(word))
    evidence = set()
    offset = 0
    for record in info['records']:
        n = len(record['path'])
        correct = [word[i % len(word)] for i in range(n)]
        if n >= 2 * len(word):
            evidence.add('cycling')
        if correct != [unique[i % len(unique)] for i in range(n)]:
            evidence.add('multiplicity')
        reset, consume = [], []
        phase = 0
        for j, length in enumerate(record['legs']):
            reset.extend(word[i % len(word)] for i in range(length))
            consume.extend(word[(phase+i+j) % len(word)] for i in range(length))
            phase += length
        if correct != reset:
            evidence.add('bend_phase')
        if correct != consume:
            evidence.add('marker_phase')
        if correct != [word[(offset+i) % len(word)] for i in range(n)]:
            evidence.add('restart')
        offset += n
        if record['collisions']:
            evidence.add('collision')
        if record['end'] == 'edge':
            r, c = record['path'][-1]
            corner = (r-1,c) if len(record['legs']) % 2 else (r,c+1)
            if _inside(info['mask'].shape, corner) and info['mask'][corner]:
                evidence.add('edge_stop')
    return evidence


_REQUIRED_EVIDENCE = {'cycling', 'multiplicity', 'bend_phase', 'marker_phase',
                      'restart', 'collision', 'edge_stop'}


def _sample_word(rng, maximum=6, repeat=False, length=None):
    n = int(rng.integers(3 if repeat else 2, maximum+1)) if length is None else length
    d = int(rng.integers(2, n if repeat else n+1))
    # Every chosen color occurs; repetition positions are unconstrained.
    word = list(range(d)) + list(map(int, rng.integers(0, d, n-d)))
    rng.shuffle(word)
    unique = tuple(dict.fromkeys(word))
    if repeat and all(x == unique[i % d] for i,x in enumerate(word)):
        # An ABAB program cannot demonstrate why deduplicating ABAB is wrong.
        word[0], word[1] = word[1], word[0]
    return tuple(word)


def _profile(rng, height, low, high, roughness):
    knots = np.unique(np.r_[0, rng.choice(np.arange(1,height-1),
                      size=min(height-2, roughness), replace=False), height-1])
    return np.rint(np.interp(np.arange(height), knots,
                           rng.uniform(low, high, len(knots)))).astype(int)


def _terrain(rng, h, w, clear, density, roughness):
    # Independent boundaries create lobes, wide ledges, necks and recesses.
    left = _profile(rng, h, -w*.28, w*(.55-density*.35), roughness)
    right = _profile(rng, h, w*(.30+density*.32), w*(.75+density*.65), roughness)
    left = np.clip(left, 0, w-3)
    right = np.clip(right, left+1, w-1)
    # Maintain overlap of successive sections without a permanent connector.
    for r in range(1,h):
        if left[r] > right[r-1]:
            left[r] = right[r-1]
        if right[r] < left[r-1]:
            right[r] = left[r-1]
    mask = np.zeros((h,w), dtype=bool)
    for r in range(h):
        mask[r,left[r]:right[r]+1] = True
    # Carve bays and sometimes interior holes away from the annotation margin.
    for _ in range(int(rng.integers(0, roughness+1))):
        r = int(rng.integers(0,h-1)); c = int(rng.integers(0,w-1))
        dh = int(rng.integers(1, min(5,h-r)+1))
        dw = int(rng.integers(1, min(7,w-c)+1))
        mask[r:r+dh,c:c+dw] = bool(rng.integers(0,2))
    mask[clear] = False
    # Retain the largest component, so the same-color regions stay one terrain.
    components = _color_components(mask.astype(np.int8), 1)
    if not components:
        return mask
    largest = max(components, key=len)
    mask[:] = False
    for p in largest:
        mask[p] = True
    return mask


def _fallback_terrain(h, w, clear, variant, count=2, informative=True):
    """Bounded constructive backstop, with a genuine exit and a shorter route."""
    a = 5 + (variant // 3) % 6
    d = 3 + (variant // 18) % 7
    mask = np.zeros((h,w), dtype=bool)
    if not informative:
        rows = np.linspace(4,h-2,count,dtype=int)
        mask[4:,:2] = True
        for j,r in enumerate(rows):
            width = max(6,w-5-(variant+j)%max(1,w//3))
            mask[r:r+(h-4 if count == 1 else 2),:width] = True
        mask[clear] = False
        return mask
    mask[5:h-5, :a] = True
    mask[5:8, a:] = True
    mask[3:6, w-3:] = True  # available corner past the horizontal exit
    mask[h-5:, :2] = True
    mask[h-2:, :d] = True
    mask[clear] = False
    return mask


def _select_routes(rng, mask, legend_row, count, informative, minimum_legs=1):
    candidates = []
    for r in range(legend_row+4, mask.shape[0]):
        for c in range(mask.shape[1]-1):
            if mask[r,c] and mask[r,c+1] and (c == 0 or not mask[r,c-1]):
                record = _trace_route(mask, (r,c))
                if len(record['path']) >= 3:
                    candidates.append(((r,c), record))
    rng.shuffle(candidates)
    if informative or minimum_legs > 1:
        # Favor, but do not preassign, a long bent route and a nontrivial exit.
        candidates.sort(key=lambda x: (len(x[1]['legs']) >= minimum_legs,
                                       informative and x[1]['end'] == 'edge',
                                       len(x[1]['legs']) > 1,
                                       len(x[1]['path'])), reverse=True)
    chosen, occupied = [], set()
    for start, record in candidates:
        footprint = set(record['path']) | set(record['collisions'])
        if footprint & occupied or any(abs(start[0]-s[0])+abs(start[1]-s[1]) < 3
                                       for s, _ in chosen):
            continue
        chosen.append((start, record))
        occupied |= footprint
        if len(chosen) == count:
            break
    return sorted(chosen)


def _dimensions(rng, regime, schedule, word_length, informative):
    test = regime == 'target_test'
    low, high = 12, 30
    if schedule == 'growth':
        low, high = (26,30) if test else (18,25)
    elif schedule == 'small_test':
        low, high = (12,20) if test else (20,30)
    elif schedule == 'size_only':
        low, high = (24,30) if test else (15,24)
    elif schedule == 'overlap':
        low, high = 16, 26
    if informative:
        low = max(low, 20)
        high = max(high, low)
    min_width = max(low, 3*word_length+2)
    w = int(rng.integers(min_width, max(high,min_width)+1))
    h = int(rng.integers(low,high+1))
    if rng.random() < .22:
        h = w
    return h,w


def _generate_pair(rng, regime, schedule, orientation, informative=False,
                   variation=1):
    maximum = (3 if schedule == 'growth' else 4) if (
        schedule in ('growth','cycle_only') and regime == 'source_train') else 6
    if schedule == 'growth' and regime == 'target_test':
        word = _sample_word(rng, 4, length=4)
    else:
        word = _sample_word(rng, maximum, informative)
    h,w = _dimensions(rng, regime, schedule, len(word), informative)
    route_count = int(rng.choice([1,2,3,4], p=[.42,.44,.11,.03]))
    if informative:
        route_count = 2
    elif schedule == 'growth' and regime == 'target_test':
        route_count = 2
    if h < 17:
        route_count = min(route_count,2)
    minimum_legs = int(rng.choice([1,3,5], p=([.85,.15,0] if variation == 0 else
                                             [.40,.35,.25] if variation == 1 else
                                             [.20,.35,.45])))
    palette = tuple(map(int,rng.permutation(10)))
    bg, route, marker_color = palette[:3]
    sequence = tuple(palette[3+x] for x in word)
    for attempt in range(180):
        fallback = attempt >= 120
        row = int(rng.integers(0, min(4,h//5))) if not fallback else 0
        slack = w - 2 - 3*len(word)
        left = int(rng.integers(0, min(5,max(0,slack))+1))
        gaps = np.ones(len(word)-1, dtype=int)
        for j in range(len(gaps)):
            cap = min(3, w-1-left-2*len(word)-int(gaps.sum())+int(gaps[j]))
            gaps[j] = int(rng.integers(1,max(1,cap)+1))
        columns = [left]
        for gap in gaps:
            columns.append(columns[-1]+2+int(gap))
        if columns[-1]+2 > w:
            continue
        clear = np.zeros((h,w), dtype=bool)
        clear[max(0,row-1):row+3, max(0,left-1):min(w,columns[-1]+3)] = True
        density = float(rng.uniform(.25,.95))
        roughness = int(rng.integers(2,5 if variation == 0 else 10))
        mask = (_fallback_terrain(h,w,clear,attempt-120,route_count,informative) if fallback else
                _terrain(rng,h,w,clear,density,roughness))
        if min(np.count_nonzero(mask),np.count_nonzero(~mask)-4*(len(word)+1)) < 30:
            continue
        if fallback:
            starts = ([(h-6-(attempt-120)%3,0),(h-2,0)] if informative else
                      [(int(r),0) for r in np.linspace(4,h-2,route_count,dtype=int)])
            chosen = [(s,_trace_route(mask,s)) for s in starts]
        else:
            chosen = _select_routes(rng,mask,row,route_count,informative,minimum_legs)
        if len(chosen) != route_count:
            continue
        records = [record for start,record in chosen]
        if attempt < 80 and max(len(r['legs']) for r in records) < minimum_legs:
            continue
        tmp_info = dict(sequence=sequence, records=records, mask=mask)
        if informative and not _REQUIRED_EVIDENCE <= _coverage(tmp_info):
            continue
        blocked = clear.copy()
        for start, record in chosen:
            for r,c in record['path'] + record['collisions']:
                blocked[max(0,r-1):r+2,max(0,c-1):c+2] = True
        marker_positions = []
        for r in range(h-1):
            if abs(r-row) < 4:
                continue
            for c in range(w-1):
                ring = (slice(max(0,r-1),min(h,r+3)),slice(max(0,c-1),min(w,c+3)))
                if not np.any(mask[ring]) and not np.any(blocked[r:r+2,c:c+2]):
                    marker_positions.append((r,c))
        if not marker_positions:
            continue
        mr,mc = marker_positions[int(rng.integers(len(marker_positions)))]
        terrain = np.where(mask,route,bg).astype(np.int8)
        inp, out = terrain.copy(), terrain.copy()
        for c,color in zip(columns,sequence):
            inp[row:row+2,c:c+2] = color
        inp[mr:mr+2,mc:mc+2] = marker_color
        for start,record in chosen:
            inp[start] = sequence[0]
            for phase,p in enumerate(record['path']):
                out[p] = sequence[phase % len(sequence)]
            for p in record['collisions']:
                out[p] = marker_color
        pair = dict(input=_orient(inp,orientation),output=_orient(out,orientation))
        # Decodability is checked on the actual rendered symbols.
        try:
            info = _analyze_input(pair['input'])
        except AssertionError:
            continue
        assert np.array_equal(info['expected_output'],pair['output'])
        return pair
    raise AssertionError('constructive fallback failed')


def _array_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p['input']),_array_key(p['output']))
                       for p in puzzle[s]) for s in ('train','test'))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 63)
    nt,ne = _resolve_counts(seed,num_train,num_test)
    rng = np.random.default_rng(seed)
    # These sampling regimes change difficulty, never the semantic rule.
    schedule = str(rng.choice(['growth','overlap','small_test','mixed','size_only','cycle_only']))
    orientation_mode = int(rng.integers(0,3))  # fixed, mixed, held-out rotation
    base_orientation = int(rng.integers(0,8))
    shifted = (base_orientation//4)*4 + (base_orientation+int(rng.integers(1,4)))%4
    variation = int(rng.integers(0,3))
    witness_index = int(rng.integers(nt))
    result = {'train':[], 'test':[]}
    used = set()
    for split,count,regime in (('train',nt,'source_train'),('test',ne,'target_test')):
        for i in range(count):
            orientation = base_orientation
            if orientation_mode == 1:
                orientation = int(rng.integers(0,8))
            elif orientation_mode == 2 and split == 'test':
                orientation = shifted
            for retry in range(16):
                pair = _generate_pair(rng,regime,schedule,orientation,
                                      split == 'train' and i == witness_index, variation)
                key = _array_key(pair['input'])
                if key not in used and _source_digest(pair) not in _SOURCE_DIGESTS:
                    break
            else:
                raise AssertionError('duplicate construction')
            result[split].append(pair)
            used.add(key)
    validate_generated_puzzle(result)
    return result


def _source_digest(pair):
    data = bytes(pair['input'].shape) + pair['input'].astype(np.uint8).tobytes()
    data += bytes(pair['output'].shape) + pair['output'].astype(np.uint8).tobytes()
    return hashlib.sha256(data).hexdigest()


_SOURCE_DIGESTS = {'66da055685b11876d71da409ead5e8077cfd84aa8a76c865c96481ca3a3b078c', 'af3d7852f9e6fed171edb24e0a69bee40da5899455f4996ab79daddb6ca66f4a', 'd33a3db6b7bf21aa31088f4d225fdb2b553809f0144ac81084de505a043b5a98', '587f9f66c3e32be145206f590c286682588342a0358a191663436db8f71c1f7a', '467b9e8970fd1c44440bae41f71974a493f162a6200fcd997046c3b459671be6'}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype,np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp,out = pair['input'],pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            key = _array_key(inp)
            assert key not in seen
            seen.add(key)
            assert _source_digest(pair) not in _SOURCE_DIGESTS
            info = _analyze_input(inp)
            assert np.array_equal(out,info['expected_output'])
            assert len(_color_components(info['mask'].astype(np.int8),1)) == 1
            for start,record in zip(info['starts'],info['records']):
                assert record == _trace_route(info['mask'],start)
            if split == 'train':
                evidence |= _coverage(info)
    # No fixed size, count, cycle length, palette, or orientation train/test rule.
    # All conventions required at test time are witnessed by the demonstrations.
    assert _REQUIRED_EVIDENCE <= evidence, _REQUIRED_EVIDENCE - evidence


def self_test() -> None:
    _edge_regressions()
    for nt,ne in ((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)):
        a = generate_puzzle(8675309,nt,ne)
        b = generate_puzzle(8675309,nt,ne)
        assert _puzzle_key(a) == _puzzle_key(b)
        assert nt is None or len(a['train']) == nt
        assert ne is None or len(a['test']) == ne
    for nt,ne in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(2.0,1),(3,True)):
        try:
            generate_puzzle(99,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    _diversity_check(puzzles)


def _edge_regressions():
    # Literal restored training terrains, pinned to observed source endpoints.
    for rows, start, endpoint, legs in _SOURCE_EDGE_CASES:
        mask = np.array([[int(c) for c in row] for row in rows], dtype=bool)
        for trace in (_trace_route, _walk_cells):
            record = trace(mask, start)
            assert record['path'][-1] == endpoint and record['legs'] == legs
            assert record['end'] == 'edge'
            r, c = endpoint
            corner = (r-1,c) if len(legs) % 2 else (r,c+1)
            assert mask[corner] and corner not in record['path']
    # Source train 1 exits right and must leave the terrain above its endpoint.
    horizontal = np.zeros((4,8),bool)
    horizontal[2,:] = True
    horizontal[1:,7] = True
    expected = tuple((2,c) for c in range(8))
    for trace in (_trace_route,_walk_cells):
        record = trace(horizontal,(2,0))
        assert record['path'] == expected and not record['collisions']
    # Source train 2 exits upward, despite available top-row terrain to its right.
    vertical = np.zeros((7,9),bool)
    vertical[5,:4] = True
    vertical[:6,3] = True
    vertical[0,3:8] = True
    expected = tuple([(5,c) for c in range(4)] + [(r,3) for r in range(4,-1,-1)])
    for trace in (_trace_route,_walk_cells):
        record = trace(vertical,(5,0))
        assert record['path'] == expected and record['collisions'] == ((5,4),)
    # A real terrain collision still permits the turn and consumes no paint phase.
    horizontal[2,6:] = False
    horizontal[0:3,5] = True
    assert _trace_route(horizontal,(2,0)) == _walk_cells(horizontal,(2,0))


def _diversity_check(puzzles):
    infos = [_analyze_input(pair['input']) for puzzle in puzzles
             for split in ('train','test') for pair in puzzle[split]]
    assert len({i['mask'].shape for i in infos}) >= 40
    assert len({i['orientation'] for i in infos}) == 8
    assert {len(i['sequence']) for i in infos} == {2,3,4,5,6}
    assert len({_equality_pattern(i['sequence']) for i in infos}) >= 25
    assert {len(i['starts']) for i in infos} >= {1,2,3}
    assert any(i['mask'].mean() > .60 for i in infos)
    assert any(i['mask'].mean() < .35 for i in infos)
    records = [r for i in infos for r in i['records']]
    assert {len(r['legs']) for r in records} >= {1,2,3,4,5}
    assert max(max(r['legs']) for r in records) >= 20
    assert any(len(set(r['legs'][::2])) >= 3 for r in records)
    assert len({(i['legend'][0]['row'],i['legend'][0]['col']) for i in infos}) >= 10
    assert len({(i['marker']['row'],i['marker']['col']) for i in infos}) >= 50
    assert any(any(b['col']-a['col'] > 3 for a,b in zip(i['legend'],i['legend'][1:]))
               for i in infos)
    assert any(len({r for r,c in i['starts']}) >= 3 for i in infos)
    assert {(20,20),(25,25),(30,30)} <= {i['mask'].shape for i in infos}
    patterns = {_equality_pattern(i['sequence']) for i in infos}
    assert {(0,1,1,0),(0,0,1,1),(0,1,2,1),(0,0,0,1),(0,1,2,3)} <= patterns
    assert len({(i['background'],i['route_color'],i['marker']['color'])
                for i in infos}) >= 100
    assert sum(len(r['legs']) >= 5 for r in records) >= 20
    assert any(len(i['starts']) > 1 and all(c == 0 for r,c in i['starts'])
               for i in infos)
    assert any(max(len(r['path']) for r in i['records']) -
               min(len(r['path']) for r in i['records']) >= 15 for i in infos)
    assert any(len({len(r['path']) % len(i['sequence']) for r in i['records']}) > 1
               for i in infos)
    assert any(len(i['starts']) >= 3 and len(i['sequence']) >= 4 and
               i['orientation'] % 2 and i['mask'].mean() > .5 for i in infos)
    for name,feature in [('orientation',lambda i:i['orientation']),
                         ('word',lambda i:len(i['sequence'])),
                         ('count',lambda i:len(i['starts'])),
                         ('size',lambda i:i['mask'].shape)]:
        varying = sum(len({feature(_analyze_input(p['input'])) for split in ('train','test')
                           for p in puzzle[split]}) > 1 for puzzle in puzzles)
        assert varying >= (10 if name == 'orientation' else 25), (name,varying)


# Source training fixtures used only for termination regression tests.
_SOURCE_EDGE_CASES = (
    ((
        '00000000000000000001',
        '00000000000000001111',
        '00000000000000001111',
        '00000000000000111110',
        '00000000000011111100',
        '00000000011111111000',
        '00000001111111110000',
        '00000111111111110000',
        '00111111111111111111',
        '11111111111111111111',
        '11111111111111111111',
        '11111111111111111110',
        '11111111111111111000',
        '11111111111111100000',
        '11111111111111000000',
        '11111111111110000000',
        '11111111111000000000',
        '11111111110000000000',
        '11111111100000000000',
        '11111111000000000000',
    ), (15, 0), (1, 19), (13, 11, 5, 3, 2)),
    ((
        '0000000000000000011111111',
        '0000000000000000011111111',
        '0000000000000001111111100',
        '0000000000001111111110000',
        '0000000000011111111100000',
        '0000000000111111100000000',
        '0000000001111111100000000',
        '0000000001111111111111000',
        '0000000111111111111111000',
        '0000000111111111111111110',
        '0001011111111111111111110',
        '1111111111111110000000000',
        '1111111111111110000000000',
        '1111111111111110000000000',
        '1111111111111111100000000',
        '0111111111111111111100000',
        '0000001111111111111110000',
        '0000000011111111111100000',
        '0000011111111100110000000',
        '0000011111111100000000000',
        '0001111111111000000000000',
        '0111111111110000000000000',
        '1111111111111111000000000',
        '1111111111111111100000000',
        '1111111111111111100000000',
    ), (13, 0), (0, 20), (15, 10, 6, 3)),
)
