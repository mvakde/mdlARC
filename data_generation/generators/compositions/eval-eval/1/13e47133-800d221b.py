"""Contour parliament: terminals vote into cyclic erosion-depth addresses.

No parent code or data is loaded. Generation retains one puzzle at a time, uses
bounded searches, and writes no files. Run self_test() for bounded diagnostics.
"""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
BACKGROUND, TERMINAL, WIRE, CLOCK = 0, 2, 8, 9
PALETTE = (1, 3, 4, 5, 6, 7)
HOST_SEEDS = (0, 417484019, 534544655, 80976180, 134868435, 394063648,
              956330936, 915443537, 985240551, 726360210, 378782324,
              54918085, 391013171, 643182125, 427123195, 612974658,
              248606673, 99644213, 156825941, 351749357, 69930255,
              118119608, 875675944, 627266307, 793635250, 264602067,
              787313416, 283717503, 807624284, 293824306, 339361780,
              927549312, 730879792, 640345035, 718234428, 942658756,
              302297335, 878948356, 88106020, 201633223, 444438821, 495073782)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _neighbors(cells, shape, diagonal=False):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells
            for dr, dc in (DIR8 if diagonal else DIR4)
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _components(mask):
    # Padded flat Python buffers avoid thousands of NumPy scalar lookups in
    # these tiny graph traversals. Keep row order and DIR4 traversal unchanged.
    h, w = mask.shape
    stride = w + 2
    opened = np.pad(mask, 1).ravel().tolist()
    ids = [-1] * len(opened)
    offsets = (-stride, stride, -1, 1)
    components = []
    for start, is_open in enumerate(opened):
        if not is_open or ids[start] >= 0:
            continue
        index = len(components)
        cells = [start]
        ids[start] = index
        for pos in cells:
            for offset in offsets:
                neighbor = pos + offset
                if opened[neighbor] and ids[neighbor] < 0:
                    ids[neighbor] = index
                    cells.append(neighbor)
        components.append([(pos // stride - 1, pos % stride - 1) for pos in cells])
    return components, np.asarray(ids, dtype=np.int16).reshape(h + 2, w + 2)[1:-1, 1:-1]


def _erosion_depth(mask, diagonal=True):
    """Input-only definition: repeatedly strip the entire exposed boundary."""
    h, w = mask.shape
    alive = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    depth = 0
    while np.any(alive):
        padded = np.pad(alive, 1)
        inner = alive.copy()
        for dr, dc in (DIR8 if diagonal else DIR4):
            inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[alive & ~inner] = depth
        alive = inner
        depth += 1
    return result


def _bfs_depth(mask):
    """Construction cross-check, independent of vectorized peeling above."""
    h, w = mask.shape
    stride = w + 2
    opened = np.pad(mask, 1).ravel().tolist()
    result = [-1] * len(opened)
    offsets = tuple(dr * stride + dc for dr, dc in DIR8)
    queue = deque()
    for pos, is_open in enumerate(opened):
        if is_open and any(not opened[pos + offset] for offset in offsets):
            result[pos] = 0
            queue.append(pos)
    while queue:
        pos = queue.popleft()
        next_depth = result[pos] + 1
        for offset in offsets:
            neighbor = pos + offset
            if opened[neighbor] and result[neighbor] < 0:
                result[neighbor] = next_depth
                queue.append(neighbor)
    return np.asarray(result, dtype=np.int16).reshape(h + 2, w + 2)[1:-1, 1:-1]


def _clock_paths(mask, depth, period):
    h, w = mask.shape
    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and mask[r, c]
    paths = []
    for r, c in np.argwhere(depth == 0):
        r, c = int(r), int(c)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if inside(r - dr, c) or inside(r, c - dc):
                continue
            cells = [(r + d * dr, c + d * dc) for d in range(period)]
            if all(inside(a, b) and depth[a, b] == d
                   for d, (a, b) in enumerate(cells)):
                paths.append(cells)
    return paths


def _box_depth(mask):
    rr, cc = np.nonzero(mask)
    result = np.full(mask.shape, -1, dtype=np.int16)
    result[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                       cc - cc.min(), cc.max() - cc))
    return result


def _election(weights, labels):
    # labels are 0/1 signal indexes; a terminal, rather than a panel, is a vote.
    votes = np.stack((weights[np.asarray(labels) == 0].sum(axis=0),
                      weights[np.asarray(labels) == 1].sum(axis=0)), axis=1)
    if np.any(votes[:, 0] == votes[:, 1]):
        return None, votes
    return np.argmax(votes, axis=1), votes


def _inspect(input_grid):
    a = _grid(input_grid)
    colors = set(map(int, np.unique(a)))
    assert {BACKGROUND, TERMINAL, WIRE, CLOCK} <= colors
    other = sorted(colors - {BACKGROUND, TERMINAL, WIRE, CLOCK})
    assert len(other) == 3
    counts = {c: int(np.count_nonzero(a == c)) for c in other}
    ordered = sorted(other, key=lambda c: -counts[c])
    assert counts[ordered[0]] > counts[ordered[1]]
    substrate = ordered[0]
    signals = sorted(c for c in other if c != substrate)
    # Footprint membership precedes interpretation of all interior decorations.
    body = a != BACKGROUND
    assert len(_components(body)[0]) == 1
    depth = _erosion_depth(body)
    clock = {tuple(map(int, p)) for p in np.argwhere(a == CLOCK)}
    period = len(clock)
    assert period in (2, 3)
    assert any(set(path) == clock for path in _clock_paths(body, depth, period))
    panels, panel_ids = _components(np.isin(a, signals))
    assert 3 <= len(panels) <= 5
    labels, areas, local_counts = [], [], []
    for cells in panels:
        rr, cc = zip(*cells)
        assert len(cells) == (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
        count = [sum(int(a[p]) == color for p in cells) for color in signals]
        assert min(count) > 0 and count[0] != count[1]
        labels.append(int(count[1] > count[0]))
        areas.append(len(cells))
        local_counts.append(count)
    circuits, _ = _components((a == WIRE) | (a == TERMINAL))
    weights = np.zeros((len(panels), period), dtype=int)
    terminal_records, circuit_records = [], []
    for cells in circuits:
        owners = {int(panel_ids[q]) for q in _neighbors(cells, a.shape)
                  if panel_ids[q] >= 0}
        assert len(owners) == 1
        owner = owners.pop()
        tips = [p for p in cells if a[p] == TERMINAL]
        assert tips and any(a[p] == WIRE for p in cells)
        for p in tips:
            around = _neighbors([p], a.shape)
            assert sum(a[q] == WIRE for q in around) == 1
            assert not any(a[q] == TERMINAL or panel_ids[q] >= 0 for q in around)
            phase = int(depth[p]) % period
            weights[owner, phase] += 1
            terminal_records.append((p, owner, int(depth[p]), phase))
        circuit_records.append((cells, owner))
    assert np.all(weights.sum(axis=1) >= 2)
    assert np.all(weights.sum(axis=1) <= 5)
    cycle, votes = _election(weights, labels)
    assert cycle is not None and np.all(votes > 0)
    assert len(set(map(int, cycle))) == 2
    return dict(grid=a, body=body, depth=depth, substrate=substrate,
                signals=signals, period=period, panels=panels,
                panel_ids=panel_ids, labels=labels, areas=areas,
                local_counts=np.asarray(local_counts), circuits=circuit_records,
                terminals=terminal_records, weights=weights, cycle=cycle, votes=votes)


def _paint(info):
    a = info['grid']
    output = a.copy()
    field = (a == info['substrate']) | (a == CLOCK)
    colors = np.asarray(info['signals'], dtype=np.int8)[info['cycle']]
    output[field] = colors[info['depth'][field] % info['period']]
    for cells, owner in info['circuits']:
        color = info['signals'][info['labels'][owner]]
        for p in cells:
            output[p] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the footprint, clocks, panels and port votes from this input only."""
    return _paint(_inspect(input_grid))


def _witnesses(info):
    weights, labels = info['weights'], info['labels']
    _, panel_votes = _election((weights > 0).astype(int), labels)
    # Inspect phase-by-phase so a tie in one wrong election cannot mask a strict
    # contradiction in another phase.
    panel_wrong = any(v[0] != v[1] and int(np.argmax(v)) != int(info['cycle'][j])
                      for j, v in enumerate(panel_votes))
    body, d, p = info['body'], info['depth'], info['period']
    box = _box_depth(body)
    l1 = _erosion_depth(body, diagonal=False)
    field = (info['grid'] == info['substrate']) | (info['grid'] == CLOCK)
    band_box = bool(np.any(info['cycle'][d[field] % p] != info['cycle'][box[field] % p]))
    band_l1 = bool(np.any(info['cycle'][d[field] % p] != info['cycle'][l1[field] % p]))
    wrong_weights = np.zeros_like(weights)
    for pos, owner, _, _ in info['terminals']:
        wrong_weights[owner, int(box[pos]) % p] += 1
    _, box_votes = _election(wrong_weights, labels)
    address_wrong = any(v[0] != v[1] and int(np.argmax(v)) != int(info['cycle'][j])
                        for j, v in enumerate(box_votes))
    contact_wrong = False
    for cells, owner in info['circuits']:
        expected = info['signals'][labels[owner]]
        contact_wrong |= any(info['grid'][q] != expected
                             for q in _neighbors(cells, info['grid'].shape)
                             if info['panel_ids'][q] == owner)
    return dict(panel_wrong=panel_wrong, band_box=band_box, band_l1=band_l1,
                address_wrong=address_wrong, contact_wrong=contact_wrong)


def _topology(rng, period, evidence):
    for _ in range(256):
        count = int(rng.integers(3, 6))
        weights = np.zeros((count, period), dtype=int)
        for i in range(count):
            slots = rng.integers(period, size=int(rng.integers(2, 6)))
            weights[i] = np.bincount(slots, minlength=period)
        if np.any(weights.sum(axis=0) < 3):
            continue
        options = []
        for bits in range(1, (1 << count) - 1):
            labels = [(bits >> i) & 1 for i in range(count)]
            cycle, votes = _election(weights, labels)
            if cycle is None or np.any(votes == 0) or len(set(map(int, cycle))) != 2:
                continue
            _, unweighted = _election((weights > 0).astype(int), labels)
            witness = any(v[0] != v[1] and np.argmax(v) != cycle[j]
                          for j, v in enumerate(unweighted))
            if not evidence or witness:
                options.append((labels, cycle))
        if options:
            labels, cycle = options[int(rng.integers(len(options)))]
            return weights, labels, cycle
    raise RuntimeError('bounded vote-composition search exhausted')


def _body(rng, period, is_test, evidence):
    lo = 23 if period == 3 else 20
    if is_test:
        lo = max(lo, 25)
    h, w = map(int, rng.integers(lo, 31, size=2))
    if rng.random() < .12:
        h = w = 30
    mask = np.zeros((h, w), dtype=bool)
    top, bottom, left, right = map(int, rng.integers(0, 3, size=4))
    if top + bottom + left + right == 0:
        top = 1
    mask[top:h - bottom if bottom else h, left:w - right if right else w] = True
    # Keep an inscribed central core; bites and holes can still split deeper
    # contour layers into several pieces without disconnecting the footprint.
    cr, cc = h // 2, w // 2
    core = np.zeros_like(mask)
    radius = period + (1 if is_test else 0)
    core[cr - radius:cr + radius + 1, cc - radius:cc + radius + 1] = True
    for _ in range(int(rng.integers(1 if evidence else 0, 5))):
        candidate = mask.copy()
        side = int(rng.integers(4))
        extent = int(rng.integers(1, 5))
        if side < 2:
            start = int(rng.integers(left, w - right - 2))
            end = min(w - right, start + int(rng.integers(2, 8)))
            if side == 0:
                candidate[top:top + extent, start:end] = False
            else:
                candidate[h - bottom - extent:h - bottom, start:end] = False
        else:
            start = int(rng.integers(top, h - bottom - 2))
            end = min(h - bottom, start + int(rng.integers(2, 8)))
            if side == 2:
                candidate[start:end, left:left + extent] = False
            else:
                candidate[start:end, w - right - extent:w - right] = False
        if not np.any(core & ~candidate) and len(_components(candidate)[0]) == 1:
            mask = candidate
    if rng.random() < (.55 if is_test else .25):
        for _ in range(12):
            candidate = mask.copy()
            r, c = int(rng.integers(top + 2, h - bottom - 3)), int(rng.integers(left + 2, w - right - 3))
            rh, rw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            candidate[r:r + rh, c:c + rw] = False
            if not np.any(core & ~candidate) and len(_components(candidate)[0]) == 1:
                mask = candidate
                break
    depth = _bfs_depth(mask)
    assert int(depth.max()) >= (2 * period if is_test else period + 1)
    return mask, depth


def _panels(rng, mask, reserved, count):
    used = set(reserved)
    panels = []
    positions = [tuple(map(int, p)) for p in np.argwhere(mask)]
    dimensions = ((1, 3), (2, 2), (2, 3), (3, 2), (2, 4), (3, 3), (3, 4))
    for _ in range(count):
        for _attempt in range(100):
            rh, rw = dimensions[int(rng.integers(len(dimensions)))]
            r, c = positions[int(rng.integers(len(positions)))]
            cells = {(a, b) for a in range(r, r + rh) for b in range(c, c + rw)}
            if any(a >= mask.shape[0] or b >= mask.shape[1] or not mask[a, b]
                   for a, b in cells) or cells & used:
                continue
            panels.append(cells)
            used |= cells | _neighbors(cells, mask.shape)
            break
        else:
            return None
    return panels


def _path(start, targets, allowed, costs):
    h, w = allowed.shape
    opened = allowed.ravel().tolist()
    cost_values = costs.ravel().tolist() if isinstance(costs, np.ndarray) else costs
    start_id = start[0] * w + start[1]
    target_ids = {r * w + c for r, c in targets}
    distances = [float('inf')] * (h * w)
    distances[start_id] = 0.0
    parents = [-1] * (h * w)
    heap = [(0.0, *start)]
    while heap:
        distance, r, c = heapq.heappop(heap)
        pos = r * w + c
        if distance != distances[pos]:
            continue
        if pos in target_ids:
            path = {divmod(pos, w)}
            while pos != start_id:
                pos = parents[pos]
                path.add(divmod(pos, w))
            return path
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbor = nr * w + nc
                if opened[neighbor]:
                    new = distance + cost_values[neighbor]
                    if new < distances[neighbor]:
                        distances[neighbor] = new
                        parents[neighbor] = pos
                        heapq.heappush(heap, (new, nr, nc))
    return None


def _networks(rng, mask, depth, clock_reserve, panels, weights, is_test):
    p = weights.shape[1]
    all_panels = set().union(*panels)
    wires, tips = [set() for _ in panels], [set() for _ in panels]
    # A single selected tip explicitly exercises wrapping. All other endpoints
    # remain sampled; no geometric position announces the election winner.
    forced_options = [(i, j, int(d)) for i in range(len(panels)) for j in range(p)
                      if weights[i, j] for d in np.unique(depth[mask])
                      if d % p == j and (d >= 2 * p if is_test else p <= d <= p + 1)]
    if not forced_options:
        return None
    forced_owner, forced_phase, forced_depth = forced_options[int(rng.integers(len(forced_options)))]
    order = [forced_owner] + [int(i) for i in rng.permutation(len(panels)) if i != forced_owner]
    for owner in order:
        other_panels = all_panels - panels[owner]
        other_networks = set().union(*(wires[i] | tips[i] for i in range(len(panels)) if i != owner))
        blocked = (all_panels | clock_reserve | other_networks
                   | _neighbors(other_panels | other_networks, mask.shape))
        phases = [j for j in range(p) for _ in range(int(weights[owner, j]))]
        rng.shuffle(phases)
        if owner == forced_owner:
            phases.remove(forced_phase)
            phases.insert(0, forced_phase)
        for ordinal, phase in enumerate(phases):
            own_tips = tips[owner]
            unavailable = blocked | own_tips | _neighbors(own_tips, mask.shape)
            allowed = mask.copy()
            for pos in unavailable:
                allowed[pos] = False
            targets = (wires[owner] | _neighbors(panels[owner], mask.shape))
            targets = {pos for pos in targets if allowed[pos]}
            if not targets:
                return None
            tip_blocked = (all_panels | set().union(*wires) | set().union(*tips))
            tip_blocked |= _neighbors(tip_blocked, mask.shape)
            eligible = allowed & (depth % p == phase)
            if not is_test:
                eligible &= depth <= p + 1
            if owner == forced_owner and ordinal == 0:
                eligible &= depth == forced_depth
            candidates = [(int(r), int(c)) for r, c in np.argwhere(eligible)
                          if (int(r), int(c)) not in tip_blocked]
            if not candidates:
                return None
            # Nearby terminals and genuinely long branches coexist. A random
            # cost field varies path bends without adding disconnected spurs.
            center_r, center_c = map(float, np.mean(np.asarray(sorted(panels[owner])), axis=0))
            rng.shuffle(candidates)
            reach = float(rng.choice((.12, .4, 1.0)))
            candidates.sort(key=lambda q: reach * (abs(q[0] - center_r) + abs(q[1] - center_c)) + rng.random() * 7)
            costs = (1 + rng.random(mask.shape) * float(rng.choice((.1, 2., 6.)))).ravel().tolist()
            accepted = False
            for tip in candidates[:16]:
                adjacent = list(_neighbors([tip], mask.shape))
                rng.shuffle(adjacent)
                for start in adjacent:
                    if not allowed[start]:
                        continue
                    work = allowed.copy()
                    work[tip] = False
                    for pos in adjacent:
                        if pos != start:
                            work[pos] = False
                    path = _path(start, targets, work, costs)
                    if path is not None:
                        wires[owner] |= path
                        tips[owner].add(tip)
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                return None
    return wires, tips


def _pattern(rng, cells, majority, minority, minority_contact=None):
    cells = sorted(cells)
    area = len(cells)
    nminor = int(rng.choice((1, max(1, (area - 1) // 3), (area - 1) // 2)))
    order = rng.permutation(area).tolist()
    if minority_contact is not None:
        chosen = cells.index(minority_contact)
        order.remove(chosen)
        order.insert(0, chosen)
    minority_cells = {cells[i] for i in order[:nminor]}
    return {pos: minority if pos in minority_cells else majority for pos in cells}


def _make_pair(rng, period, is_test, evidence):
    # At most 96 candidates, with no persistent retries or retained candidates.
    for _attempt in range(96):
        try:
            weights, labels, cycle = _topology(rng, period, evidence)
            mask, depth = _body(rng, period, is_test, evidence)
        except (AssertionError, RuntimeError):
            continue
        paths = _clock_paths(mask, depth, period)
        if not paths:
            continue
        clock = set(paths[int(rng.integers(len(paths)))])
        clock_reserve = clock | _neighbors(clock, mask.shape)
        panels = _panels(rng, mask, clock_reserve, len(labels))
        if panels is None:
            continue
        networks = _networks(rng, mask, depth, clock_reserve, panels, weights, is_test)
        if networks is None:
            continue
        wires, tips = networks
        substrate, s0, s1 = map(int, rng.choice(PALETTE, size=3, replace=False))
        signals = sorted((s0, s1))
        inp = np.zeros(mask.shape, dtype=np.int8)
        inp[mask] = substrate
        output = inp.copy()
        output[mask] = np.asarray(signals)[cycle[depth[mask] % period]]
        for pos in clock:
            inp[pos] = CLOCK
        for i, cells in enumerate(panels):
            contact = sorted(cells & _neighbors(wires[i], mask.shape))
            if not contact:
                break
            pattern = _pattern(rng, cells, signals[labels[i]], signals[1 - labels[i]],
                               contact[int(rng.integers(len(contact)))])
            for pos, color in pattern.items():
                inp[pos] = output[pos] = color
            for pos in wires[i]:
                inp[pos] = WIRE
                output[pos] = signals[labels[i]]
            for pos in tips[i]:
                inp[pos] = TERMINAL
                output[pos] = signals[labels[i]]
        else:
            try:
                info = _inspect(inp)
                assert np.array_equal(output, _paint(info))
                features = _witnesses(info)
                if evidence:
                    assert all(features.values())
                return {'input': inp, 'output': output}
            except AssertionError:
                continue
    raise RuntimeError('bounded contour-parliament construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    periods = list(map(int, rng.permutation((2, 3))))
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            period = periods[i] if split == 'train' and i < 2 else int(rng.integers(2, 4))
            for _ in range(4):
                pair = _make_pair(rng, period, split == 'test', split == 'train' and i == 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == output.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _inspect(inp)
            assert np.array_equal(output, _paint(info))
            assert np.array_equal(output != BACKGROUND, info['body'])
            panel_mask = info['panel_ids'] >= 0
            assert np.array_equal(output[panel_mask], inp[panel_mask])
            assert set(map(int, np.unique(output))) == {BACKGROUND, *info['signals']}
            assert np.array_equal(info['depth'], _bfs_depth(info['body']))
            tip_depths = [d for _, _, d, _ in info['terminals']]
            if split == 'train':
                assert max(tip_depths) <= info['period'] + 1
                assert max(tip_depths) >= info['period']
            else:
                assert max(tip_depths) >= 2 * info['period']
            analyses[split].append(info)
    assert {info['period'] for info in analyses['train']} == {2, 3}
    assert any(all(_witnesses(info).values()) for info in analyses['train'])
    return None


def self_test():
    """42 specified seeds, geometric cross-checks, ablations and API boundaries.

    Only small sets of hashes/measurements are retained, never all 42 puzzles.
    The returned summary is optional diagnostic data, not a saved trace.
    """
    shapes, geometries, palettes, weights, periods, panel_counts = (set() for _ in range(6))
    input_hashes, clock_directions, max_depths, terminal_counts = set(), set(), set(), set()
    holes = nonrectangles = forks = address_witnesses = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _puzzle_hash(puzzle) == _puzzle_hash(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                pairs += 1
                shapes.add(info['grid'].shape)
                geometries.add(hashlib.sha256(info['body'].tobytes()).hexdigest())
                palettes.add((info['substrate'], *info['signals']))
                weights.add(tuple(sorted(tuple(map(int, row)) for row in info['weights'])))
                periods.add(info['period'])
                panel_counts.add(len(info['panels']))
                input_hashes.add(hashlib.sha256(info['grid'].tobytes()).hexdigest())
                max_depths.add(int(info['depth'].max()))
                terminal_counts.add(len(info['terminals']))
                rr, cc = np.nonzero(info['body'])
                nonrectangles += len(rr) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
                background_comps, _ = _components(~info['body'])
                h, w = info['grid'].shape
                holes += any(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells)
                             for cells in background_comps)
                clock = sorted((int(info['depth'][r, c]), int(r), int(c))
                               for r, c in np.argwhere(info['grid'] == CLOCK))
                clock_directions.add((clock[1][1] - clock[0][1], clock[1][2] - clock[0][2]))
                for cells, _ in info['circuits']:
                    cellset = set(cells)
                    forks += any(len(_neighbors([pos], info['grid'].shape) & cellset) >= 3 for pos in cells)
                address_witnesses += _witnesses(info)['address_wrong']
    assert pairs == 210 and len(input_hashes) == pairs
    assert periods == {2, 3} and panel_counts == {3, 4, 5}
    assert len(shapes) >= 35 and len(geometries) >= 100 and len(palettes) >= 40
    assert len(weights) >= 40 and len(terminal_counts) >= 6 and len(max_depths) >= 4
    assert len(clock_directions) == 4 and holes >= 10 and nonrectangles >= 100
    assert forks >= 100 and address_witnesses >= 42
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(7301, ntrain, ntest)
            assert len(puzzle['train']) == ntrain and len(puzzle['test']) == ntest
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                         (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(0, 2, 1)
    # These metamorphic checks exercise solve on valid members outside the
    # constructor's particular path/shape/color samples.
    for pair in original['train'] + original['test']:
        for turns in range(4):
            for reflect in (False, True):
                inp = np.rot90(pair['input'], turns)
                expected = np.rot90(pair['output'], turns)
                if reflect:
                    inp, expected = inp[:, ::-1], expected[:, ::-1]
                assert np.array_equal(solve(inp), expected)
        mapping = np.arange(10, dtype=np.int8)
        mapping[list(PALETTE)] = list(reversed(PALETTE))
        assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
        info = _inspect(pair['input'])
        inp, expected = pair['input'].copy(), pair['output'].copy()
        cells = sorted(info['panels'][0])
        values = [int(inp[pos]) for pos in cells][::-1]
        for pos, value in zip(cells, values):
            inp[pos] = expected[pos] = value
        assert np.array_equal(solve(inp), expected)
    changed = {s: [{k: np.array(v, copy=True) for k, v in pair.items()}
                   for pair in original[s]] for s in original}
    info = _inspect(changed['test'][0]['input'])
    field = (info['grid'] == info['substrate'])
    pos = tuple(np.argwhere(field)[0])
    old = int(changed['test'][0]['output'][pos])
    changed['test'][0]['output'][pos] = next(s for s in info['signals'] if s != old)
    try:
        validate_generated_puzzle(changed)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong contour color accepted')
    return {'seeds': len(HOST_SEEDS), 'pairs': pairs, 'shapes': len(shapes),
            'footprints': len(geometries), 'palettes': len(palettes),
            'weight_profiles': len(weights), 'terminal_counts': sorted(terminal_counts),
            'max_depths': sorted(max_depths), 'clock_directions': sorted(clock_directions),
            'with_holes': int(holes), 'nonrectangles': int(nonrectangles),
            'branching_circuits': int(forks), 'address_witnesses': int(address_witnesses)}


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
