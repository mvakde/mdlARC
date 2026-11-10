"""Folded electorates: one ballot per reflection orbit, then one per wire port.

The rectangle frames define centered folds. Background-colored holes are erased
ballots only inside frames. All roles are recovered geometrically from the input.
"""
from __future__ import annotations

import heapq
import numbers
from collections import Counter
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.add(q)
        result.append(cells)
    return result


def _rect(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _neighbors(cells, h, w):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _orbits(h, w):
    # Frames fix both folds, including through-cell axes for odd dimensions.
    return [tuple(sorted({(r, c), (h - 1 - r, c), (r, w - 1 - c),
                          (h - 1 - r, w - 1 - c)}))
            for r in range((h + 1) // 2) for c in range((w + 1) // 2)]


def _hub_windows(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    a = np.ones((h - 2, w - 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            a &= mask[dr:dr + h - 2, dc:dc + w - 2]
    return list(map(tuple, np.argwhere(a)))


def _ports(r, c):
    return [(r - 1, c + j) for j in range(3)] + \
           [(r + 3, c + j) for j in range(3)] + \
           [(r + j, c - 1) for j in range(3)] + \
           [(r + j, c + 3) for j in range(3)]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            np.any(a < 0) or np.any(a > 9)):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=True)


def _interpret(a, frame, boxes, rims, wire, hub):
    h, w = a.shape
    r, c = hub
    assert 0 < r < h - 3 and 0 < c < w - 3
    occupied = set().union(*(_rect(*box) for box in boxes))
    hub_cells = _rect(r, c, 3, 3)
    assert not occupied & hub_cells
    exterior = np.ones(a.shape, dtype=bool)
    for p in occupied | hub_cells:
        exterior[p] = False
    exterior[a == wire] = False
    backgrounds = np.unique(a[exterior])
    assert len(backgrounds) == 1
    bg = int(backgrounds[0])
    assert bg not in (frame, wire)
    signals = tuple(int(x) for x in np.unique(a) if x not in (frame, wire, bg))
    assert len(signals) == 2
    completed = a.copy()
    panels = []
    for box in boxes:
        br, bc, bh, bw = box
        block = a[br + 1:br + bh - 1, bc + 1:bc + bw - 1]
        assert set(map(int, np.unique(block))) <= {bg, *signals}
        assert np.any(block == bg)
        restored = block.copy()
        votes = []
        sizes, missing, diagonal, triples = [], 0, 0, 0
        for orbit in _orbits(*block.shape):
            witnesses = {int(block[p]) for p in orbit if block[p] != bg}
            assert len(witnesses) == 1, 'empty or contradictory ballot orbit'
            color = next(iter(witnesses))
            votes.append(color)
            sizes.append(len(orbit))
            triples += int(len(orbit) == 4 and sum(block[p] == bg for p in orbit) == 3)
            for y, x in orbit:
                if block[y, x] == bg:
                    missing += 1
                    ry, rx = block.shape[0] - 1 - y, block.shape[1] - 1 - x
                    diagonal += int(block[ry, x] == bg and block[y, rx] == bg
                                    and block[ry, rx] != bg)
                    restored[y, x] = color
        counts = [votes.count(s) for s in signals]
        assert min(counts) > 0 and counts[0] != counts[1]
        winner = signals[int(counts[1] > counts[0])]
        completed[br + 1:br + bh - 1, bc + 1:bc + bw - 1] = restored
        raw_counts = [int(np.sum(restored == s)) for s in signals]
        visible_counts = [int(np.sum(block == s)) for s in signals]
        panels.append(dict(box=box, winner=winner, votes=counts,
                           raw_counts=raw_counts, visible_counts=visible_counts,
                           orbit_sizes=sizes, missing=missing, diagonal=diagonal,
                           triples=triples,
                           clean=restored))
    route_mask = a == wire
    route_mask[r:r + 3, c:c + 3] = False
    routes = _components(route_mask)
    assert routes
    owner_map = {}
    for i, rim in enumerate(rims):
        for p in _neighbors(rim, h, w) - occupied:
            owner_map.setdefault(p, set()).add(i)
    route_owners, position_owner = [], {}
    ports = set(_ports(r, c))
    weights = [0] * len(boxes)
    for route in routes:
        owners = set().union(*(owner_map.get(p, set()) for p in route))
        assert len(owners) == 1, 'each wire component must touch one frame'
        owner = next(iter(owners))
        touches = route & ports
        assert touches
        weights[owner] += len(touches)
        route_owners.append(owner)
        for p in route:
            position_owner[p] = owner
    assert all(1 <= n <= 4 for n in weights)
    side_counts = []
    port_votes = []
    all_ports = _ports(r, c)
    for side in range(4):
        n = 0
        for p in all_ports[side * 3:side * 3 + 3]:
            assert int(a[p]) in (wire, bg)
            if a[p] == wire:
                n += 1
                port_votes.append(panels[position_owner[p]]['winner'])
        side_counts.append(n)
    assert max(side_counts) <= 2 and 3 <= len(port_votes) <= 8
    counts = [port_votes.count(s) for s in signals]
    assert min(counts) > 0 and counts[0] != counts[1]
    winner = signals[int(counts[1] > counts[0])]
    for route, owner in zip(routes, route_owners):
        for p in route:
            completed[p] = panels[owner]['winner']
    completed[r + 1, c + 1] = winner
    return dict(background=bg, frame=frame, wire=wire, signals=signals, hub=hub,
                panels=panels, rims=rims, routes=routes, owners=route_owners,
                weights=weights, side_counts=side_counts, votes=counts,
                winner=winner, output=completed)


def _analyze(input_grid):
    a = _grid(input_grid)
    assert len(np.unique(a)) == 5, 'five geometric color roles required'
    interpretations = []
    for frame in map(int, np.unique(a)):
        rims = _components(a == frame)
        if not 2 <= len(rims) <= 4:
            continue
        boxes = []
        for rim in rims:
            rs, cs = zip(*rim)
            r, c = min(rs), min(cs)
            h, w = max(rs) - r + 1, max(cs) - c + 1
            if min(h, w) < 6 or rim != _rect(r, c, h, w) - _rect(r + 1, c + 1, h - 2, w - 2):
                break
            boxes.append((r, c, h, w))
        else:
            for wire in map(int, np.unique(a)):
                if wire == frame:
                    continue
                hubs = _hub_windows(a == wire)
                if len(hubs) != 1:
                    continue
                try:
                    interpretations.append(_interpret(a, frame, boxes, rims, wire, hubs[0]))
                except AssertionError:
                    continue
    assert len(interpretations) == 1, 'input must have a unique circuit interpretation'
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive frames, orbit ballots, wire ownership and port winner from input."""
    try:
        return _analyze(input_grid)['output'].copy()
    except AssertionError as error:
        raise ValueError(str(error)) from error


def _pattern(rng, h, w, winner, loser, style, inverted=False):
    orbits = _orbits(h, w)
    n = len(orbits)
    if inverted:
        # The smallest orbits form a strict majority of independent ballots;
        # larger minority orbits can nevertheless occupy more physical pixels.
        order = sorted(range(n), key=lambda i: (len(orbits[i]), float(rng.random())))
        nwin = n // 2 + 1
    else:
        nwin = int(rng.integers(n // 2 + 1, n))
        anchors = np.array([o[0] for o in orbits])
        rr, cc = anchors.T
        if style == 0:
            order = list(map(int, rng.permutation(n)))
        elif style == 1:
            order = list(np.argsort(rr * (w + 1) + cc + rng.random(n) * .5))
        elif style == 2:
            order = list(np.argsort(abs(rr - int(rng.integers((h + 1) // 2))) +
                                    abs(cc - int(rng.integers((w + 1) // 2))) + rng.random(n) * .3))
        else:
            order = list(np.argsort((rr + cc) % 2 * n + rng.random(n)))
    selected = set(order[:nwin])
    clean = np.empty((h, w), dtype=np.int8)
    for i, orbit in enumerate(orbits):
        for p in orbit:
            clean[p] = winner if i in selected else loser
    if inverted:
        assert np.sum(clean == winner) < np.sum(clean == loser)
    return clean


def _mask(rng, clean, level, hard, preserve_inversion=False, majority=None):
    h, w = clean.shape
    mask = np.zeros((h, w), dtype=bool)
    family = int(rng.integers(5))
    if not preserve_inversion:
        if family == 0:
            mask = rng.random((h, w)) < (.12 if level == 0 else .30)
        else:
            for _ in range(1 if level == 0 else 2):
                r, c = int(rng.integers(h)), int(rng.integers(w))
                dh, dw = int(rng.integers(1, h // 2 + 1)), int(rng.integers(1, w // 2 + 1))
                if family == 2:
                    dh = 1
                if family == 3:
                    dw = 1
                if family == 4:
                    for dr in range(dh):
                        mask[r + dr:min(h, r + dr + 1), c:min(w, c + 1 + dr)] = True
                else:
                    mask[r:min(h, r + dh), c:min(w, c + dw)] = True
    orbits = _orbits(h, w)
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
        # Training has at most one triple-erased orbit per panel, added below.
        # Tests combine two such cases inside one electorate.
        if level == 0:
            erased = [p for p in orbit if mask[p]]
            while len(erased) > 2:
                p = erased.pop(int(rng.integers(len(erased))))
                mask[p] = False
    if preserve_inversion:
        choices = [o for o in orbits if len(o) > 1 and clean[o[0]] == majority]
        orbit = choices[int(rng.integers(len(choices)))]
        mask[orbit[int(rng.integers(len(orbit)))]] = True
    if hard:
        four = [o for o in orbits if len(o) == 4]
        for i in rng.choice(len(four), size=min(len(four), 1 + level), replace=False):
            orbit = four[int(i)]
            survivor = orbit[int(rng.integers(4))]
            for p in orbit:
                mask[p] = p != survivor
    if not mask.any():
        choices = [o for o in orbits if len(o) > 1]
        orbit = choices[int(rng.integers(len(choices)))]
        mask[orbit[int(rng.integers(len(orbit)))]] = True
    return mask


def _path(rng, start, targets, allowed, roughness):
    h, w = allowed.shape
    if not allowed[start]:
        return None
    costs = 1 + rng.random((h, w)) * roughness
    if roughness > 2:
        coarse = rng.random(((h + 3) // 4, (w + 3) // 4)) * roughness * 2
        costs += np.repeat(np.repeat(coarse, 4, 0), 4, 1)[:h, :w]
    distances, previous = {start: 0.0}, {}
    heap = [(0.0, start)]
    while heap:
        distance, p = heapq.heappop(heap)
        if distance != distances[p]:
            continue
        if p in targets:
            route = {p}
            while p != start:
                p = previous[p]
                route.add(p)
            return route
        for dr, dc in DIRS:
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                d = distance + float(costs[q])
                if d < distances.get(q, float('inf')):
                    distances[q], previous[q] = d, p
                    heapq.heappush(heap, (d, q))
    return None


def _layouts(rng, h, w, count, witness):
    cr, cc = int(rng.integers(4, h - 6)), int(rng.integers(4, w - 6))
    hub = cr, cc
    reserved = _rect(cr - 2, cc - 2, 7, 7)
    boxes = []
    for i in range(count):
        for _ in range(55):
            if witness and i == 0:
                ph, pw = int(rng.choice([5, 7])), int(rng.choice([5, 7]))
            else:
                ph, pw = int(rng.integers(4, 9)), int(rng.integers(4, 9))
            bh, bw = ph + 2, pw + 2
            side = int(rng.integers(4))
            pad = int(rng.integers(2)) if rng.random() < .28 else 0
            if side < 2:
                r = pad if side == 0 else h - bh - pad
                c = int(rng.integers(w - bw + 1))
            else:
                c = pad if side == 2 else w - bw - pad
                r = int(rng.integers(h - bh + 1))
            cells = _rect(r, c, bh, bw)
            if cells & reserved:
                continue
            boxes.append((r, c, bh, bw, i == 0 and witness))
            reserved |= cells | _neighbors(cells, h, w)
            break
        else:
            return None
    boxes.sort(key=lambda b: float(np.arctan2(b[0] + (b[2] - 1) / 2 - cr - 1,
                                             b[1] + (b[3] - 1) / 2 - cc - 1)))
    return hub, boxes


def _topology(rng, boxes, witness):
    n = len(boxes)
    if witness:
        special = next(i for i, b in enumerate(boxes) if b[4])
        weights = [3 if i == special else 1 for i in range(n)]
        winners = {special}
    else:
        options = []
        # Choose actual weights and labels jointly, retaining strict results.
        for _ in range(30):
            weights = list(map(int, rng.integers(1, 4, size=n)))
            if not 3 <= sum(weights) <= 8:
                continue
            for bits in range(1, (1 << n) - 1):
                winners = {i for i in range(n) if bits >> i & 1}
                if sum(weights[i] for i in winners) > sum(weights) / 2:
                    options.append((weights, winners))
        if not options:
            return None
        weights, winners = options[int(rng.integers(len(options)))]
    return weights, winners


def _wires(rng, shape, hub, boxes, weights, roughness):
    h, w = shape
    r, c = hub
    ring = [(r - 1, c), (r - 1, c + 2), (r, c + 3), (r + 2, c + 3),
            (r + 3, c + 2), (r + 3, c), (r + 2, c - 1), (r, c - 1)]
    chosen = sorted(map(int, rng.choice(8, size=sum(weights), replace=False)))
    candidates = []
    for shift in range(len(chosen)):
        sequence = chosen[shift:] + chosen[:shift]
        groups, index, score = [], 0, 0.0
        for box, weight in zip(boxes, weights):
            angle = float(np.arctan2(box[0] + (box[2] - 1) / 2 - r - 1,
                                     box[1] + (box[3] - 1) / 2 - c - 1))
            group = {ring[j] for j in sequence[index:index + weight]}
            groups.append(group)
            index += weight
            for p in group:
                d = abs(angle - float(np.arctan2(p[0] - r - 1, p[1] - c - 1)))
                score += min(d, 2 * np.pi - d) ** 2
        candidates.append((score + float(rng.random()) * .4, groups))
    groups = min(candidates, key=lambda x: x[0])[1]
    # A middle port can replace a singleton or one end of a same-owner pair.
    for side in range(4):
        p0, p1, p2 = _ports(r, c)[side * 3:side * 3 + 3]
        touched = [i for i, group in enumerate(groups) if p0 in group or p2 in group]
        if len(touched) == 1 and rng.random() < .3:
            group = groups[touched[0]]
            old = p0 if p0 in group else p2
            group.remove(old)
            group.add(p1)
    hub_cells = _rect(r, c, 3, 3)
    collar = _rect(r - 1, c - 1, 5, 5) - hub_cells
    panel_cells = [_rect(*b[:4]) for b in boxes]
    all_panels = set().union(*panel_cells)
    networks = [set(group) for group in groups]
    for owner in sorted(range(len(boxes)), key=lambda i: -weights[i]):
        others = set().union(*(net for i, net in enumerate(networks) if i != owner))
        other_panels = all_panels - panel_cells[owner]
        blocked = hub_cells | collar | all_panels | others | _neighbors(others | other_panels, h, w)
        blocked -= groups[owner]
        allowed = np.ones((h, w), dtype=bool)
        for p in blocked:
            allowed[p] = False
        border = _neighbors(panel_cells[owner], h, w) - all_panels
        border = {p for p in border if allowed[p]}
        if not border:
            return None
        net = set()
        for port in sorted(groups[owner]):
            path = _path(rng, port, border | net, allowed, roughness)
            if path is None:
                return None
            net |= path
        if net & collar != groups[owner]:
            return None
        networks[owner] = net
    mask = np.zeros((h, w), dtype=bool)
    for p in hub_cells | set().union(*networks):
        mask[p] = True
    if _hub_windows(mask) != [hub]:
        return None
    return networks


def _make_pair(rng, roles, winner_index, bounds, count, witness, hard, level, roughness):
    bg, frame, wire, s0, s1 = roles
    signals = sorted((s0, s1))
    winner, loser = signals[winner_index], signals[1 - winner_index]
    for _ in range(220):
        h, w = map(int, rng.integers(bounds[0], bounds[1] + 1, size=2))
        if rng.random() < .2:
            w = h
        layout = _layouts(rng, h, w, count, witness)
        if layout is None:
            continue
        hub, boxes = layout
        topology = _topology(rng, boxes, witness)
        if topology is None:
            continue
        weights, winners = topology
        networks = _wires(rng, (h, w), hub, boxes, weights, roughness)
        if networks is None:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        out = inp.copy()
        r, c = hub
        inp[r:r + 3, c:c + 3] = wire
        out[r:r + 3, c:c + 3] = wire
        hard_owner = next((i for i, b in enumerate(boxes) if not b[4]), 0)
        for i, (br, bc, bh, bw, special) in enumerate(boxes):
            major, minor = (winner, loser) if i in winners else (loser, winner)
            clean = _pattern(rng, bh - 2, bw - 2, major, minor,
                             int(rng.integers(4)), special)
            mask = _mask(rng, clean, level, hard and i == hard_owner,
                         special, major)
            observed = clean.copy()
            observed[mask] = bg
            inp[br:br + bh, bc:bc + bw] = frame
            out[br:br + bh, bc:bc + bw] = frame
            inp[br + 1:br + bh - 1, bc + 1:bc + bw - 1] = observed
            out[br + 1:br + bh - 1, bc + 1:bc + bw - 1] = clean
            for p in networks[i]:
                inp[p], out[p] = wire, major
        out[r + 1, c + 1] = winner
        # Labels are constructed above, not obtained by calling solve.
        try:
            info = _analyze(inp)
            assert np.array_equal(info['output'], out)
            if witness:
                m = _metrics(info)
                assert m['panel_count_wrong'] and m['pixel_rule_wrong']
                assert m['visible_rule_wrong'] and m['raw_total_wrong']
                assert m['component_rule_wrong']
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    mode = int(rng.integers(3))
    stable = bool(rng.random() < .3)
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    evidence = int(rng.integers(nt))
    winners = list(map(int, rng.integers(2, size=nt)))
    if len(set(winners)) == 1:
        winners[(evidence + 1) % nt] ^= 1
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', ne)):
        for i in range(n):
            witness = split == 'train' and i == evidence
            roles = palette if stable else list(map(int, rng.choice(10, 5, replace=False)))
            wi = winners[i] if split == 'train' else int(rng.integers(2))
            if split == 'train':
                count = 3 if witness else int(rng.choice([2, 3]))
                bounds = (23, 26) if mode == 1 else (21, 30)
                if count == 2 and not witness and mode != 1:
                    bounds = (18, 28)
                level, hard = 0, witness or bool(rng.random() < .35)
                roughness = float(rng.choice([0., .8, 3., 6.]))
            else:
                count = 4 if mode == 2 else int(rng.choice([2, 3, 4]))
                bounds = (27, 30) if mode == 1 or count == 4 else (22, 30)
                level, hard = 1, True
                roughness = float(rng.choice([.8, 3., 6., 9.]))
            pair = None
            for _ in range(12):
                candidate = _make_pair(rng, roles, wi, bounds, count, witness, hard, level, roughness)
                if candidate is not None and _signature(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError('bounded circuit packing exhausted')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _metrics(info):
    signals = info['signals']
    winner = info['winner']
    def result(counts):
        if counts[0] == counts[1]:
            return None
        return signals[int(counts[1] > counts[0])]
    panels = info['panels']
    panel_vote = result([sum(p['winner'] == s for p in panels) for s in signals])
    raw = result([sum(p['raw_counts'][j] for p in panels) for j in range(2)])
    def substitute(name):
        labels = [result(p[name]) for p in panels]
        if None in labels:
            return None
        return result([sum(weight for weight, label in zip(info['weights'], labels) if label == s)
                       for s in signals])
    pixel, visible = substitute('raw_counts'), substitute('visible_counts')
    component_vote = result([sum(panels[owner]['winner'] == s for owner in info['owners'])
                             for s in signals])
    return dict(panel_count=len(panels), ports=sum(info['weights']),
                weights=tuple(sorted(info['weights'])),
                panel_count_wrong=panel_vote is not None and panel_vote != winner,
                raw_total_wrong=raw is not None and raw != winner,
                pixel_rule_wrong=pixel is not None and pixel != winner,
                visible_rule_wrong=visible is not None and visible != winner,
                component_rule_wrong=component_vote is not None and component_vote != winner,
                local_pixel_disagreements=sum(result(p['raw_counts']) not in (None, p['winner']) for p in panels),
                diagonal=sum(p['diagonal'] for p in panels),
                triples=tuple(p['triples'] for p in panels),
                missing=sum(p['missing'] for p in panels),
                parities=tuple((p['clean'].shape[0] % 2, p['clean'].shape[1] % 2) for p in panels),
                dimensions=tuple(p['clean'].shape for p in panels),
                vote_margin=abs(info['votes'][0] - info['votes'][1]),
                wire_cells=sum(map(len, info['routes'])))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, evidence, winner_indices = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(out, info['output'])
            bg, frame, wire = info['background'], info['frame'], info['wire']
            # Family-specific preservation and reconstruction, rather than only
            # checking that solve agrees with the generator.
            assert np.array_equal(out[inp == frame], inp[inp == frame])
            changed = inp != out
            assert changed.any()
            assert not np.any(changed & (inp != bg) & (inp != wire))
            for panel in info['panels']:
                r, c, h, w = panel['box']
                block = out[r + 1:r + h - 1, c + 1:c + w - 1]
                assert not np.any(block == bg)
                assert np.array_equal(block, block[::-1])
                assert np.array_equal(block, block[:, ::-1])
                original = inp[r + 1:r + h - 1, c + 1:c + w - 1]
                assert np.array_equal(block[original != bg], original[original != bg])
            hub_r, hub_c = info['hub']
            border = out[hub_r:hub_r + 3, hub_c:hub_c + 3].copy()
            assert border[1, 1] == info['winner']
            border[1, 1] = wire
            assert np.all(border == wire)
            m = _metrics(info)
            if split == 'train':
                assert max(m['triples']) <= 1
                evidence.append(m)
                winner_indices.add(info['signals'].index(info['winner']))
            else:
                assert m['diagonal'] > 0
                assert max(m['triples']) >= 2
    assert winner_indices == {0, 1}
    assert any(all(m[k] for k in ('panel_count_wrong', 'pixel_rule_wrong',
                                  'visible_rule_wrong', 'raw_total_wrong',
                                  'component_rule_wrong')) for m in evidence)
    assert any(m['diagonal'] > 0 for m in evidence)


def self_test() -> dict:
    """Bounded 42-seed study; measurements are generator-owned, not certification."""
    shapes, palettes, structures, dimensions, weights = set(), set(), set(), set(), set()
    parities, counts, ports, margins = set(), set(), set(), set()
    rows, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_signature = tuple(_signature(p['input']) for s in puzzle for p in puzzle[s])
        assert puzzle_signature not in signatures
        signatures.add(puzzle_signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                info = _analyze(inp)
                assert np.array_equal(solve(inp), pair['output'])
                m = _metrics(info)
                rows.append(m)
                shapes.add(inp.shape)
                palettes.add((info['background'], info['frame'], info['wire'], *info['signals']))
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
                dimensions.update(m['dimensions'])
                parities.update(m['parities'])
                counts.add(m['panel_count'])
                ports.add(m['ports'])
                margins.add(m['vote_margin'])
                weights.add(m['weights'])
    assert len(shapes) >= 35 and len(structures) == 210
    assert len(palettes) >= 50 and len(dimensions) >= 16
    assert parities == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert counts == {2, 3, 4} and ports == set(range(3, 9))
    assert len(weights) >= 8 and len(margins) >= 4
    assert sum(m['pixel_rule_wrong'] and m['panel_count_wrong'] for m in rows) >= 42
    assert sum(m['component_rule_wrong'] for m in rows) >= 42
    assert sum(m['diagonal'] > 0 for m in rows) >= 84
    # Determinism and all legal explicit count combinations.
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7000 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Mutations at each output role must be detected.
    sample = generate_puzzle(9, 2, 1)
    info = _analyze(sample['test'][0]['input'])
    r, c, h, w = info['panels'][0]['box']
    targets = [(r, c), (r + 1, c + 1), next(iter(info['routes'][0])),
               (info['hub'][0] + 1, info['hub'][1] + 1), (0, 0)]
    for point in targets:
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
        bad['test'][0]['output'][point] = (int(bad['test'][0]['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(rows), 'shapes': len(shapes),
            'palettes': len(palettes), 'structures': len(structures),
            'panel_dimensions': len(dimensions), 'panel_counts': sorted(counts),
            'port_counts': sorted(ports), 'weight_profiles': len(weights),
            'global_margins': sorted(margins),
            'diagonal_pairs': sum(m['diagonal'] > 0 for m in rows),
            'pixel_rule_disagreements': sum(m['pixel_rule_wrong'] for m in rows),
            'component_rule_disagreements': sum(m['component_rule_wrong'] for m in rows),
            'panel_count_disagreements': sum(m['panel_count_wrong'] for m in rows)}
