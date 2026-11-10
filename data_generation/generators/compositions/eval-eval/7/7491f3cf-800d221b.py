"""Polarity map: spatially switched local votes, collected by occupied ports.

Only standard-library modules and NumPy are used. Neither solving nor validation
reads files, generator records, seeds, or examples.
"""
from __future__ import annotations

import heapq
from collections import Counter

import numpy as np

DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
CHECK_SEEDS = (0, 851882467, 676216519, 379553206, 41898205, 407689751,
               343986272, 318440622, 126860624, 692675917, 252049519,
               664738047, 607335300, 696648359, 383915823, 288748111,
               638331700, 680421038, 965530883, 811130702, 672627633,
               473094293, 96204747, 154816316, 861434352, 347090389,
               732385484, 953386917, 188208595, 895543668, 50995604,
               554580492, 894825721, 844332284, 191320923, 389856195,
               734353208, 573565083, 561062547, 713153149, 928055983,
               150566851)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _need(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _need(bool(np.all((a >= 0) & (a <= 9))), "ARC colors required")
    return a


def _components(mask, diagonal=False):
    h, w = mask.shape
    ids = np.full((h, w), -1, dtype=np.int16)
    components = []
    steps = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if dr or dc) if diagonal else DIRS
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if ids[r, c] >= 0:
            continue
        index, stack, cells = len(components), [(r, c)], []
        ids[r, c] = index
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in steps:
                q = p[0] + dr, p[1] + dc
                if (0 <= q[0] < h and 0 <= q[1] < w and mask[q]
                        and ids[q] < 0):
                    ids[q] = index
                    stack.append(q)
        components.append(cells)
    return components, ids


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    if cells:
        r, c = zip(*cells)
        result[r, c] = True
    return result


def _neighbor_set(cells, shape, diagonal=False):
    h, w = shape
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if r or c) if diagonal else DIRS
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _cut_data(cut):
    """One digital path, two four-connected regions, isolated marker sites."""
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    h, w = cut.shape
    _need(len(cells) >= 3, "cut too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc) or (r + dr, c + dc) not in cells:
                    continue
                # A diagonal is an edge only when it is not an elbow shortcut.
                if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                    continue
                neighbors.append((r + dr, c + dc))
        graph[r, c] = neighbors
    ends = [p for p in sorted(graph) if len(graph[p]) == 1]
    _need(len(ends) == 2 and all(1 <= len(n) <= 2 for n in graph.values()),
          "cut must be unbranched")
    boundary = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _need(all(boundary(p) == (p in ends) for p in cells), "cut boundary contacts")
    order, previous, current = [], None, ends[0]
    while current is not None and current not in order:
        order.append(current)
        nxt = [p for p in graph[current] if p != previous]
        previous, current = current, nxt[0] if nxt else None
    _need(len(order) == len(cells), "disconnected cut")
    regions, ids = _components(~cut)
    _need(len(regions) == 2, "cut must divide two regions")
    near = cells | _neighbor_set(cells, cut.shape, diagonal=True)
    sites = [sorted(set(region) - near) for region in regions]
    _need(all(sites), "both regions must allow an isolated marker")
    steps = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(order, order[1:])]
    bends = sum(a != b for a, b in zip(steps, steps[1:]))
    return ids, sites, order, bends


def _ports(hub):
    r, c = hub
    return ([(r - 1, c + j) for j in range(3)]
            + [(r + j, c + 3) for j in range(3)]
            + [(r + 3, c + j) for j in range(3)]
            + [(r + j, c - 1) for j in range(3)])


def _solid3(mask):
    h, w = mask.shape
    if min(h, w) < 3:
        return []
    windows = np.ones((h - 2, w - 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            windows &= mask[dr:dr + h - 2, dc:dc + w - 2]
    return [tuple(map(int, p)) for p in np.argwhere(windows)]


def _read_circuit(circuit, bg):
    values = set(map(int, np.unique(circuit))) - {bg}
    _need(len(values) == 3, "circuit needs scaffold and two signals")
    interpretations = []
    h, w = circuit.shape
    for scaffold in sorted(values):
        hubs = _solid3(circuit == scaffold)
        if len(hubs) != 1:
            continue
        hub = r, c = hubs[0]
        if not (0 < r < h - 3 and 0 < c < w - 3):
            continue
        signals = tuple(sorted(values - {scaffold}))
        panel_mask = (circuit == signals[0]) | (circuit == signals[1])
        panels, panel_ids = _components(panel_mask)
        if len(panels) < 2:
            continue
        try:
            majors, boxes = [], []
            for cells in panels:
                rr, cc = zip(*cells)
                box = min(rr), min(cc), max(rr) + 1, max(cc) + 1
                _need(len(cells) >= 3 and len(cells) ==
                      (box[2] - box[0]) * (box[3] - box[1]), "rectangular ballot")
                n = sum(circuit[p] == signals[0] for p in cells)
                _need(0 < n < len(cells) and 2 * n != len(cells), "strict local vote")
                majors.append(signals[0] if 2 * n > len(cells) else signals[1])
                boxes.append(box)
            wires = circuit == scaffold
            wires[r:r + 3, c:c + 3] = False
            routes, route_ids = _components(wires)
            _need(bool(routes), "missing routes")
            owners = []
            port_cells = [p for p in _ports(hub) if circuit[p] == scaffold]
            _need(3 <= len(port_cells) <= 8, "need three to eight ports")
            for side in range(4):
                _need(sum(circuit[p] == scaffold for p in _ports(hub)[3*side:3*side+3]) <= 2,
                      "at most two ports on a collector side")
            for p in _ports(hub):
                _need(circuit[p] in (bg, scaffold), "collector collar must be empty or wired")
            for cells in routes:
                touching = {int(panel_ids[p]) for p in _neighbor_set(cells, circuit.shape)
                            if panel_ids[p] >= 0}
                _need(len(touching) == 1, "a route must touch exactly one ballot")
                _need(any(route_ids[p] == route_ids[cells[0]] for p in port_cells),
                      "every route reaches a collector port")
                owners.append(touching.pop())
            _need(set(owners) == set(range(len(panels))), "every ballot must be connected")
            port_owners = [owners[int(route_ids[p])] for p in port_cells]
            interpretations.append(dict(scaffold=scaffold, signals=signals, hub=hub,
                                        panels=panels, boxes=boxes, majors=majors,
                                        routes=routes, route_ids=route_ids, owners=owners,
                                        ports=port_cells, port_owners=port_owners))
        except AssertionError:
            continue
    _need(len(interpretations) == 1, "circuit must have one role/collector interpretation")
    return interpretations[0]


def _decode(input_grid):
    inp = _array(input_grid)
    choices = []
    for transpose in (False, True):
        g = inp.T if transpose else inp
        h, width = g.shape
        if width < 9 or (width - 3) % 2 or h < 7:
            continue
        w = (width - 3) // 2
        frame = int(g[0, 0])
        if not (np.all(g[[0, -1], :] == frame)
                and np.all(g[:, [0, w + 1, width - 1]] == frame)):
            continue
        ctrl, circuit = g[1:-1, 1:w + 1], g[1:-1, w + 2:-1]
        cv, nv = set(map(int, np.unique(ctrl))), set(map(int, np.unique(circuit)))
        if len(cv) != 2 or len(nv) != 4 or len(cv & nv) != 1 or frame in cv | nv:
            continue
        bg = next(iter(cv & nv))
        control = next(iter(cv - {bg}))
        try:
            components, _ = _components(ctrl == control, diagonal=True)
            singletons = [cells for cells in components if len(cells) == 1]
            _need(len(components) == 2 and len(singletons) == 1, "one cut and one marker")
            marker = singletons[0][0]
            cut_cells = next(cells for cells in components if len(cells) > 1)
            cut = _mask(ctrl.shape, cut_cells)
            regions, sites, path, bends = _cut_data(cut)
            side = int(regions[marker])
            _need(side in (0, 1) and marker in sites[side], "marker must be separated")
            d = _read_circuit(circuit, bg)
            d.update(canonical=g, transpose=transpose, control_panel=ctrl, circuit=circuit,
                     bg=bg, frame=frame, control=control, cut=cut, marker=marker,
                     regions=regions, marked=regions == side, side=side, sites=sites,
                     path=path, bends=bends)
            choices.append(d)
        except AssertionError:
            continue
    _need(len(choices) == 1, "input must have one aligned map/circuit interpretation")
    return choices[0]


def _opposite_signal(major, signals):
    return signals[1] if major == signals[0] else signals[0]


def _strict_winner(labels):
    counts = Counter(labels)
    _need(bool(counts), "empty vote")
    ranked = counts.most_common()
    _need(len(ranked) == 1 or ranked[0][1] > ranked[1][1], "collector vote ties")
    return int(ranked[0][0])


def _port_labels(d, marked=None, representative=False):
    marked = d['marked'] if marked is None else marked
    labels = []
    for p, owner in zip(d['ports'], d['port_owners']):
        major = int(d['circuit'][d['boxes'][owner][0], d['boxes'][owner][1]]) \
            if representative else d['majors'][owner]
        labels.append(major if marked[p] or d['cut'][p]
                      else _opposite_signal(major, d['signals']))
    return labels


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover both layers, switch signals per coordinate, then count ports."""
    d = _decode(input_grid)
    out = d['canonical'].copy()
    w = d['circuit'].shape[1]
    dest = out[1:-1, w + 2:-1]
    positive = d['marked'] | d['cut']
    for route, owner in zip(d['routes'], d['owners']):
        major = d['majors'][owner]
        minor = _opposite_signal(major, d['signals'])
        rr, cc = zip(*route)
        dest[rr, cc] = np.where(positive[rr, cc], major, minor)
    r, c = d['hub']
    dest[r + 1, c + 1] = _strict_winner(_port_labels(d))
    return out.T.copy() if d['transpose'] else out


def _pack(control, circuit, frame, transpose):
    h, w = control.shape
    g = np.full((h + 2, 2 * w + 3), frame, dtype=np.int8)
    g[1:-1, 1:w + 1] = control
    g[1:-1, w + 2:-1] = circuit
    return g.T.copy() if transpose else g


def _sample_cut(rng, shape, kind):
    h, w = shape
    for _ in range(80):
        cut = np.zeros(shape, dtype=bool)
        if kind == 'straight':
            if rng.random() < .5:
                cut[:, int(rng.integers(2, w - 2))] = True
            else:
                cut[int(rng.integers(2, h - 2)), :] = True
        elif kind == 'diagonal':
            a, b = sorted(map(int, rng.choice(np.arange(1, w - 1), 2, replace=False)))
            if rng.random() < .5:
                a, b = b, a
            cols = np.rint(np.linspace(a, b, h)).astype(int)
            cut[np.arange(h), cols] = True
        elif kind == 'elbow':
            r, c = int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2))
            cut[:r + 1, c] = True
            cut[r, c:] = True
        elif kind == 'stepped':
            across = bool(rng.integers(2))
            hh, ww = (w, h) if across else (h, w)
            trial = np.zeros((hh, ww), dtype=bool)
            rows = np.arange(2, hh - 2, 3)
            n = int(rng.integers(1, min(3, len(rows)) + 1))
            rows = sorted(map(int, rng.choice(rows, n, replace=False)))
            old, last = int(rng.integers(2, ww - 2)), 0
            for row in rows:
                trial[last:row + 1, old] = True
                nxt = int(rng.choice([c for c in range(2, ww - 2) if c != old]))
                trial[row, min(old, nxt):max(old, nxt) + 1] = True
                old, last = nxt, row
            trial[last:, old] = True
            cut = trial.T.copy() if across else trial
        else:
            raise ValueError('unknown cut type')
        if rng.integers(2):
            cut = cut[::-1, :].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            _cut_data(cut)
            return cut
        except AssertionError:
            pass
    raise RuntimeError('cut construction exhausted')


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}


def _shortest(start, targets, allowed, costs):
    h, w = allowed.shape
    if not allowed[start]:
        return None
    queue, dist, previous = [(0., start)], {start: 0.}, {}
    while queue:
        distance, p = heapq.heappop(queue)
        if distance != dist[p]:
            continue
        if p in targets:
            path = {p}
            while p != start:
                p = previous[p]
                path.add(p)
            return path
        for dr, dc in DIRS:
            q = p[0] + dr, p[1] + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and allowed[q]:
                nd = distance + float(costs[q])
                if nd < dist.get(q, float('inf')):
                    dist[q], previous[q] = nd, p
                    heapq.heappush(queue, (nd, q))
    return None


def _skeleton(rng, bounds, witness=False, rich=False):
    """Pack rectangles and route planar port groups without using phase data."""
    for _ in range(180):
        h = int(rng.integers(bounds[0], bounds[1] + 1))
        w = int(rng.integers(bounds[2], bounds[3] + 1))
        shape = h, w
        count = 3 if witness else int(rng.choice([2, 3, 4, 5], p=[.18, .38, .30, .14]))
        hub = cr, cc = int(rng.integers(3, h - 5)), int(rng.integers(3, w - 5))
        reserved = _rect(cr - 1, cc - 1, 5, 5)
        panels, used = [], set(reserved)
        for _panel in range(count):
            for _attempt in range(75):
                ph = int(rng.integers(1, min(7, h // 3) + 1))
                pw = int(rng.integers(1, min(6, w // 2) + 1))
                if ph * pw < 3:
                    continue
                side = int(rng.integers(4))
                pad = int(rng.integers(2)) if rng.random() < .2 else 0
                if side < 2:
                    r = pad if side == 0 else h - ph - pad
                    c = int(rng.integers(w - pw + 1))
                else:
                    c = pad if side == 2 else w - pw - pad
                    r = int(rng.integers(h - ph + 1))
                cells = _rect(r, c, ph, pw)
                if cells & used:
                    continue
                panels.append((r, c, ph, pw, cells))
                used |= cells | _neighbor_set(cells, shape)
                break
            else:
                break
        if len(panels) != count:
            continue
        # Circular order avoids interleaving unrelated networks.
        panels.sort(key=lambda b: float(np.arctan2(b[0] + (b[2] - 1) / 2 - cr - 1,
                                                   b[1] + (b[3] - 1) / 2 - cc - 1)))
        if witness:
            weights = [int(rng.choice([3, 4])), 1, 1]
            rng.shuffle(weights)
        else:
            total = int(rng.integers(max(3, count), 9))
            weights = [1] * count
            for _extra in range(total - count):
                i = int(rng.choice([j for j in range(count) if weights[j] < 4]))
                weights[i] += 1
        ring = [(cr - 1, cc), (cr - 1, cc + 2), (cr, cc + 3), (cr + 2, cc + 3),
                (cr + 3, cc + 2), (cr + 3, cc), (cr + 2, cc - 1), (cr, cc - 1)]
        indices = sorted(map(int, rng.choice(8, sum(weights), replace=False)))
        rotations = []
        for offset in range(len(indices)):
            sequence = indices[offset:] + indices[:offset]
            groups, score, k = [], 0., 0
            for panel, weight in zip(panels, weights):
                group = {ring[i] for i in sequence[k:k + weight]}
                pr, pc = panel[0] + (panel[2] - 1) / 2, panel[1] + (panel[3] - 1) / 2
                pa = float(np.arctan2(pr - cr - 1, pc - cc - 1))
                for r, c in group:
                    delta = abs(pa - float(np.arctan2(r - cr - 1, c - cc - 1)))
                    score += min(delta, 2 * np.pi - delta) ** 2
                groups.append(group)
                k += weight
            rotations.append((score + float(rng.random()) * 2, groups))
        groups = min(rotations, key=lambda x: x[0])[1]
        all_ports = _ports(hub)
        # A singleton can occupy the middle port; a shared pair can broaden.
        for side in range(4):
            ports = all_ports[side * 3:side * 3 + 3]
            for group in groups:
                occupied = group & set(ports)
                if len(occupied) == 2 and rng.random() < .18:
                    group.remove(sorted(occupied)[int(rng.integers(2))])
                    group.add(ports[1])
                elif len(occupied) == 1 and not any(
                        other & set(ports) for other in groups if other is not group) and rng.random() < .3:
                    group.remove(next(iter(occupied)))
                    group.add(ports[1])
        hubcells = _rect(cr, cc, 3, 3)
        collar = _rect(cr - 1, cc - 1, 5, 5) - hubcells
        allpanels = set().union(*(p[4] for p in panels))
        networks = [set(g) for g in groups]
        okay = True
        for owner in sorted(range(count), key=lambda i: -len(groups[i])):
            own = panels[owner][4]
            others = allpanels - own
            othernets = set().union(*(net for i, net in enumerate(networks) if i != owner))
            blocked = (allpanels | hubcells | collar | othernets
                       | _neighbor_set(others | othernets, shape)) - groups[owner]
            allowed = ~_mask(shape, blocked)
            targets = _neighbor_set(own, shape) - allpanels
            targets = {p for p in targets if allowed[p]}
            if not targets:
                okay = False
                break
            costs = 1 + rng.random(shape) * (5 if rich else 1.5)
            if rich:
                costs += np.repeat(np.repeat(rng.random(((h + 2)//3, (w + 2)//3))*5,
                                             3, axis=0), 3, axis=1)[:h, :w]
            net = set()
            for port in sorted(groups[owner]):
                path = _shortest(port, targets | net, allowed, costs)
                if path is None:
                    okay = False
                    break
                net |= path
            if not okay:
                break
            if rich and rng.random() < .4:
                # Add an attached edge skirt, not an unconnected decoration.
                edge, _ = _components(_mask(shape, targets))
                for segment in edge:
                    if set(segment) & (net | _neighbor_set(net, shape)) and rng.random() < .5:
                        net.update(segment)
            if (net & collar) != groups[owner]:
                okay = False
                break
            networks[owner] = net
        if okay and _solid3(_mask(shape, hubcells | set().union(*networks))) == [hub]:
            return dict(shape=shape, hub=hub, panels=panels, networks=networks)
    return None


def _ballot(rng, shape, major, minor, witness=False):
    h, w = shape
    n = h * w
    strength = int(rng.integers(3))
    k = (n - 1)//2 if strength == 0 else (1 if strength == 2 else max(1, int(n*.27)))
    k = min(k, (n - 1)//2)
    rr, cc = np.indices(shape)
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(n)
    elif style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        order = np.argsort((abs(rr-r) + abs(cc-c) + rng.random(shape)*.2).ravel())
    elif style == 2:
        order = np.argsort(((rr+cc) % 2*n + rr*w + cc).ravel())
    else:
        order = np.argsort((rr*w+cc if rng.integers(2) else cc*h+rr).ravel())
    if witness:
        order = np.concatenate(([0], order[order != 0]))
    flat = np.full(n, major, dtype=np.int8)
    flat[order[:k]] = minor
    return flat.reshape(shape)


def _paint_circuit(rng, skeleton, roles, witness):
    frame, bg, control, scaffold, s0, s1 = roles
    inp = np.full(skeleton['shape'], bg, dtype=np.int8)
    r, c = skeleton['hub']
    inp[r:r + 3, c:c + 3] = scaffold
    count = len(skeleton['panels'])
    if witness:
        heavy = max(range(count), key=lambda i: len(skeleton['networks'][i] & set(_ports(skeleton['hub']))))
        majority_bits = [1] * count
        majority_bits[heavy] = 0
        if rng.integers(2):
            majority_bits = [1-b for b in majority_bits]
    else:
        majority_bits = list(map(int, rng.integers(2, size=count)))
        if len(set(majority_bits)) == 1:
            majority_bits[int(rng.integers(count))] ^= 1
    majors = []
    for i, (panel, network) in enumerate(zip(skeleton['panels'], skeleton['networks'])):
        r, c, ph, pw, cells = panel
        major, minor = (s0, s1) if majority_bits[i] == 0 else (s1, s0)
        majors.append(major)
        inp[r:r + ph, c:c + pw] = _ballot(rng, (ph, pw), major, minor, witness and i == 0)
        for p in network:
            inp[p] = scaffold
    return inp, majors


def _make_pair(circuit, skeleton, majors, cut, marker, roles, transpose):
    frame, bg, control, scaffold, s0, s1 = roles
    region_ids, sites, _, _ = _cut_data(cut)
    chosen = int(region_ids[marker])
    positive = (region_ids == chosen) | cut
    ctrl = np.full(cut.shape, bg, dtype=np.int8)
    ctrl[cut] = control
    ctrl[marker] = control
    out = circuit.copy()
    port_labels = []
    ports = set(_ports(skeleton['hub']))
    for network, major in zip(skeleton['networks'], majors):
        minor = s1 if major == s0 else s0
        for p in network:
            out[p] = major if positive[p] else minor
            if p in ports:
                port_labels.append(int(out[p]))
    r, c = skeleton['hub']
    out[r + 1, c + 1] = _strict_winner(port_labels)
    return {'input': _pack(ctrl, circuit, frame, transpose),
            'output': _pack(ctrl, out, frame, transpose)}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(d):
    labels = _port_labels(d)
    winner = _strict_winner(labels)
    plain_labels = [d['majors'][owner] for owner in d['port_owners']]
    plain = Counter(plain_labels)
    plain_winner = None if len(plain) == 2 and len(set(plain.values())) == 1 else _strict_winner(plain_labels)
    collapsed = {}
    for owner, label in zip(d['port_owners'], labels):
        collapsed.setdefault(owner, label)
    counts = Counter(collapsed.values())
    collapsed_winner = None if len(counts) == 2 and len(set(counts.values())) == 1 else _strict_winner(list(collapsed.values()))
    split_owners = sum(len({label for owner2, label in zip(d['port_owners'], labels)
                            if owner2 == owner}) == 2 for owner in set(d['port_owners']))
    regions_seen, seam = set(), 0
    for cells in d['routes']:
        regions_seen |= {int(d['regions'][p]) for p in cells if not d['cut'][p]}
        seam += sum(d['cut'][p] for p in cells)
    hr, hc = d['hub']
    center_positive = bool(d['marked'][hr+1, hc+1] or d['cut'][hr+1, hc+1])
    post_vote = None if plain_winner is None else (plain_winner if center_positive
                 else _opposite_signal(plain_winner, d['signals']))
    steps = [(q[0]-p[0], q[1]-p[1]) for p,q in zip(d['path'], d['path'][1:])]
    cut_kind = ('straight' if d['bends'] == 0 else 'diagonal' if any(dr and dc for dr,dc in steps)
                else 'elbow' if d['bends'] == 1 else 'stepped')
    return dict(side=d['side'], shape=d['circuit'].shape,
                panel_count=len(d['panels']), port_count=len(labels),
                weights=sorted(Counter(d['port_owners']).values()),
                winner=winner, port_margin=abs(labels.count(d['signals'][0])-labels.count(d['signals'][1])),
                split_owners=split_owners, seam_cells=int(seam), both_regions=regions_seen == {0, 1},
                ignores_map_changes_center=plain_winner is not None and plain_winner != winner,
                post_vote_changes_center=post_vote is not None and post_vote != winner,
                collapse_changes_center=collapsed_winner is not None and collapsed_winner != winner,
                representative_diff=any(int(d['circuit'][b[0], b[1]]) != m
                                        for b, m in zip(d['boxes'], d['majors'])),
                center_in_other=not d['marked'][hr+1, hc+1] and not d['cut'][hr+1, hc+1],
                bends=d['bends'], route_cells=sum(map(len, d['routes'])),
                cut_kind=cut_kind,
                roles=(d['frame'], d['bg'], d['control'], d['scaffold'], *d['signals']),
                areas=sorted(map(len, d['panels'])), transpose=d['transpose'])


def _new_example(rng, bounds, roles, transpose, kinds, side=None, witness=False,
                 rich=False, post_vote_witness=False, min_split_owners=0,
                 max_split_owners=None):
    for _ in range(90):
        skeleton = _skeleton(rng, bounds, witness, rich)
        if skeleton is None:
            continue
        circuit, majors = _paint_circuit(rng, skeleton, roles, witness)
        try:
            cd = _read_circuit(circuit, roles[1])
            if min_split_owners and (sum(n >= 2 for n in Counter(cd['port_owners']).values())
                                     < min_split_owners or len(cd['ports']) <= 4):
                continue
            if post_vote_witness:
                _strict_winner([cd['majors'][owner] for owner in cd['port_owners']])
        except AssertionError:
            continue
        for _control in range(55 if witness else 12):
            kind = str(rng.choice(kinds))
            cut = _sample_cut(rng, circuit.shape, kind)
            ids, sites, _, _ = _cut_data(cut)
            chosen = int(rng.integers(2)) if side is None else side
            marker = sites[chosen][int(rng.integers(len(sites[chosen])))]
            try:
                pair = _make_pair(circuit, skeleton, majors, cut, marker, roles, transpose)
                d = _decode(pair['input'])
                f = _features(d)
                if not (f['both_regions'] and f['seam_cells'] > 0):
                    continue
                if post_vote_witness and not f['post_vote_changes_center']:
                    continue
                if f['split_owners'] < min_split_owners or (max_split_owners is not None
                        and f['split_owners'] > max_split_owners):
                    continue
                if witness:
                    # One panel supplies conflicting ports. Both levels of vote
                    # are needed, with a strict counterexample to two shortcuts.
                    if not (f['split_owners'] > 0 and f['ignores_map_changes_center']
                            and f['collapse_changes_center'] and f['representative_diff']
                            and f['center_in_other']):
                        continue
                return pair, (skeleton, circuit, majors, cut, roles, transpose)
            except AssertionError:
                continue
    raise RuntimeError('bounded example construction exhausted')


def _counts(num_train, num_test):
    for x, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if (isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
                or not lo <= int(x) <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    roles = tuple(map(int, rng.choice(10, 6, replace=False)))
    transpose = bool(rng.integers(2))
    varying_palette = bool(rng.random() < .45)
    train_bounds = (14, 23, 10, 13) if regime == 2 else (16, 28, 11, 13)
    test_bounds = (24, 28, 11, 13) if regime == 2 else train_bounds
    train_kinds = ('straight', 'elbow') if regime == 0 else ('straight', 'diagonal', 'elbow', 'stepped')
    test_kinds = ('stepped',) if regime == 0 else train_kinds
    first_side = int(rng.integers(2))
    first, record = _new_example(rng, train_bounds, roles, transpose, train_kinds,
                                 first_side, witness=True, rich=regime == 1)
    puzzle = {'train': [first], 'test': []}
    seen = {_key(first['input'])}
    # Controlled twins are one way to teach the marker. Other puzzles use a
    # different circuit and cut for the second, opposite-side demonstration.
    twin = nt >= 3 and rng.random() < .6
    for index in range(1, nt):
        if index == 1 and twin:
            skeleton, circuit, majors, cut, used_roles, used_transpose = record
            _, sites, _, _ = _cut_data(cut)
            marker = sites[1-first_side][int(rng.integers(len(sites[1-first_side])))]
            try:
                pair = _make_pair(circuit, skeleton, majors, cut, marker, used_roles, used_transpose)
                _need(_features(_decode(pair['input']))['post_vote_changes_center'],
                      'twin must also exclude post-vote inversion')
            except AssertionError:
                twin = False
        if index != 1 or not twin:
            palette = tuple(map(int, rng.choice(10, 6, replace=False))) if varying_palette else roles
            pair, _ = _new_example(rng, train_bounds, palette, transpose, train_kinds,
                                  1-first_side if index == 1 else None,
                                  rich=False, post_vote_witness=index == 1,
                                  max_split_owners=1 if regime == 1 else None)
        for _duplicate in range(20):
            if _key(pair['input']) not in seen:
                break
            pair, _ = _new_example(rng, train_bounds, roles, transpose, train_kinds,
                                  1-first_side if index == 1 else None,
                                  rich=False, post_vote_witness=index == 1,
                                  max_split_owners=1 if regime == 1 else None)
        else:
            raise RuntimeError('duplicate training input')
        seen.add(_key(pair['input']))
        puzzle['train'].append(pair)
    # Randomly position the diagnostic witness, removing ordinal output hints.
    rng.shuffle(puzzle['train'])
    for _ in range(nq):
        for _duplicate in range(20):
            palette = tuple(map(int, rng.choice(10, 6, replace=False))) if varying_palette or regime == 2 else roles
            pair, _ = _new_example(rng, test_bounds, palette, transpose, test_kinds,
                                  rich=regime == 1, min_split_owners=2 if regime == 1 else 0)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('duplicate test input')
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'list splits required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _array(pair['input']), _array(pair['output'])
            _need(inp.shape == out.shape, 'shape preservation')
            _need(_key(inp) not in seen, 'duplicate inputs')
            seen.add(_key(inp))
            d = _decode(inp)
            actual = out.T if d['transpose'] else out
            expected = d['canonical'].copy()
            w = d['circuit'].shape[1]
            dest = expected[1:-1, w + 2:-1]
            # Cellwise semantics, separately from solve's vectorized assignment.
            for route, owner in zip(d['routes'], d['owners']):
                for p in route:
                    major = d['majors'][owner]
                    dest[p] = major if d['cut'][p] or d['marked'][p] \
                        else _opposite_signal(major, d['signals'])
            r, c = d['hub']
            dest[r+1, c+1] = _strict_winner([int(dest[p]) for p in d['ports']])
            _need(np.array_equal(actual, expected), 'wrong switched route, vote, or preserved cell')
            f = _features(d)
            _need(f['both_regions'] and f['seam_cells'] > 0, 'ineffective polarity map')
            _need(set(d['majors']) == set(d['signals']), 'both local majority colors required')
            if split == 'train':
                training.append(f)
    _need({f['side'] for f in training} == {0, 1}, 'teach both marked regions')
    _need(any(f['split_owners'] and f['ignores_map_changes_center'] and
              f['collapse_changes_center'] and f['representative_diff'] for f in training),
          'missing integrated training witness')
    _need(any(f['center_in_other'] for f in training), 'teach that collector is not inverted again')
    _need(any(f['post_vote_changes_center'] for f in training), 'exclude inversion after voting')


def self_test():
    """Bounded structural, semantic, metamorphic and diversity checks: 42 seeds."""
    puzzles, features, structures, shapes, cuts = [], [], set(), set(), set()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        local = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                untouched = inp.copy()
                _need(np.array_equal(solve(inp), out), 'solver disagreement')
                _need(np.array_equal(inp, untouched), 'solve mutated input')
                d = _decode(inp)
                f = _features(d)
                features.append(f)
                local[split].append(f)
                shapes.add(inp.shape)
                cuts.add((d['cut'].shape, d['cut'].tobytes()))
                role_order = {}
                structures.add((inp.shape, tuple(role_order.setdefault(int(v), len(role_order))
                                                  for v in inp.flat)))
        scale_shift = max(f['shape'][0] for f in local['train']) < min(f['shape'][0] for f in local['test'])
        map_shift = ({f['cut_kind'] for f in local['train']} <= {'straight','elbow'}
                     and {f['cut_kind'] for f in local['test']} == {'stepped'})
        topology_shift = (max(f['split_owners'] for f in local['train']) <= 1
                          and min(f['split_owners'] for f in local['test']) >= 2)
        _need(scale_shift or map_shift or topology_shift, 'missing observable train/test shift')
        # Consistent transposition and recoloring commute with the entire rule.
        pair = puzzle['train'][0]
        _need(np.array_equal(solve(pair['input'].T), pair['output'].T), 'transpose covariance')
        perm = np.random.default_rng(seed + 99).permutation(10)
        _need(np.array_equal(solve(perm[pair['input']]), perm[pair['output']]), 'color covariance')
        d = _decode(pair['input'])
        # Ballot arrangement is immaterial to its majority; even a newly solid
        # signal patch must not be mistaken for the scaffold collector.
        rearranged = d['circuit'].copy()
        r0, c0, r1, c1 = max(d['boxes'], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        block = rearranged[r0:r1, c0:c1]
        block[:] = np.sort(block, axis=None).reshape(block.shape)
        shuffled_in = _pack(d['control_panel'], rearranged, d['frame'], d['transpose'])
        expected = pair['output'].T.copy() if d['transpose'] else pair['output'].copy()
        expected[1+r0:1+r1, d['circuit'].shape[1]+2+c0:d['circuit'].shape[1]+2+c1] = block
        if d['transpose']:
            expected = expected.T.copy()
        _need(np.array_equal(solve(shuffled_in), expected), 'ballot permutation invariance')
        ctrl = d['control_panel'].copy()
        ctrl[d['marker']] = d['bg']
        ctrl[d['sites'][1-d['side']][0]] = d['control']
        swapped = _pack(ctrl, d['circuit'], d['frame'], d['transpose'])
        sd = _decode(swapped)
        # A tied modified collector is outside the family; compare switched wires
        # directly in that case instead of inventing a tie-breaking convention.
        for route, owner in zip(d['routes'], d['owners']):
            for p in route:
                if d['cut'][p]:
                    _need((sd['marked'] | sd['cut'])[p], 'seam priority moved')
                else:
                    _need(d['marked'][p] != sd['marked'][p], 'marker failed to swap phases')
        if len(set(Counter(_port_labels(sd)).values())) > 1 or len(set(_port_labels(sd))) == 1:
            changed = solve(swapped)
            canon = changed.T if sd['transpose'] else changed
            nc = canon[1:-1, d['circuit'].shape[1]+2:-1]
            oc = (pair['output'].T if d['transpose'] else pair['output'])[1:-1, d['circuit'].shape[1]+2:-1]
            for route in d['routes']:
                for p in route:
                    _need(nc[p] == oc[p] if d['cut'][p] else nc[p] != oc[p], 'marker switching equation')
    _need(len({tuple(_key(pair['input']) for s in ('train','test') for pair in p[s])
               for p in puzzles}) == 42, 'seed diversity')
    _need(len(structures) >= 150 and len(shapes) >= 20 and len(cuts) >= 100, 'geometry diversity')
    _need({f['transpose'] for f in features} == {False, True}, 'map orientations')
    _need({f['panel_count'] for f in features} >= {2, 3, 4, 5}, 'ballot counts')
    _need({f['port_count'] for f in features} >= {3, 4, 5, 6, 7, 8}, 'port counts')
    _need(any(f['bends'] >= 4 for f in features), 'multi-bend maps')
    _need(len({tuple(f['weights']) for f in features}) >= 10, 'route multiplicities')
    _need(max(max(f['areas']) for f in features) >= 24, 'unequal ballot sizes')
    _need(len({f['roles'] for f in features}) >= 35, 'role palettes')
    a, b = generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1)
    _need(all(np.array_equal(x[k], y[k]) for s in a for x,y in zip(a[s],b[s])
              for k in ('input','output')), 'reproducibility')
    for nt, nq in ((10,4), (3,2), (5,4)):
        validate_generated_puzzle(generate_puzzle(23+nt, nt, nq))
    for nt, nq in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4.0,1), (4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: a.copy() for k,a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['train'][0]['output'][0,0] = (int(bad['train'][0]['output'][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed frame accepted')
    original = puzzles[0]['train'][0]
    d = _decode(original['input'])
    r, c = d['hub']
    targets = [(r+1, c+1), d['routes'][0][0], d['panels'][0][0]]
    for p in targets:
        bad = {s: [{k: a.copy() for k,a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        output = bad['train'][0]['output']
        canonical = output.T if d['transpose'] else output
        dest = canonical[1:-1, d['circuit'].shape[1]+2:-1]
        dest[p] = (int(dest[p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('incorrect center, route or preserved ballot accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
