"""Relocated council: chamber correspondence transports signals, windows weight them.

Only NumPy and the standard library are used. Generation keeps one candidate in
memory; no examples, logs, execution traces, or caches are written by this module.
"""
from collections import Counter
import hashlib
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 934456713, 720305158, 747449073, 586597151, 695719690,
              39448648, 181515662, 626176118, 441111806, 634941055, 464379213,
              754308300, 820650692, 451400367, 372960770, 522833550, 976603320,
              594131889, 543555152, 603334499, 811557710, 746198725, 745345549,
              180877602, 815939951, 630376870, 673559186, 13164476, 370861747,
              887586829, 542092424, 390803670, 656946275, 303287924, 92489832,
              659253331, 79266284, 418231739, 344985404, 563240814, 248287180)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask)))
    parts = []
    steps = DIRS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    while pending:
        p = min(pending)
        pending.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in pending:
                    pending.remove(q)
                    stack.append(q)
        parts.append(np.array(sorted(cells), dtype=int))
    return parts


def _box(points):
    return (*map(int, points.min(0)), *map(int, points.max(0) + 1))


def _rooms(walls):
    padded = np.pad(~walls, 1, constant_values=True)
    return [p - 1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape) - 1)]


def _room_ids(shape, rooms):
    ids = np.full(shape, -1, dtype=int)
    for i, p in enumerate(rooms):
        ids[tuple(p.T)] = i
    return ids


def _opposite(ids, r, c):
    pairs = []
    h, w = ids.shape
    for dr, dc in ((1, 0), (0, 1)):
        if 0 <= r-dr < h and 0 <= r+dr < h and 0 <= c-dc < w and 0 <= c+dc < w:
            a, b = int(ids[r-dr, c-dc]), int(ids[r+dr, c+dc])
            if a >= 0 and b >= 0 and a != b:
                pairs.append((a, b))
    return pairs


def _graph(rooms, shape):
    boxes = [_box(p) for p in rooms]
    ids = _room_ids(shape, rooms)
    n = len(rooms)
    adjacency = np.zeros((n, n), dtype=bool)
    for r, c in np.argwhere(ids < 0):
        for a, b in _opposite(ids, r, c):
            adjacency[a, b] = adjacency[b, a] = True
    relations = [[tuple(-1 if a[k+2] <= b[k] else 1 if b[k+2] <= a[k] else 0
                        for k in (0, 1)) for b in boxes] for a in boxes]
    return relations, adjacency


def _correspondence(dr, tr, ds, ts, council_d, council_t):
    """Exact qualitative spatial graph matching, anchored by the council seal.

    Areas, colors, discovery ranks, and construction labels are not keys.
    This bounded family requires a unique orientation-preserving mapping.
    """
    a, _ = _graph(dr, ds)
    b, _ = _graph(tr, ts)
    n = len(dr)
    assert n == len(tr) and 4 <= n <= 10
    options = {}
    for i in range(n):
        options[i] = [j for j in range(n)
                      if (i == council_d) == (j == council_t)
                      and Counter(a[i]) == Counter(b[j])]
    answers, visits = [], 0

    def visit(mapping, used):
        nonlocal visits
        visits += 1
        assert visits <= 20000, 'spatial correspondence search exceeded bound'
        if len(answers) > 1:
            return
        if len(mapping) == n:
            answers.append(dict(mapping))
            return
        possible = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v]
            for k, v in mapping.items())] for i in range(n) if i not in mapping}
        i = min(possible, key=lambda k: (len(possible[k]), k))
        for j in possible[i]:
            mapping[i] = j
            visit(mapping, used | {j})
            del mapping[i]

    visit({council_d: council_t}, {council_t})
    assert len(answers) == 1, 'compartment correspondence must be unique'
    return answers[0]


def _figure(inp, part, bg, wall, marker, signals):
    r, c, R, C = _box(part)
    crop = inp[r:R, c:C].copy()
    frame = np.zeros(crop.shape, dtype=bool)
    frame[part[:, 0]-r, part[:, 1]-c] = True
    rooms = _rooms(frame)
    assert 4 <= len(rooms) <= 10
    ids = _room_ids(crop.shape, rooms)
    inside = ids >= 0
    assert np.all(crop[~(frame | inside)] == bg)
    seals = np.argwhere((crop == marker) & ~frame)
    assert len(seals) == 1
    seal = tuple(map(int, seals[0]))
    council = int(ids[seal])
    assert council >= 0
    assert all(0 <= seal[0]+dr < len(crop) and 0 <= seal[1]+dc < crop.shape[1]
               and ids[seal[0]+dr, seal[1]+dc] == council for dr, dc in DIRS)
    council_values = crop[tuple(rooms[council].T)]
    assert set(map(int, council_values)) == {bg, marker}
    colored = bool(np.any(np.isin(crop, signals)))
    local, ballots = {}, {}
    for i, points in enumerate(rooms):
        if i == council:
            continue
        values = crop[tuple(points.T)]
        if colored:
            assert len(values) >= 3 and set(map(int, values)) == set(signals)
            counts = [int(np.count_nonzero(values == s)) for s in signals]
            assert counts[0] != counts[1]
            local[i] = signals[int(counts[1] > counts[0])]
            ballots[i] = counts
        else:
            assert np.all(values == bg)
    gates = []
    for x, y in np.argwhere(frame & (crop == marker)):
        assert not colored, 'only the destination has voting windows'
        opposites = _opposite(ids, int(x), int(y))
        assert len(opposites) == 1 and council in opposites[0]
        a, b = opposites[0]
        owner = b if a == council else a
        assert any(0 <= x+dx < len(crop) and 0 <= y+dy < crop.shape[1]
                   and crop[x+dx, y+dy] == wall for dx, dy in DIRS)
        gates.append(((int(x), int(y)), owner))
    assert np.all(np.isin(crop[frame], (wall, marker)))
    if colored:
        assert not gates and np.all(crop[frame] == wall)
    else:
        assert 3 <= len(gates) <= 12
        _, adjacency = _graph(rooms, crop.shape)
        assert {owner for p, owner in gates} == set(np.flatnonzero(adjacency[council]))
    return dict(crop=crop, frame=frame, rooms=rooms, ids=ids, seal=seal,
                council=council, colored=colored, local=local, ballots=ballots,
                gates=gates, bbox=(r, c, R, C))


def _interpret(inp, bg, wall, marker, parts):
    signals = tuple(int(s) for s in np.unique(inp) if s not in (bg, wall, marker))
    assert len(signals) == 2
    figures = [_figure(inp, p, bg, wall, marker, signals) for p in parts]
    assert sum(f['colored'] for f in figures) == 1
    donor = next(f for f in figures if f['colored'])
    target = next(f for f in figures if not f['colored'])
    mapping = _correspondence(donor['rooms'], target['rooms'], donor['crop'].shape,
                              target['crop'].shape, donor['council'], target['council'])
    labels = {mapping[i]: color for i, color in donor['local'].items()}
    votes = Counter(labels[owner] for p, owner in target['gates'])
    assert set(votes) == set(signals) and votes[signals[0]] != votes[signals[1]]
    winner = max(votes, key=votes.get)
    out = target['crop'].copy()
    for i, points in enumerate(target['rooms']):
        out[tuple(points.T)] = winner if i == target['council'] else labels[i]
    out[target['seal']] = marker
    for p, owner in target['gates']:
        out[p] = labels[owner]
    return out, dict(background=bg, wall=wall, marker=marker, signals=signals,
                     donor=donor, target=target, mapping=mapping, labels=labels,
                     votes=votes, winner=winner)


def _analyze(input_grid):
    inp = _grid(input_grid)
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    assert np.all(border == border[0]), 'family has a uniform one-cell margin'
    bg = int(border[0])
    colors = [int(c) for c in np.unique(inp) if c != bg]
    assert len(colors) == 4
    answers = []
    for wall in colors:
        for marker in colors:
            if marker == wall:
                continue
            parts = _components((inp == wall) | (inp == marker), diagonal=True)
            frames = [p for p in parts if len(p) > 1]
            singles = [p for p in parts if len(p) == 1]
            if len(frames) != 2 or len(singles) != 2:
                continue
            if not all(inp[tuple(p[0])] == marker for p in singles):
                continue
            # The ballot building's framework has no marker windows. This also
            # distinguishes wall from marker without a literal color legend.
            if not any(np.all(inp[tuple(p.T)] == wall) for p in frames):
                continue
            try:
                answers.append(_interpret(inp, bg, wall, marker, frames))
            except AssertionError:
                continue
    assert len(answers) == 1, 'input must have one geometric role interpretation'
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible rooms, ballots, seals, and voting windows."""
    return _analyze(input_grid)[0]


def _plan(rng, richer, witness):
    choices = ((2, 3), (3, 2)) if not richer else ((3, 3), (3, 4), (4, 3))
    shape = choices[int(rng.integers(len(choices)))]
    rows, cols = shape
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    eligible = [p for p in cells if not witness or sum(
        (p[0]+dr, p[1]+dc) in cells for dr, dc in DIRS) >= 3]
    council = eligible[int(rng.integers(len(eligible)))]
    plan = np.arange(rows*cols, dtype=int).reshape(shape) + 1
    plan[council] = 0
    for p in ((0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)):
        if p != council and rng.random() < .22 and np.count_nonzero(plan >= 0) > (7 if richer else 4):
            plan[p] = -1
    count = int(rng.integers(7, min(10, np.count_nonzero(plan >= 0))+1)) if richer else int(rng.integers(4, 7))
    while len(set(plan.flat)-{-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r+dr, c+dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((1, 0), (0, 1))
                 if r+dr < rows and c+dc < cols and plan[r, c] > 0
                 and plan[r+dr, c+dc] > 0 and plan[r, c] != plan[r+dr, c+dc]]
        if not edges:
            break
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    for new, old in enumerate(sorted(set(plan.flat)-{-1})):
        plan[plan == old] = new
    return plan, council


def _render(plan, widths, heights, rng, detail):
    ys = np.r_[0, np.cumsum(heights+1)]
    xs = np.r_[0, np.cumsum(widths+1)]
    ids = np.full((int(ys[-1])+1, int(xs[-1])+1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        k = int(plan[r, c])
        ids[ys[r]+1:ys[r+1], xs[c]+1:xs[c+1]] = k
        if r and plan[r-1, c] == k:
            ids[ys[r], xs[c]+1:xs[c+1]] = k
        if c and plan[r, c-1] == k:
            ids[ys[r]+1:ys[r+1], xs[c]] = k
        if r and c and np.all(plan[r-1:r+1, c-1:c+1] == k):
            ids[ys[r], xs[c]] = k
    for k in sorted(set(plan.flat)-{-1, 0}):
        if rng.random() >= detail:
            continue
        points = np.argwhere(ids == k)
        r, c, R, C = _box(points)
        axis = int(rng.integers(4))
        cut = points[(points[:, 0] == r) if axis == 0 else
                     (points[:, 0] == R-1) if axis == 1 else
                     (points[:, 1] == c) if axis == 2 else (points[:, 1] == C-1)]
        mask = ids == k
        mask[tuple(cut.T)] = False
        if mask.sum() >= 3 and len(_components(mask)) == 1:
            ids[tuple(cut.T)] = -1
        # A beveled room corner is a genuine shape change, unlike trimming a
        # complete side. It preserves a room's bounding projections when valid.
        points = np.argwhere(ids == k)
        if len(points) >= 5:
            r, c, R, C = _box(points)
            corners = [(r, c), (r, C-1), (R-1, c), (R-1, C-1)]
            for index in rng.permutation(4):
                p = corners[int(index)]
                if ids[p] != k:
                    continue
                mask = ids == k
                mask[p] = False
                if len(_components(mask)) == 1:
                    ids[p] = -1
                    break
    # Room boundary pixels, including diagonal corners; all other pixels exterior.
    near = np.zeros(ids.shape, dtype=bool)
    padded = np.pad(ids >= 0, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr+len(ids), dc:dc+ids.shape[1]]
    ids[(ids < 0) & near] = -2
    return ids


def _seal(ids):
    candidates = [(int(r), int(c)) for r, c in np.argwhere(ids == 0)
                  if all(ids[r+dr, c+dc] == 0 for dr, dc in DIRS)]
    assert candidates
    center = np.argwhere(ids == 0).mean(0)
    return min(candidates, key=lambda p: (float(np.sum((np.array(p)-center)**2)), p))


def _slots(ids):
    groups = {}
    for r, c in np.argwhere(ids == -2):
        pairs = _opposite(ids, int(r), int(c))
        if len(pairs) == 1 and 0 in pairs[0]:
            a, b = pairs[0]
            owner = b if a == 0 else a
            groups.setdefault(owner, []).append((int(r), int(c)))
    return groups


def _windows(rng, ids, witness, dense):
    slots = _slots(ids)
    assert len(slots) >= (3 if witness else 2)
    gates = {}
    for owner, cells in slots.items():
        # A gap between markers keeps sockets individually visible on a shared wall.
        selected = []
        for index in rng.permutation(len(cells)):
            p = cells[int(index)]
            if all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in selected):
                selected.append(p)
        count = len(selected) if dense else int(rng.integers(1, len(selected)+1))
        for p in selected[:min(3, count)]:
            gates[p] = owner
    assert 3 <= len(gates) <= 12
    if dense:
        assert len(gates) >= 7
    return gates


def _assignment(rng, ids, gates, witness):
    n = int(ids.max())
    areas = {i: int(np.sum(ids == i)) for i in range(1, n+1)}
    assert all(a >= 3 for a in areas.values())
    weights = Counter(gates.values())
    options = []
    for bits in range(1, (1 << n)-1):
        wins = {i for i in range(1, n+1) if bits >> (i-1) & 1}
        margin = sum(w if i in wins else -w for i, w in weights.items())
        if margin <= 0 or not any(i not in wins for i in weights):
            continue
        if witness:
            if 2*len(set(weights) & wins) >= len(weights):
                continue
            min_winner_margin = sum(1 if areas[i] % 2 else 2 for i in wins)
            max_loser_margin = sum(areas[i]-2 for i in areas if i not in wins)
            if min_winner_margin >= max_loser_margin:
                continue
            if not any(margin - 2*(weights[i]-weights[j]) < 0
                       for i in weights if i in wins for j in weights if j not in wins):
                continue
        options.append(wins)
    assert options
    return options[int(rng.integers(len(options)))], areas


def _paint_pair(rng, d, t, gates, wins, roles, witness):
    bg, wall, marker, winner, loser = roles
    donor = np.full(d.shape, bg, dtype=np.int8)
    target = np.full(t.shape, bg, dtype=np.int8)
    donor[d == -2] = wall
    target[t == -2] = wall
    expected = target.copy()
    local = {i: winner if i in wins else loser for i in range(1, int(d.max())+1)}
    for i, color in local.items():
        points = np.argwhere(d == i)
        area = len(points)
        minority_count = ((area-1)//2 if i in wins else 1) if witness else int(rng.integers(1, (area-1)//2+1))
        # Clustered, scattered and row-ordered ballots; only counts matter.
        style = int(rng.integers(3))
        order = rng.permutation(area) if style == 0 else np.argsort(
            np.sum(abs(points-points[int(rng.integers(area))]), axis=1), kind='stable') if style == 1 else np.arange(area)
        donor[tuple(points.T)] = color
        donor[tuple(points[order[:minority_count]].T)] = loser if color == winner else winner
        expected[t == i] = color
    donor[_seal(d)] = target[_seal(t)] = marker
    expected[t == 0] = winner
    expected[_seal(t)] = marker
    for p, owner in gates.items():
        target[p] = marker
        expected[p] = local[owner]
    return donor, target, expected


def _embed(rng, donor, target, bg):
    dh, dw = donor.shape
    th, tw = target.shape
    options = []
    for vertical in (False, True):
        a, b = (donor, target) if not vertical else (donor.T, target.T)
        if a.shape[1]+b.shape[1]+3 <= 30 and max(a.shape[0], b.shape[0])+2 <= 30:
            options.append(vertical)
    assert options, 'two figures must fit with a blank border and gap'
    vertical = options[int(rng.integers(len(options)))]
    a, b = (donor, target) if not vertical else (donor.T, target.T)
    reverse = bool(rng.integers(2))
    if reverse:
        a, b = b, a
    width = a.shape[1]+b.shape[1]+3
    height = max(a.shape[0], b.shape[0])+2
    width += int(rng.integers(min(3, 30-width)+1))
    height += int(rng.integers(min(4, 30-height)+1))
    canvas = np.full((height, width), bg, dtype=np.int8)
    ac = int(rng.integers(1, width-a.shape[1]-b.shape[1]-1))
    bc = int(rng.integers(ac+a.shape[1]+1, width-b.shape[1]))
    for grid, c in ((a, ac), (b, bc)):
        r = int(rng.integers(1, height-grid.shape[0]))
        canvas[r:r+grid.shape[0], c:c+grid.shape[1]] = grid
    if vertical:
        canvas = canvas.T
    return canvas.copy()


def measure_pair(pair):
    predicted, info = _analyze(pair['input'])
    assert np.array_equal(predicted, _grid(pair['output']))
    d, t = info['donor'], info['target']
    m, labels, winner = info['mapping'], info['labels'], info['winner']
    weights = Counter(owner for p, owner in t['gates'])
    other = next(c for c in info['signals'] if c != winner)
    feeding_margin = sum(1 if labels[i] == winner else -1 for i in weights)
    raw_margin = int(np.sum(d['crop'] == winner)-np.sum(d['crop'] == other))
    di = [i for i in range(len(d['rooms'])) if i != d['council']]
    ti = [i for i in range(len(t['rooms'])) if i != t['council']]
    scan_labels = {j: d['local'][i] for i, j in zip(di, ti)}
    scan_wrong = any(scan_labels[j] != labels[j] for j in ti)
    scan_votes = Counter(scan_labels[j] for p, j in t['gates'])
    scan_reverses = scan_votes[winner] < scan_votes[other]
    swap_reverses = any(
        info['votes'][winner]-info['votes'][other]-2*(weights[i]-weights[j]) < 0
        for i in weights if labels[i] == winner for j in weights if labels[j] == other)
    gate_mask = np.zeros(predicted.shape, dtype=bool)
    for p, owner in t['gates']:
        gate_mask[p] = True
    assert np.array_equal(predicted[t['frame'] & ~gate_mask], t['crop'][t['frame'] & ~gate_mask])
    assert predicted[t['seal']] == info['marker']
    assert np.array_equal(predicted[t['ids'] < 0][~t['frame'][t['ids'] < 0]],
                          t['crop'][t['ids'] < 0][~t['frame'][t['ids'] < 0]])
    for i, points in enumerate(t['rooms']):
        vals = predicted[tuple(points.T)]
        if i == t['council']:
            assert set(map(int, vals)) == {winner, info['marker']}
        else:
            assert np.all(vals == labels[i])
    signs = {int(np.sign(len(t['rooms'][j])-len(d['rooms'][i]))) for i, j in m.items()}
    nonrect = sum(len(p) != (_box(p)[2]-_box(p)[0])*(_box(p)[3]-_box(p)[1]) for p in d['rooms']+t['rooms'])
    return dict(rooms=len(t['rooms']), ports=len(t['gates']), weights=tuple(sorted(weights.values())),
                winner_index=info['signals'].index(winner), feeding_margin=feeding_margin,
                raw_margin=raw_margin, scan_wrong=scan_wrong, scan_reverses=scan_reverses,
                swap_reverses=swap_reverses, area_signs=tuple(sorted(signs)),
                nonrectangular=nonrect, donor_shape=d['crop'].shape, target_shape=t['crop'].shape,
                palette=(info['background'], info['wall'], info['marker'], *info['signals']),
                signature=(t['frame'].shape, t['frame'].tobytes(), tuple(sorted(t['gates']))))


def _fallback(rng, roles, richer=False, dense=False, witness=False):
    # A constructive witness, not a stored rendered puzzle. Both geometries,
    # ballots, palette, orientation and placement are still generated.
    if (richer or dense) and not witness:
        plan = np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
        d = _render(plan, np.array([2, 3, 2]), np.array([2, 3, 2]), rng, 0)
        t = _render(plan, np.array([2, 5, 2]), np.array([2, 5, 2]), rng, 0)
        gates = {p: owner for owner, cells in _slots(t).items()
                 for p in (cells[::2] if dense else cells[:1])}
        wins, _ = _assignment(rng, d, gates, False)
        return _paint_pair(rng, d, t, gates, wins, roles, False)
    plan = np.array([[1, 0, 2], [3, 4, 5]])
    dh = int(rng.integers(3, 5))
    dw = int(rng.integers(3, 5))
    bottom = int(rng.integers(3, 5))
    d = _render(plan, np.array([2, dw, 2]), np.array([dh, bottom]), rng, 0)
    t = _render(plan, np.array([2, 5, 2]), np.array([3, int(rng.integers(2, 5))]), rng, 0)
    d[dh+2, 1:3] = -2  # Change discovery order without changing spatial relations.
    gates = {p: owner for owner, cells in _slots(t).items()
             for p in (cells[::2] if owner == 4 else cells[:1])}
    assert Counter(gates.values()) == {1: 1, 2: 1, 4: 3}
    return _paint_pair(rng, d, t, gates, {4}, roles, True)


def _make_pair(rng, roles, richer, detail, witness=False, dense=False, port_cap=12):
    last_error = ''
    for attempt in range(97):
        try:
            if attempt == 96:
                donor, target, expected = _fallback(rng, roles, richer, dense, witness)
            else:
                plan, (cr, cc) = _plan(rng, richer, witness)
                widths = rng.choice([2, 3], size=plan.shape[1], p=[.8, .2])
                heights = rng.choice([2, 3], size=plan.shape[0], p=[.8, .2])
                widths[cc], heights[cr] = int(rng.integers(3, 5)), int(rng.integers(3, 5))
                tw = rng.choice([2, 3], size=plan.shape[1], p=[.8, .2])
                th = rng.choice([2, 3], size=plan.shape[0], p=[.8, .2])
                tw[cc], th[cr] = int(rng.integers(3, 6)), int(rng.integers(3, 6))
                if witness:
                    if rng.random() < .5:
                        tw[cc] = 5
                    else:
                        th[cr] = 5
                d = _render(plan, widths, heights, rng, detail)
                t = _render(plan, tw, th, rng, detail)
                gates = _windows(rng, t, witness, dense)
                wins, areas = _assignment(rng, d, gates, witness)
                donor, target, expected = _paint_pair(rng, d, t, gates, wins, roles, witness)
            turns = 0 if attempt == 96 and witness else int(rng.integers(4))
            donor, target, expected = [np.rot90(a, turns).copy() for a in (donor, target, expected)]
            inp = _embed(rng, donor, target, roles[0])
            pair = {'input': inp, 'output': expected}
            m = measure_pair(pair)
            assert m['ports'] <= port_cap
            if attempt < 96:
                assert m['donor_shape'] != m['target_shape'] or m['nonrectangular']
                if richer:
                    assert m['rooms'] >= 7
                if dense:
                    assert m['ports'] >= 7
            if richer and not witness:
                assert m['rooms'] >= 7
            if dense:
                assert m['ports'] >= 7
            if witness:
                assert m['feeding_margin'] < 0 and m['raw_margin'] < 0
                assert m['scan_wrong'] and m['swap_reverses']
            return pair
        except AssertionError as error:
            last_error = str(error)
            continue
    raise AssertionError('bounded council construction exhausted: '+last_error)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(name+' is out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # hold out room count, window count, or palette roles
    stable = bool(rng.integers(2))
    palette = list(map(int, rng.choice(10, 5, replace=False)))
    winner_indices = list(map(int, rng.integers(2, size=num_train)))
    if len(set(winner_indices)) == 1:
        winner_indices[int(rng.integers(num_train))] ^= 1
    witness_slot = int(rng.integers(num_train))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            colors = palette.copy() if stable or mode == 2 else list(map(int, rng.choice(10, 5, replace=False)))
            if mode == 2 and split == 'test':
                new_signals = list(map(int, rng.choice([c for c in range(10) if c not in palette[3:]], 2, replace=False)))
                colors = []
                for role in range(3):
                    pool = [c for c in range(10) if c not in new_signals+colors and c != palette[role]]
                    colors.append(int(rng.choice(pool)))
                colors += new_signals
            signals = sorted(colors[3:])
            win_index = winner_indices[i] if split == 'train' else int(rng.integers(2))
            roles = colors[:3]+[signals[win_index], signals[1-win_index]]
            witness = split == 'train' and i == witness_slot
            richer = (split == 'test') if mode == 0 else bool(rng.random() < .45)
            detail = .5
            dense = mode == 1 and split == 'test'
            port_cap = 6 if mode == 1 and split == 'train' else 12
            for retry in range(12):
                pair = _make_pair(rng, roles, richer, detail, witness, dense, port_cap)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('bounded duplicate retry exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, metrics, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = measure_pair({'input': inp, 'output': out})
            if split == 'train':
                metrics.append(m)
            else:
                tests.append(m)
    assert {m['winner_index'] for m in metrics} == {0, 1}
    assert any(m['feeding_margin'] < 0 and m['raw_margin'] < 0
               and m['scan_wrong'] and m['swap_reverses'] for m in metrics)
    room_shift = max(m['rooms'] for m in metrics) < min(m['rooms'] for m in tests)
    port_shift = max(m['ports'] for m in metrics) < min(m['ports'] for m in tests)
    palette = metrics[0]['palette']
    palette_shift = (all(m['palette'] == palette for m in metrics)
                     and all(all(m['palette'][i] != palette[i] for i in range(3))
                             and not (set(m['palette'][3:]) & set(palette[3:])) for m in tests))
    assert room_shift or port_shift or palette_shift, 'missing observable train/test shift'


def self_test():
    """42 supplied seeds, explicit-count edges, metamorphic checks, corruption.

    Streams one puzzle at a time and returns only a compact summary. Peak grid
    payload is under 40 KiB; Python object/search overhead is also bounded.
    """
    shapes, structures, palettes, profiles, rooms, ports, signs, winners = [set() for _ in range(8)]
    nonrect, scan, counterfactual, pairs = 0, 0, 0, 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                m = measure_pair(pair)
                pairs += 1
                shapes.add(_grid(pair['input']).shape)
                structures.add(hashlib.sha256(repr(m['signature']).encode()).hexdigest())
                palettes.add(m['palette'])
                profiles.add(m['weights'])
                rooms.add(m['rooms'])
                ports.add(m['ports'])
                signs.update(m['area_signs'])
                winners.add(m['winner_index'])
                nonrect += m['nonrectangular'] > 0
                scan += m['scan_wrong']
                counterfactual += m['swap_reverses']
    assert len(shapes) >= 30 and len(structures) >= 100 and len(palettes) >= 30
    assert len(profiles) >= 8 and len(rooms) >= 5 and len(ports) >= 5
    assert signs == {-1, 0, 1} and winners == {0, 1}
    assert nonrect >= 20 and scan >= 42 and counterfactual >= 42
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(900+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(777), generate_puzzle(777)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    pair = a['test'][0]
    pred, info = _analyze(pair['input'])
    # Relabeling all roles is equivariant; ballot rearrangement cannot matter.
    perm = np.array([9, 4, 7, 0, 6, 2, 8, 1, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(perm[pair['input']]), perm[pred])
    moved = pair['input'].copy()
    d = info['donor']
    r, c, R, C = d['bbox']
    for i, points in enumerate(d['rooms']):
        if i != d['council']:
            xy = points + (r, c)
            vals = np.sort(moved[tuple(xy.T)])
            moved[tuple(xy.T)] = vals
    assert np.array_equal(solve(moved), pred)
    # Reject both a wrong council label and destruction of a preserved wall.
    for location in (tuple(np.argwhere(info['target']['ids'] == info['target']['council'])[0]),
                     tuple(np.argwhere(info['target']['crop'] == info['wall'])[0])):
        wrong = {s: [dict(x) for x in a[s]] for s in a}
        bad = pred.copy()
        bad[location] = (int(bad[location])+1) % 10
        wrong['test'][0] = {'input': pair['input'], 'output': bad}
        try:
            validate_generated_puzzle(wrong)
        except AssertionError:
            continue
        raise AssertionError('incorrect answer accepted')
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), structures=len(structures),
                palettes=len(palettes), weight_profiles=len(profiles), room_counts=sorted(rooms),
                port_counts=sorted(ports), nonrectangular_pairs=nonrect,
                scan_order_counterexamples=scan, correspondence_changes_council=counterfactual)
