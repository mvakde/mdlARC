"""Ballot beams: local source majorities meet through keyed ray routing.

No parent code, example fixtures, seed state, or generation metadata is used by
solve. Only Python's standard library and NumPy are required.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # clockwise: N, E, S, W


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((0 <= a) & (a <= 9)), "grid colors must be integers 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    h, w = mask.shape
    ids = np.full((h, w), -1, dtype=np.int16)
    boxes, groups = [], []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if ids[r, c] >= 0:
            continue
        index = len(groups)
        ids[r, c] = index
        todo, cells = [(r, c)], []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dr, dc in DIRS:
                a, b = x + dr, y + dc
                if 0 <= a < h and 0 <= b < w and mask[a, b] and ids[a, b] < 0:
                    ids[a, b] = index
                    todo.append((a, b))
        rs, cs = zip(*cells)
        box = min(rs), max(rs) + 1, min(cs), max(cs) + 1
        _require(len(cells) == (box[1] - box[0]) * (box[3] - box[2]),
                 "object must be a filled rectangle")
        boxes.append(box)
        groups.append(cells)
    return boxes, groups, ids


def _decode(input_grid):
    grid = _grid(input_grid)
    neutral = int(grid[0, 0])
    _require(np.all(grid[0] == neutral), "missing top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == neutral, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(grid) - 1, "invalid instruction band")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "key rows must repeat")
    row = band[0]
    _require(row[0] == row[-1] == neutral, "missing key walls")
    runs, c = [], 0
    while c < len(row):
        if row[c] == neutral:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] != neutral:
            c += 1
        runs.append((start, c))
    _require(1 <= len(runs) <= 3 and all(b - a >= 3 for a, b in runs),
             "key needs one to three legible panels")
    background = int(row[runs[0][0] + 1])
    _require(background != neutral, "ambiguous background")
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == background), "inconsistent key interior")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == background) != (right == background), "ambiguous key bar")
        color, side = (right, 1) if left == background else (left, -1)
        _require(color not in (neutral, background) and color not in turns,
                 "conflicting obstacle key")
        turns[color] = side
    field = grid[bottom + 1:].copy()
    signals = tuple(sorted(set(map(int, np.unique(field))) -
                           {neutral, background} - set(turns)))
    _require(len(signals) == 2, "field needs two signal colors")
    boxes, panels, ids = _components(np.isin(field, signals))
    _require(len(panels) >= 2, "field needs multiple panels")
    labels, counts = [], []
    for cells in panels:
        n0 = sum(field[p] == signals[0] for p in cells)
        n1 = len(cells) - n0
        _require(n0 > 0 and n1 > 0 and n0 != n1, "panel needs a strict mixed majority")
        labels.append(int(n1 > n0))
        counts.append((int(n0), int(n1)))
    ports = []
    for rr, cc in np.argwhere(field == neutral):
        r, c = int(rr), int(cc)
        adjacent = []
        for d, (dr, dc) in enumerate(DIRS):
            x, y = r - dr, c - dc  # source is inward from outward direction d
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and ids[x, y] >= 0:
                adjacent.append((int(ids[x, y]), d, int(field[x, y])))
        _require(len(adjacent) == 1, "socket must have one source and normal")
        owner, direction, pixel = adjacent[0]
        ports.append((owner, r, c, direction, signals.index(pixel)))
    _require({p[0] for p in ports} == set(range(len(panels))), "every panel needs a socket")
    obstacles = []
    for color in turns:
        oboxes, _, _ = _components(field == color)
        obstacles.extend((*box, color) for box in oboxes)
    _require(set(map(int, np.unique(field))) <=
             {background, neutral, *signals, *turns}, "unexpected field color")
    return dict(grid=grid, field=field, crop=bottom + 1, neutral=neutral,
                background=background, signals=signals, turns=turns,
                boxes=boxes, panels=panels, ids=ids, labels=labels,
                counts=counts, ports=ports, obstacles=obstacles,
                key_widths=tuple(b - a + 2 for a, b in runs))


def _trace(info, mapping=None, stop_at_obstacle=False):
    """One immutable-input trace per socket; votes are sets, states are ordered."""
    field = info['field']
    h, w = field.shape
    turns = info['turns']
    if mapping is not None:
        turns = {color: mapping[side] for color, side in turns.items()}
    signal_set = set(info['signals'])
    paths = []
    for owner, sr, sc, initial, pixel in info['ports']:
        r, c, direction = sr, sc, initial
        cells, seen, states, events = set(), set(), [], []
        moved_since_turn = True
        while True:
            state = (r, c, direction)
            _require(state not in seen, "cyclic beam")
            seen.add(state)
            states.append(state)
            cells.add((r, c))
            dr, dc = DIRS[direction]
            x, y = r + dr, c + dc
            if not (0 <= x < h and 0 <= y < w):
                endpoint = 'edge'
                break
            value = int(field[x, y])
            if value in signal_set:
                endpoint = 'panel'
                break
            if value in turns:
                if stop_at_obstacle:
                    endpoint = 'obstacle'
                    break
                _require(moved_since_turn, "successive turns need free travel")
                new_direction = (direction + turns[value]) % 4
                events.append((value, direction, new_direction, r, c))
                direction = new_direction
                moved_since_turn = False
                continue
            _require(value in (info['background'], info['neutral']), "unrecognized collision")
            r, c = x, y
            moved_since_turn = True
        paths.append(dict(owner=owner, pixel=pixel, cells=cells, states=states,
                          events=events, endpoint=endpoint, initial=initial))
    return paths


def _ballots(info, paths, by_panel=False, socket_pixels=False):
    votes = np.zeros((2, *info['field'].shape), dtype=np.int16)
    owner_cells = [set() for _ in info['panels']]
    for path in paths:
        owner = path['owner']
        cells = path['cells']
        if by_panel:
            cells = cells - owner_cells[owner]
            owner_cells[owner].update(cells)
        label = path['pixel'] if socket_pixels else info['labels'][owner]
        for r, c in cells:
            votes[label, r, c] += 1
    return votes


def _paint(info, votes):
    out = info['field'].copy()
    positive = votes.sum(axis=0) > 0
    out[positive & (votes[0] == votes[1])] = info['neutral']
    for i in (0, 1):
        out[votes[i] > votes[1 - i]] = info['signals'][i]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the distributed ballot result using only the supplied grid."""
    info = _decode(input_grid)
    return _paint(info, _ballots(info, _trace(info)))


def _analyze(input_grid):
    info = _decode(input_grid)
    paths = _trace(info)
    votes = _ballots(info, paths)
    distinct = _ballots(info, paths, by_panel=True)
    out = _paint(info, votes)
    contested = np.all(votes > 0, axis=0)
    weighted_opposite = contested & ((votes[0] - votes[1]) *
                                     (distinct[0] - distinct[1]) < 0)
    raw = np.sum(info['field'] == info['signals'][0]) - np.sum(info['field'] == info['signals'][1])
    raw_opposite = weighted_opposite & ((votes[0] - votes[1]) * int(raw) < 0)
    headings = {-1: set(), 1: set()}
    for path in paths:
        for color, incoming, _, _, _ in path['events']:
            headings[info['turns'][color]].add(incoming)
    return dict(info=info, paths=paths, votes=votes, output=out,
                ports=len(paths), panels=len(info['panels']),
                max_turns=max(len(p['events']) for p in paths),
                ties=int(np.sum(contested & (votes[0] == votes[1]))),
                weighted_opposite=int(np.sum(weighted_opposite)),
                raw_opposite=int(np.sum(raw_opposite)),
                contested=int(np.sum(contested)), headings=headings,
                minority_socket=any(p['pixel'] != info['labels'][p['owner']] for p in paths),
                pixel_ablation=not np.array_equal(out, _paint(info, _ballots(info, paths, socket_pixels=True))),
                port_ablation=not np.array_equal(out, _paint(info, distinct)),
                turn_ablation=not np.array_equal(out, _paint(info, _ballots(info, _trace(info, stop_at_obstacle=True)))))


def _pattern(rng, h, w, label, minority_at=(), close=False):
    area = h * w
    minimum = max(1, len(minority_at))
    maximum = (area - 1) // 2
    _require(minimum <= maximum, "not enough space for designated minority")
    if close:
        nminor = maximum
    else:
        mode = int(rng.integers(3))
        if mode == 0:
            nminor = maximum
        elif mode == 1:
            nminor = max(minimum, int(area * rng.uniform(.18, .35)))
        else:
            nminor = minimum
    nminor = min(maximum, max(minimum, nminor))
    forced = {r * w + c for r, c in minority_at}
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        order = list(map(int, rng.permutation(area)))
    elif style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        order = list(map(int, np.argsort((abs(rr - r) + abs(cc - c) + rng.random((h, w)) * .3).ravel())))
    elif style == 2:
        order = list(map(int, np.argsort(((rr + cc) % 2 * area + rr * w + cc).ravel())))
    else:
        order = list(map(int, np.argsort((cc * h + rr).ravel())))
    chosen = list(sorted(forced)) + [x for x in order if x not in forced]
    pattern = np.full(area, 2 + label, dtype=np.int8)
    pattern[chosen[:nminor]] = 3 - label
    return pattern.reshape(h, w)


def _header(rng, width, height, turns):
    keys = list(map(int, rng.permutation(list(turns))))
    sizes = np.full(len(keys), 5, dtype=int)
    extra = width - 5 * len(keys)
    _require(extra >= 0, "key band is too narrow")
    sizes += rng.multinomial(extra, np.full(len(keys), 1 / len(keys)))
    band = np.zeros((height, width), dtype=np.int8)
    band[0] = band[-1] = 1
    col = 0
    for color, size in zip(keys, sizes):
        band[:, col] = band[:, col + size - 1] = 1
        bar = col + 1 if turns[color] == -1 else col + size - 2
        band[1:-1, bar] = color
        col += int(size)
    return band


def _input(rng, field, turns):
    hh = int(rng.integers(3, min(6, 30 - field.shape[0]) + 1))
    return np.vstack((_header(rng, field.shape[1], hh, turns), field))


def _make_pair(rng, field, turns, palette):
    canonical = _input(rng, field, turns)
    a = palette[canonical].astype(np.int8)
    return dict(input=a, output=solve(a))


def _weight_scene(rng, rotation, reflected, winner):
    """Parameterized lesson: 3 rays from one panel beat 2 opposing panels."""
    h, w = int(rng.integers(23, 28)), int(rng.integers(23, 28))
    field = np.zeros((h, w), dtype=np.int8)
    wall_top = int(rng.integers(3, 6))
    wall_depth = int(rng.integers(1, 3))
    row = wall_top + wall_depth
    bx = int(rng.integers(7, 10))
    bwidth = int(rng.integers(4, 7))
    bcol = bx - bwidth // 2
    btop = row + int(rng.integers(3, 5))
    bheight = int(rng.integers(2, 4))
    cwidth = int(rng.integers(3, 7))
    ccol = bx - cwidth // 2
    cheight = int(rng.integers(1, min(3, wall_top - 1)))
    ctop = int(rng.integers(0, max(1, row - cheight - 2)))
    atop = btop + bheight + int(rng.integers(3, 5))
    aheight = int(rng.integers(3, min(5, h - atop - 1) + 1))
    awidth = int(rng.integers(3, 7))
    acol = int(rng.integers(max(bcol + bwidth + 3, 14), w - awidth))
    columns = sorted(map(int, rng.choice(awidth, 3, replace=False)))
    field[atop:atop + aheight, acol:acol + awidth] = _pattern(
        rng, aheight, awidth, winner, [(0, c) for c in columns], close=True)
    loser = 1 - winner
    # Nearly unanimous opposing panels ensure the raw-pixel vote loses.
    bpattern = np.full((bheight, bwidth), 2 + loser, dtype=np.int8)
    bpattern[int(rng.integers(bheight)), int(rng.integers(bwidth))] = 2 + winner
    cpattern = np.full((cheight, cwidth), 2 + loser, dtype=np.int8)
    cpattern[int(rng.integers(cheight)), int(rng.integers(cwidth))] = 2 + winner
    field[btop:btop + bheight, bcol:bcol + bwidth] = bpattern
    field[ctop:ctop + cheight, ccol:ccol + cwidth] = cpattern
    for c in columns:
        field[atop - 1, acol + c] = 1
    field[btop - 1, bx] = 1
    lateral_row = btop + int(rng.integers(bheight))
    field[lateral_row, bcol + bwidth] = 1
    field[ctop + cheight, bx] = 1
    wall_left = acol + columns[0] - int(rng.integers(0, 3))
    wall_right = min(w, acol + columns[-1] + 1 + int(rng.integers(0, 4)))
    field[wall_top:row, wall_left:wall_right] = 4
    west = int(rng.integers(1, bcol - 2))
    field[max(0, row - int(rng.integers(0, 3))):row + int(rng.integers(1, 4)), west:west + 1] = 5
    turns = {4: -1, 5: 1}
    if rng.random() < .35:
        turns[6] = int(rng.choice([-1, 1]))  # unused instruction
    if reflected:
        field = field[:, ::-1].copy()
        turns = {color: -side for color, side in turns.items()}
    field = np.rot90(field, rotation).copy()
    return field, turns


def _place_sources(rng, h, w, panel_count, total_ports):
    field = np.zeros((h, w), dtype=np.int8)
    boxes = []
    for _ in range(panel_count):
        for attempt in range(100):
            ph = int(rng.integers(1, 5))
            pw = int(rng.integers(2, 7))
            if ph * pw < 3:
                continue
            r = int(rng.integers(1, h - ph))
            c = int(rng.integers(1, w - pw))
            # Two-cell buffer permits sockets without ambiguous ownership.
            if np.any(field[max(0, r - 2):min(h, r + ph + 2),
                            max(0, c - 2):min(w, c + pw + 2)]):
                continue
            field[r:r + ph, c:c + pw] = _pattern(rng, ph, pw, int(rng.integers(2)))
            boxes.append((r, r + ph, c, c + pw))
            break
        else:
            return None
    weights = np.ones(panel_count, dtype=int)
    for _ in range(total_ports - panel_count):
        choices = np.flatnonzero(weights < 4)
        if not len(choices):
            return None
        weights[int(rng.choice(choices))] += 1
    for (a, b, c, d), count in zip(boxes, weights):
        candidates = [(a - 1, y) for y in range(c, d)] + [(b, y) for y in range(c, d)]
        candidates += [(x, c - 1) for x in range(a, b)] + [(x, d) for x in range(a, b)]
        candidates = [p for p in candidates if field[p] == 0 and
                      sum(0 <= p[0] + dr < h and 0 <= p[1] + dc < w and
                          field[p[0] + dr, p[1] + dc] in (2, 3) for dr, dc in DIRS) == 1]
        if len(candidates) < count:
            return None
        for i in rng.choice(len(candidates), int(count), replace=False):
            field[candidates[int(i)]] = 1
    return field


def _scene_info(rng, field, turns):
    # Temporary header is only a way to exercise the same visible-role decoder;
    # it is not retained or used to seed the scene construction.
    return _decode(np.vstack((_header(rng, field.shape[1], 3, turns), field)))


def _grow_obstacles(rng, field, turns, target, max_turns):
    info = _scene_info(rng, field, turns)
    paths = _trace(info)
    placed = 0
    for _ in range(target * 25 + 15):
        if placed >= target:
            break
        eligible = []
        for path in paths:
            for r, c, d in path['states'][1:]:
                dr, dc = DIRS[d]
                x, y = r + dr, c + dc
                if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0:
                    eligible.append((x, y, d))
        if not eligible:
            break
        x, y, d = eligible[int(rng.integers(len(eligible)))]
        depth = int(rng.integers(1, 4))
        before, after = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        if d == 0:
            a, b, c, e = max(0, x - depth + 1), x + 1, max(0, y - before), min(field.shape[1], y + after + 1)
        elif d == 2:
            a, b, c, e = x, min(field.shape[0], x + depth), max(0, y - before), min(field.shape[1], y + after + 1)
        elif d == 1:
            a, b, c, e = max(0, x - before), min(field.shape[0], x + after + 1), y, min(field.shape[1], y + depth)
        else:
            a, b, c, e = max(0, x - before), min(field.shape[0], x + after + 1), max(0, y - depth + 1), y + 1
        if np.any(field[a:b, c:e]):
            continue
        # Separated obstacles and a clear source/socket buffer keep objects legible.
        if np.any(field[max(0, a - 1):min(field.shape[0], b + 1),
                        max(0, c - 1):min(field.shape[1], e + 1)]):
            continue
        unused = [color for color in turns if not np.any(field == color)]
        color = int(rng.choice(unused if unused and rng.random() < .75 else list(turns)))
        candidate = field.copy()
        candidate[a:b, c:e] = color
        try:
            candidate_info = _scene_info(rng, candidate, turns)
            candidate_paths = _trace(candidate_info)
            if max(len(p['events']) for p in candidate_paths) > max_turns:
                continue
        except ValueError:
            continue
        field, info, paths = candidate, candidate_info, candidate_paths
        placed += 1
    # Occasionally add a truly unhit keyed rectangle, preserving existing paths.
    if rng.random() < .3:
        visited = {p for path in paths for p in path['cells']}
        for _ in range(20):
            ph, pw = int(rng.integers(1, 4)), int(rng.integers(1, 5))
            a, c = int(rng.integers(field.shape[0] - ph + 1)), int(rng.integers(field.shape[1] - pw + 1))
            b, e = a + ph, c + pw
            if np.any(field[max(0, a - 1):min(field.shape[0], b + 1),
                            max(0, c - 1):min(field.shape[1], e + 1)]):
                continue
            if any((r, cc) in visited for r in range(a, b) for cc in range(c, e)):
                continue
            candidate = field.copy()
            candidate[a:b, c:e] = int(rng.choice(list(turns)))
            try:
                new_paths = _trace(_scene_info(rng, candidate, turns))
                if any(p['cells'] != q['cells'] for p, q in zip(paths, new_paths)):
                    continue
            except ValueError:
                continue
            field = candidate
            break
    return field


def _general_pair(rng, palette, regime, test=False):
    for _ in range(80 if test else 50):
        key_count = int(rng.integers(2, 4))
        h = int(rng.integers(14, 28))
        w = int(rng.integers(max(17, 5 * key_count), 31))
        if regime == 'voters' and test:
            count = int(rng.integers(4, 6))
            ports = int(rng.integers(10, min(14, 4 * count) + 1))
        else:
            count = int(rng.integers(2, 5 if not test else 6))
            ports = int(rng.integers(max(7 if test else 4, count),
                                   min(9 if not test else 12, 4 * count) + 1))
        field = _place_sources(rng, h, w, count, ports)
        if field is None:
            continue
        keys = list(range(4, 4 + key_count))
        sides = [-1, 1] + ([int(rng.choice([-1, 1]))] if key_count == 3 else [])
        rng.shuffle(sides)
        turns = dict(zip(keys, sides))
        target = int(rng.integers(6 if test and regime == 'depth' else 2,
                                  7 if not test else 10))
        limit = 2 if regime == 'depth' and not test else 7
        field = _grow_obstacles(rng, field, turns, target, limit)
        pair = _make_pair(rng, field, turns, palette)
        m = _analyze(pair['input'])
        if not m['contested'] or not m['max_turns']:
            continue
        if regime == 'depth' and test and m['max_turns'] < 3:
            continue
        if test and not (m['port_ablation'] and m['pixel_ablation'] and m['turn_ablation']):
            # The held-out answer must actually need payload majority, socket
            # multiplicity and recursive geometry, not merely contain them.
            continue
        return pair
    raise RuntimeError("bounded random scene construction exhausted")


def _turn_hypotheses(train):
    fits = []
    for left in (-1, 1):
        for right in (-1, 1):
            mapping = {-1: left, 1: right}
            valid = True
            for pair in train:
                info = _decode(pair['input'])
                try:
                    out = _paint(info, _ballots(info, _trace(info, mapping)))
                    if not np.array_equal(out, pair['output']):
                        valid = False
                        break
                except ValueError:
                    valid = False
                    break
            if valid:
                fits.append((left, right))
    return fits


def _headings(metrics):
    return {side: set().union(*(m['headings'][side] for m in metrics)) for side in (-1, 1)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10),
                                (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    varying_palette = bool(rng.random() < .7)
    regime = str(rng.choice(['voters', 'depth']))
    rotation = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    winner = int(rng.integers(2))

    def colors():
        return rng.permutation(10) if varying_palette else palette

    # The structured witness is evidence, not the construction template for tests.
    field, turns = _weight_scene(rng, rotation, reflected, winner)
    first = _make_pair(rng, field, turns, colors())
    first_m = _analyze(first['input'])
    _require(first_m['raw_opposite'] > 0 and first_m['ties'] > 0 and
             first_m['pixel_ablation'], "teaching witness failed")
    train, metrics = [first], [first_m]
    seen = {_signature(first['input'])}
    for i in range(1, int(num_train)):
        chosen = None
        for _ in range(16 if i == 1 else 4):
            pair = _general_pair(rng, colors(), regime)
            m = _analyze(pair['input'])
            if _signature(pair['input']) in seen:
                continue
            if i == 1 and not all(len(x) >= 2 for x in _headings(metrics + [m]).values()):
                continue
            chosen = pair, m
            break
        if chosen is None:
            field, turns = _weight_scene(rng, (rotation + 1) % 4, reflected, 1 - winner)
            pair = _make_pair(rng, field, turns, colors())
            chosen = pair, _analyze(pair['input'])
        pair, m = chosen
        _require(_signature(pair['input']) not in seen, "duplicate demonstration")
        seen.add(_signature(pair['input']))
        train.append(pair)
        metrics.append(m)
    rng.shuffle(train)
    test = []
    for _ in range(int(num_test)):
        for attempt in range(8):
            pair = _general_pair(rng, colors(), regime, test=True)
            if _signature(pair['input']) not in seen:
                test.append(pair)
                seen.add(_signature(pair['input']))
                break
        else:
            raise RuntimeError("unique test construction exhausted")
    puzzle = dict(train=train, test=test)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, "duplicate puzzle input")
            seen.add(signature)
            m = _analyze(inp)
            info = m['info']
            _require(np.array_equal(m['output'], out), "incorrect ballot output")
            _require(out.shape == info['field'].shape, "incorrect field crop")
            objects = np.isin(info['field'], [*info['signals'], *info['turns']])
            _require(np.array_equal(out[objects], info['field'][objects]), "source or obstacle altered")
            visited = m['votes'].sum(axis=0) > 0
            _require(np.array_equal(out[~visited], info['field'][~visited]), "unvisited pixel altered")
            _require(not np.array_equal(out, info['field']), "identity field transformation")
            _require(2 <= m['panels'] <= 5 and 4 <= m['ports'] <= 14, "unsupported generated counts")
            _require(m['contested'] > 0 and m['max_turns'] > 0, "missing concept interaction")
            metrics[split].append(m)
    train, test = metrics['train'], metrics['test']
    _require(any(m['raw_opposite'] > 0 for m in train), "missing weighted/raw vote counterexample")
    _require(any(m['ties'] > 0 for m in train), "ties are undemonstrated")
    _require(any(m['pixel_ablation'] for m in train) and any(m['port_ablation'] for m in train),
             "parent A is not necessary in demonstrations")
    _require(any(m['turn_ablation'] for m in train), "parent B is not necessary in demonstrations")
    _require(all(m['port_ablation'] and m['pixel_ablation'] and m['turn_ablation'] for m in test),
             "a test answer can omit an inherited mechanism")
    _require(all(len(x) >= 2 for x in _headings(train).values()), "relative turns need multiple headings")
    _require(max(m['max_turns'] for m in train) >= 2, "recursive tracing is undemonstrated")
    _require(_turn_hypotheses(puzzle['train']) == [(-1, 1)], "ambiguous turn demonstration")
    count_shift = max(m['ports'] for m in train) <= 9 and min(m['ports'] for m in test) >= 10
    depth_shift = max(m['max_turns'] for m in train) <= 2 and min(m['max_turns'] for m in test) >= 3
    _require(count_shift or depth_shift, "missing measured train/test shift")
    return None


def _semantic_checks():
    # A small hand-calculated crossing: two sources, one vote each, neutral tie.
    # This fixture is newly authored, not a parent pair or another solver.
    field = np.zeros((9, 10), dtype=np.int8)
    field[5, 1:4] = [2, 2, 3]
    field[1, 6:9] = [3, 2, 3]
    field[4, 2] = field[1, 5] = 1
    expected = field.copy()
    expected[:5, 2] = 2
    expected[1, :6] = 3
    expected[1, 2] = 1
    band = _header(np.random.default_rng(819), 10, 3, {4: -1, 5: 1})
    _require(np.array_equal(solve(np.vstack((band, field))), expected),
             "hand-calculated two-source tie failed")
    # Geometry that would cycle, and geometry with two turns without travel.
    for cyclic in (True, False):
        field = np.zeros((14, 14), dtype=np.int8)
        field[4, 1:4] = [2, 2, 3]
        field[4, 4] = 1
        field[10, 10:13] = [3, 2, 3]
        field[9, 11] = 1
        if cyclic:
            for r, c in ((4, 7), (7, 6), (6, 3), (3, 4)):
                field[r, c] = 4
            turns = {4: 1}
        else:
            field[4, 5] = 4
            field[5, 4] = 5
            turns = {4: 1, 5: -1}
        inp = np.vstack((_header(np.random.default_rng(820), 14, 3, turns), field))
        try:
            solve(inp)
        except ValueError as error:
            _require(('cyclic' if cyclic else 'free travel') in str(error),
                     "negative fixture failed for an unrelated reason")
        else:
            raise AssertionError("invalid trajectory accepted")


def self_test():
    """Bounded family checks over 42 seeds; no independent semantic claim."""
    shapes, palettes, layouts, signatures = set(), set(), set(), set()
    source_counts, port_counts, turn_depths, margins, local_margins = set(), set(), set(), set(), set()
    directions, endpoints, shift_kinds = set(), set(), set()
    total_ties = weighted = repeated = unhit = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        train_m, test_m = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                prediction = solve(pair['input'])
                _require(np.array_equal(before, pair['input']), "solve mutates its input")
                _require(np.array_equal(prediction, pair['output']), "solve mismatch")
                m = _analyze(pair['input'])
                info = m['info']
                (train_m if split == 'train' else test_m).append(m)
                shapes.add(pair['input'].shape)
                palettes.add((info['background'], info['neutral'], *info['signals']))
                layouts.add((info['field'].shape, tuple(info['boxes']),
                             tuple((p[1], p[2], p[3]) for p in info['ports']), tuple(info['obstacles'])))
                source_counts.add(m['panels'])
                port_counts.add(m['ports'])
                turn_depths.add(m['max_turns'])
                margins.update(map(int, np.abs(m['votes'][0] - m['votes'][1])[m['votes'].sum(axis=0) > 0]))
                local_margins.update(abs(a - b) for a, b in info['counts'])
                directions.update(p['initial'] for p in m['paths'])
                endpoints.update(p['endpoint'] for p in m['paths'])
                total_ties += m['ties'] > 0
                weighted += m['weighted_opposite'] > 0
                repeated += len(info['obstacles']) > len({x[4] for x in info['obstacles']})
                hit_positions = {(r + DIRS[d][0], c + DIRS[d][1])
                                 for p in m['paths'] for _, d, _, r, c in p['events']}
                unhit += any(not any(a <= r < b and c <= y < d for r, y in hit_positions)
                             for a, b, c, d, _ in info['obstacles'])
        if max(x['ports'] for x in train_m) < min(x['ports'] for x in test_m):
            shift_kinds.add('voters')
        if max(x['max_turns'] for x in train_m) < min(x['max_turns'] for x in test_m):
            shift_kinds.add('depth')
    _require(len(signatures) == 42 and len(layouts) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 50 and len(palettes) >= 70, "insufficient geometry/palette variation")
    _require(source_counts == {2, 3, 4, 5} and len(port_counts) >= 9, "missing count variation")
    _require(max(turn_depths) >= 4 and len(turn_depths) >= 4, "missing route-depth variation")
    _require({0, 1, 2, 3} <= directions and endpoints == {'edge', 'panel'}, "missing headings/endpoints")
    _require({0, 1, 2, 3} <= margins and len(local_margins) >= 8, "missing voting margins")
    _require(total_ties >= 42 and weighted >= 42 and repeated >= 20 and unhit >= 3, "missing interaction coverage")
    _require(shift_kinds == {'voters', 'depth'}, "missing shift regimes")
    _semantic_checks()
    # Determinism, serialized-list support, wrong-output rejection and palette covariance.
    a, b = generate_puzzle(714, 2, 1), generate_puzzle(714, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p), "nondeterminism")
    validate_generated_puzzle({s: [{k: v.tolist() for k, v in p.items()} for p in a[s]] for s in a})
    p = a['train'][0]
    permutation = np.random.default_rng(1001).permutation(10)
    _require(np.array_equal(solve(permutation[p['input']]), permutation[p['output']]), "palette covariance failed")
    # Rearranging minority pixels with unchanged counts preserves all beam paint.
    changed_input, expected = p['input'].copy(), p['output'].copy()
    q = _decode(changed_input)
    r0, r1, c0, c1 = q['boxes'][0]
    rectangle = changed_input[q['crop'] + r0:q['crop'] + r1, c0:c1]
    rectangle[:] = np.sort(rectangle, axis=None).reshape(rectangle.shape)
    expected[r0:r1, c0:c1] = rectangle
    _require(np.array_equal(solve(changed_input), expected), "local majority depends on arrangement")
    changed = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    m = _analyze(changed['test'][0]['input'])
    r, c = np.argwhere(m['votes'].sum(axis=0) > 0)[0]
    changed['test'][0]['output'][r, c] = m['info']['background']
    try:
        validate_generated_puzzle(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong ballot accepted")
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), layouts=len(layouts),
                palettes=len(palettes), source_counts=sorted(source_counts),
                port_counts=sorted(port_counts), route_depths=sorted(turn_depths),
                vote_margins=sorted(margins), local_margins=sorted(local_margins),
                tie_scenes=total_ties, weighted_scenes=weighted,
                repeated_obstacle_scenes=repeated, unhit_obstacle_scenes=unhit,
                shifts=sorted(shift_kinds), independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
