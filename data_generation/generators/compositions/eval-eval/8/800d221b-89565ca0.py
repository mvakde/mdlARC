"""Gated chamber voting: reconstructed rooms are simultaneous peer collectors.

Only NumPy and the Python standard library are used. The construction records
produce labels; solve() reconstructs everything from the returned input pixels.
"""

from collections import Counter
import json

import numpy as np


EMPTY, WALL, PORT, NOISE = 0, 5, 8, 9
SIGNAL_POOL = (1, 2, 3, 4, 6, 7)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (
    0, 313241998, 856327708, 232247843, 10709988, 567664909,
    425485274, 978312624, 724227928, 714226553, 972341408,
    823030399, 214921514, 801274697, 123972013, 912184037,
    252496035, 648003071, 16062258, 108150404, 754586465,
    444108875, 919940337, 245451960, 91958298, 100850567,
    22718586, 407401626, 870668953, 492653022, 989411726,
    108604946, 187029813, 911353156, 309696003, 614026544,
    509810330, 259592217, 629183598, 726446822, 966438243,
    681221308,
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    """Four-connected components and their pixel labels."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    components = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(components)
        labels[r, c] = index
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in DIRS:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                        and labels[yy, xx] < 0):
                    labels[yy, xx] = index
                    stack.append((yy, xx))
        components.append(cells)
    return components, labels


def _bounded_regions(walls):
    """Actual elementary rooms, not products of observed line coordinates."""
    components, _ = _components(~walls)
    h, w = walls.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    rooms = []
    for cells in components:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        index = len(rooms)
        for p in cells:
            labels[p] = index
        rooms.append(cells)
    return rooms, labels


def _walls_from_boxes(shape, boxes):
    walls = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        walls[t, l:r + 1] = True
        walls[b, l:r + 1] = True
        walls[t:b + 1, l] = True
        walls[t:b + 1, r] = True
    return walls


def _repair_wall_mask(g):
    """A noise pixel is a wall only with opposite immediate wall/port support.

    Support is read from the original image, never from iterative completion.
    Thus unrelated noise is retained, and no background gap can become a wall.
    """
    visible = (g == WALL) | (g == PORT)
    padded = np.pad(visible, 1)
    vertical = padded[:-2, 1:-1] & padded[2:, 1:-1]
    horizontal = padded[1:-1, :-2] & padded[1:-1, 2:]
    repair = (g == NOISE) & (vertical | horizontal)
    return visible | repair, repair


def _decision(majorities, weights, signals):
    """One self vote, plus one neighbor vote per port; ties keep self."""
    votes = np.zeros((len(majorities), 2), dtype=int)
    index = {c: i for i, c in enumerate(signals)}
    for i, own in enumerate(majorities):
        votes[i, index[own]] += 1
        for j, color in enumerate(majorities):
            votes[i, index[color]] += int(weights[i, j])
    winners = [own if a == b else signals[int(b > a)]
               for own, (a, b) in zip(majorities, votes)]
    return winners, votes


def _analyze(input_grid):
    g = _grid(input_grid)
    colors = set(map(int, np.unique(g)))
    signals = tuple(sorted(colors - {EMPTY, WALL, PORT, NOISE}))
    if len(signals) != 2 or not {EMPTY, WALL, PORT, NOISE} <= colors:
        raise ValueError("input needs the four neutral roles and two signals")
    walls, repair = _repair_wall_mask(g)
    wall_cells = np.argwhere(walls)
    if not len(wall_cells):
        raise ValueError("missing frame")
    top, left = map(int, wall_cells.min(axis=0))
    bottom, right = map(int, wall_cells.max(axis=0))
    if not (walls[top, left:right + 1].all()
            and walls[bottom, left:right + 1].all()
            and walls[top:bottom + 1, left].all()
            and walls[top:bottom + 1, right].all()):
        raise ValueError("the outer frame must be complete after supported repair")

    rooms, labels = _bounded_regions(walls)
    if not rooms:
        raise ValueError("no closed chambers")
    boxes, panel_boxes, panels, majorities, panel_counts = [], [], [], [], []
    for cells in rooms:
        coords = np.asarray(cells)
        t, l = map(int, coords.min(axis=0))
        b, r = map(int, coords.max(axis=0))
        if len(cells) != (b - t + 1) * (r - l + 1) or min(b - t + 1, r - l + 1) < 3:
            raise ValueError("each elementary chamber must be a rectangle at least 3 by 3")
        boxes.append((t - 1, l - 1, b + 1, r + 1))
        signal_cells = [p for p in cells if int(g[p]) in signals]
        if len(signal_cells) < 3:
            raise ValueError("chamber lacks a ballot panel")
        pp = np.asarray(signal_cells)
        pt, pl = map(int, pp.min(axis=0))
        pb, pr = map(int, pp.max(axis=0))
        if len(signal_cells) != (pb - pt + 1) * (pr - pl + 1):
            raise ValueError("each chamber has exactly one filled rectangular panel")
        counts = [sum(int(g[p]) == color for p in signal_cells) for color in signals]
        if min(counts) == 0 or counts[0] == counts[1]:
            raise ValueError("panel votes must include both signals and have a strict majority")
        if not any(g[p] == EMPTY for p in cells):
            raise ValueError("chamber must have some empty interior to fill")
        panel_boxes.append((pt, pl, pb, pr))
        panels.append(signal_cells)
        panel_counts.append(counts)
        majorities.append(signals[int(counts[1] > counts[0])])

    if not np.array_equal(walls, _walls_from_boxes(g.shape, boxes)):
        raise ValueError("unexplained walls or non-partition geometry")
    padded = np.pad(walls, 1)
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    straight = (up & down & ~west & ~east) | (west & east & ~up & ~down)
    if np.any(repair & ~straight):
        raise ValueError("wall damage must be on straight segments, away from junctions")
    rp = np.pad(repair, 1)
    adjacent_damage = rp[:-2, 1:-1] | rp[2:, 1:-1] | rp[1:-1, :-2] | rp[1:-1, 2:]
    if np.any(repair & adjacent_damage):
        raise ValueError("wall interruptions must be isolated")
    if np.any(np.isin(g, signals) & (labels < 0)):
        raise ValueError("ballots must be inside chambers")

    n = len(rooms)
    weights = np.zeros((n, n), dtype=int)
    contacts = np.zeros((n, n), dtype=int)
    ports = []
    h, w = g.shape
    for r, c in wall_cells:
        r, c = int(r), int(c)
        neighbors = [int(labels[r + dr, c + dc])
                     if 0 <= r + dr < h and 0 <= c + dc < w else -1
                     for dr, dc in DIRS]
        up, down, west, east = neighbors
        pair = None
        if up >= 0 and down >= 0 and up != down and west < 0 and east < 0:
            pair = up, down
        elif west >= 0 and east >= 0 and west != east and up < 0 and down < 0:
            pair = west, east
        if pair is not None:
            a, b = pair
            contacts[a, b] += 1
            contacts[b, a] += 1
        if g[r, c] == PORT:
            if pair is None:
                raise ValueError("a port must lie on a straight shared wall away from junctions")
            weights[a, b] += 1
            weights[b, a] += 1
            ports.append(((r, c), a, b))
    if not ports:
        raise ValueError("at least one internal port is required")
    winners, votes = _decision(majorities, weights, signals)
    return dict(grid=g, signals=signals, walls=walls, repair=repair, rooms=rooms,
                labels=labels, boxes=boxes, panels=panels, panel_boxes=panel_boxes,
                panel_counts=np.array(panel_counts), majorities=majorities,
                weights=weights, contacts=contacts, ports=ports, winners=winners,
                votes=votes, outer_box=(top, left, bottom, right))


def _output(info):
    g = info["grid"]
    result = g.copy()
    result[info["repair"]] = WALL
    for i, winner in enumerate(info["winners"]):
        result[(info["labels"] == i) & (g == EMPTY)] = winner
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire output from the supplied input, without examples/state."""
    return _output(_analyze(input_grid))


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 4) * ((r - l) // 4)


def _offsets(parts, span, rng):
    gaps = np.full(parts, 4, dtype=int)
    for _ in range(span - parts * 4):
        gaps[int(rng.integers(parts))] += 1
    return [0] + list(map(int, np.cumsum(gaps)))


def _partial(walls, box):
    t, l, b, r = box
    # Any wall row/column whose segment stops inside the outer frame is partial.
    for y in range(t + 1, b):
        line = walls[y, l:r + 1]
        if np.any(line[1:-1] & line[:-2]) and not line.all():
            return True
    for x in range(l + 1, r):
        line = walls[t:b + 1, x]
        if np.any(line[1:-1] & line[:-2]) and not line.all():
            return True
    return False


def _partition(rng, height, width, count, style):
    outer = (0, 0, height - 1, width - 1)
    if _capacity(outer) < count:
        return None
    if style in ("strip", "grid"):
        factors = [(nr, count // nr) for nr in range(1, count + 1)
                   if count % nr == 0 and 4 * nr < height
                   and 4 * (count // nr) < width
                   and (style == "grid" or nr == 1 or nr == count)]
        if factors:
            nr, nc = factors[int(rng.integers(len(factors)))]
            ys, xs = _offsets(nr, height - 1, rng), _offsets(nc, width - 1, rng)
            return [(ys[i], xs[j], ys[i + 1], xs[j + 1])
                    for i in range(nr) for j in range(nc)]
    boxes = [outer]
    first_axis = None
    for step in range(count - 1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if style == "partial" and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 4, end - 3):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else (
                        (t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                    if capacity + sum(_capacity(x) for x in halves) >= count:
                        choices.append((i, axis, halves))
        if not choices:
            return None
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        first_axis = axis if step == 0 else first_axis
        boxes[i:i + 1] = halves
    if style == "partial" and not _partial(_walls_from_boxes((height, width), boxes), outer):
        return None
    return boxes


def _panel(rng, height, width, majority, minority):
    area = height * width
    band = int(rng.integers(3))
    amount = ((area - 1) // 2 if band == 0 else
              max(1, int(area * rng.uniform(.15, .35))) if band == 1 else 1)
    amount = min(amount, (area - 1) // 2)
    rr, cc = np.indices((height, width))
    style = int(rng.integers(4))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        y, x = int(rng.integers(height)), int(rng.integers(width))
        order = np.argsort((abs(rr - y) + abs(cc - x) + rng.random((height, width)) * .3).ravel())
    elif style == 2:
        order = np.argsort((((rr + cc) % 2) * area + rr * width + cc).ravel())
    else:
        order = np.argsort((rr * width + cc if rng.random() < .5 else cc * height + rr).ravel())
    pattern = np.full(area, majority, dtype=np.int8)
    pattern[order[:amount]] = minority
    return pattern.reshape(height, width)


def _gate_candidates(walls, labels):
    h, w = walls.shape
    result = {}
    for r, c in np.argwhere(walls):
        r, c = int(r), int(c)
        vals = [int(labels[r + dy, c + dx])
                if 0 <= r + dy < h and 0 <= c + dx < w else -1
                for dy, dx in DIRS]
        a, b, d, e = vals
        pair = (a, b) if a >= 0 and b >= 0 and a != b and d < 0 and e < 0 else (
            (d, e) if d >= 0 and e >= 0 and d != e and a < 0 and b < 0 else None)
        if pair is not None:
            result.setdefault(tuple(sorted(pair)), []).append((r, c))
    return result


def _draw_scene(rng, count, signals, style, require_damage=False, paired_ties=False):
    """Construct one scene and its label from explicit boxes, not solve()."""
    for _ in range(50):
        fh, fw = map(int, rng.integers(5, 31, size=2))
        boxes = _partition(rng, fh, fw, count, style)
        if boxes is not None:
            break
    else:
        return None
    tm = int(rng.integers(min(3, 30 - fh) + 1))
    bm = int(rng.integers(min(3, 30 - fh - tm) + 1))
    lm = int(rng.integers(min(3, 30 - fw) + 1))
    rm = int(rng.integers(min(3, 30 - fw - lm) + 1))
    shape = fh + tm + bm, fw + lm + rm
    boxes = [(t + tm, l + lm, b + tm, r + lm) for t, l, b, r in boxes]
    walls = _walls_from_boxes(shape, boxes)
    labels = np.full(shape, -1, dtype=np.int16)
    for i, (t, l, b, r) in enumerate(boxes):
        labels[t + 1:b, l + 1:r] = i
    g = np.full(shape, EMPTY, dtype=np.int8)
    g[walls] = WALL
    majorities = list(map(int, rng.choice(signals, size=count)))
    if len(set(majorities)) == 1:
        slot = int(rng.integers(count))
        majorities[slot] = next(c for c in signals if c != majorities[slot])
    for i, (t, l, b, r) in enumerate(boxes):
        ih, iw = b - t - 1, r - l - 1
        candidates = [(ph, pw) for ph in range(1, min(8, ih) + 1)
                      for pw in range(1, min(8, iw) + 1)
                      if 3 <= ph * pw <= ih * iw - 2]
        # Sometimes tiny panels coexist with large panels in the same map.
        if rng.random() < .35:
            candidates = [(ph, pw) for ph, pw in candidates if ph * pw <= 6]
        ph, pw = candidates[int(rng.integers(len(candidates)))]
        pt, pl = int(rng.integers(t + 1, b - ph + 1)), int(rng.integers(l + 1, r - pw + 1))
        majority = majorities[i]
        minority = next(c for c in signals if c != majority)
        g[pt:pt + ph, pl:pl + pw] = _panel(rng, ph, pw, majority, minority)

    candidates = _gate_candidates(walls, labels)
    weights = np.zeros((count, count), dtype=int)
    total_ports = 0
    tie_pair = None
    if paired_ties:
        options = [pair for pair in candidates if majorities[pair[0]] != majorities[pair[1]]]
        tie_pair = options[int(rng.integers(len(options)))]
    for (a, b), sites in candidates.items():
        if tie_pair is not None and (a in tie_pair or b in tie_pair):
            if (a, b) == tie_pair:
                p = sites[int(rng.integers(len(sites)))]
                g[p] = PORT
                weights[a, b] = weights[b, a] = 1
                total_ports += 1
            continue
        if rng.random() < .25:
            continue
        # A contact can have several separate windows; every pixel is a vote.
        chosen = set()
        for _ in range(int(rng.integers(1, 4))):
            length = int(rng.integers(1, min(3, len(sites)) + 1))
            start = int(rng.integers(len(sites) - length + 1))
            chosen.update(sites[start:start + length])
        for p in chosen:
            g[p] = PORT
        weights[a, b] = weights[b, a] = len(chosen)
        total_ports += len(chosen)
    if not total_ports:
        a, b = next(iter(candidates))
        p = candidates[a, b][int(rng.integers(len(candidates[a, b])))]
        g[p] = PORT
        weights[a, b] = weights[b, a] = 1

    # Damage only straight non-port wall pixels, never adjacent erasures.
    padded = np.pad(walls, 1)
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    west, east = padded[1:-1, :-2], padded[1:-1, 2:]
    straight = (up & down & ~west & ~east) | (west & east & ~up & ~down)
    eligible = [(int(r), int(c)) for r, c in np.argwhere(straight & (g == WALL))]
    rng.shuffle(eligible)
    amount = int(rng.integers(1, max(2, len(eligible) // 9 + 1))) if (
        require_damage or rng.random() < .75) else 0
    damaged = []
    for p in eligible:
        if len(damaged) >= amount:
            break
        if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1 for q in damaged):
            damaged.append(p)
            g[p] = NOISE
    if require_damage and not damaged:
        return None

    # Irrelevant noise may lie outside or inside rooms, but cannot impersonate a
    # supported wall gap. Leave every chamber with at least one empty cell.
    visible = (g == WALL) | (g == PORT)
    padded = np.pad(visible, 1)
    supported = ((padded[:-2, 1:-1] & padded[2:, 1:-1])
                 | (padded[1:-1, :-2] & padded[1:-1, 2:]))
    noise_sites = [(int(r), int(c)) for r, c in np.argwhere((g == EMPTY) & ~supported)]
    rng.shuffle(noise_sites)
    amount = int(rng.integers(1, min(14, len(noise_sites)) + 1))
    placed = 0
    for p in noise_sites:
        i = int(labels[p])
        if i >= 0 and np.count_nonzero((labels == i) & (g == EMPTY)) <= 1:
            continue
        g[p] = NOISE
        placed += 1
        if placed == amount:
            break

    winners, _ = _decision(majorities, weights, signals)
    out = g.copy()
    for p in damaged:
        out[p] = WALL
    for i, color in enumerate(winners):
        out[(labels == i) & (g == EMPTY)] = color
    pair = {"input": g, "output": out}
    # Pixel-derived recovery must reproduce the full intended partition and
    # port matrix, not just happen to have the same room count or output.
    info = _analyze(g)
    if not np.array_equal(info["walls"], walls):
        return None
    expected_by_box = {box: i for i, box in enumerate(boxes)}
    remap = [expected_by_box[b] for b in info["boxes"]]
    if not np.array_equal(info["weights"], weights[np.ix_(remap, remap)]):
        return None
    if info["majorities"] != [majorities[i] for i in remap]:
        return None
    if not np.array_equal(_output(info), out):
        return None
    return pair


def _metrics(info):
    signals, local = info["signals"], info["majorities"]
    correct = info["winners"]
    weights, contacts = info["weights"], info["contacts"]
    unit_winners, _ = _decision(local, (weights > 0).astype(int), signals)
    window_weights = np.zeros_like(weights)
    owner = {p: (a, b) for p, a, b in info["ports"]}
    for cells in _components(info["grid"] == PORT)[0]:
        a, b = owner[cells[0]]
        assert all(set(owner[p]) == {a, b} for p in cells)
        window_weights[a, b] += 1
        window_weights[b, a] += 1
    window_winners, _ = _decision(local, window_weights, signals)
    ungated, _ = _decision(local, (contacts > 0).astype(int), signals)
    raw_votes = (np.eye(len(local), dtype=int) + weights) @ info["panel_counts"]
    raw_winners = [own if a == b else signals[int(b > a)]
                   for own, (a, b) in zip(local, raw_votes)]
    current = list(local)
    for i, own in enumerate(local):
        votes = Counter({own: 1})
        for j, color in enumerate(current):
            votes[color] += int(weights[i, j])
        a, b = (votes[s] for s in signals)
        current[i] = own if a == b else signals[int(b > a)]
    ties = [i for i, (a, b) in enumerate(info["votes"]) if a == b]
    tie_opposite = any(any(weights[i, j] and local[j] != local[i]
                           for j in range(len(local))) for i in ties)
    tie_colors = {local[i] for i in ties if any(weights[i, j] and local[j] != local[i]
                                               for j in range(len(local)))}
    visible = (info["grid"] == WALL) | (info["grid"] == PORT)
    uncorrected_count = len(_bounded_regions(visible)[0])
    open_port_count = len(_bounded_regions(info["walls"] & (info["grid"] != PORT))[0])
    n = len(local)
    adjacency = tuple(tuple(map(int, row)) for row in (weights > 0))
    area = [len(p) for p in info["panels"]]
    room_dimensions = [(b - t - 1, r - l - 1) for t, l, b, r in info["boxes"]]
    return dict(rooms=n, ports=len(info["ports"]),
                partial=_partial(info["walls"], info["outer_box"]),
                damage=int(info["repair"].sum()),
                noise=int(np.count_nonzero(info["grid"] == NOISE) - info["repair"].sum()),
                weighted_diff=unit_winners != correct,
                window_diff=window_winners != correct,
                raw_diff=raw_winners != correct,
                ungated_diff=ungated != correct,
                sequential_diff=current != correct,
                override=correct != local, both_winners=len(set(correct)) == 2,
                tie=tie_opposite, recovery_diff=uncorrected_count != n,
                both_tie_colors=len(tie_colors) == 2,
                open_ports_diff=open_port_count != n,
                isolated=int(np.count_nonzero(weights.sum(axis=1) == 0)),
                edge=any(x == 0 for x in info["outer_box"][:2]) or
                     info["outer_box"][2] == info["grid"].shape[0] - 1 or
                     info["outer_box"][3] == info["grid"].shape[1] - 1,
                panel_areas=area, room_dimensions=room_dimensions,
                gate_weights=sorted(map(int, weights[np.triu(weights > 0, 1)])),
                adjacency=adjacency,
                decisions=tuple(local) + tuple(correct))


def _make_pair(rng, count, signals, witness):
    # The witnesses constrain semantic effects, never a fixed layout or palette.
    for _ in range(350):
        style = "partial" if witness == "majority" else str(rng.choice(
            ["partial", "partial", "strip", "grid"]))
        pair = _draw_scene(rng, count, signals, style, require_damage=witness == "majority",
                           paired_ties=witness == "interaction")
        if pair is None:
            continue
        m = _metrics(_analyze(pair["input"]))
        if witness == "majority" and not all(m[k] for k in (
            "partial", "recovery_diff", "weighted_diff", "window_diff", "raw_diff",
            "override", "sequential_diff")):
            continue
        if witness == "interaction" and not all(m[k] for k in (
            "both_tie_colors", "ungated_diff", "both_winners")):
            continue
        return pair
    raise RuntimeError("bounded scene search failed to produce the requested witness")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    ceiling = int(rng.integers(4, 7))
    palette = tuple(sorted(map(int, rng.choice(SIGNAL_POOL, size=2, replace=False))))
    local_palette = bool(rng.random() < .5)
    train_counts = list(map(int, rng.integers(3, ceiling + 1, size=nt)))
    # At least two room counts occur in training, even with only two examples.
    if len(set(train_counts)) == 1:
        train_counts[1] = 3 if train_counts[0] != 3 else ceiling
    test_counts = list(map(int, rng.integers(ceiling + 1, ceiling + 4, size=ne)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, counts in (("train", train_counts), ("test", test_counts)):
        for i, count in enumerate(counts):
            signals = tuple(sorted(map(int, rng.choice(SIGNAL_POOL, 2, replace=False)))) if local_palette else palette
            witness = ("majority" if i == 0 else "interaction" if i == 1 else None) if split == "train" else None
            for _ in range(8):
                pair = _make_pair(rng, count, signals, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, pixel-derived decisions, preservation, and demonstration effects."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, measures = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            info = _analyze(g)
            assert np.array_equal(_output(info), out)
            changed = g != out
            permitted = info["repair"] | ((g == EMPTY) & (info["labels"] >= 0))
            assert np.array_equal(changed, permitted), "exactly empty interiors and damaged walls change"
            assert np.array_equal(g[~permitted], out[~permitted]), "panels, gates, noise and exterior persist"
            assert np.array_equal(info["weights"], info["weights"].T)
            assert np.all(np.diag(info["weights"]) == 0)
            assert int(info["weights"].sum()) == 2 * len(info["ports"])
            for i, (a, b) in enumerate(info["votes"]):
                assert a + b == 1 + int(info["weights"][i].sum())
                if a == b:
                    assert info["winners"][i] == info["majorities"][i]
            m = _metrics(info)
            assert m["noise"] > 0 and m["open_ports_diff"]
            measures[split].append(m)
    training = measures["train"]
    for feature in ("partial", "recovery_diff", "weighted_diff", "window_diff", "raw_diff",
                    "override", "tie", "both_tie_colors", "sequential_diff",
                    "ungated_diff", "both_winners"):
        assert any(m[feature] for m in training), f"missing training evidence for {feature}"
    assert len({m["rooms"] for m in training}) >= 2
    assert min(m["rooms"] for m in measures["test"]) > max(m["rooms"] for m in training)


def self_test():
    """Bounded 42-seed variation checks plus semantic mutation/metamorphic checks."""
    # A hand-specified T partition, independent of the randomized constructor:
    # A--B has three gates, B--C one, A--C none. Local labels are 1,2,1.
    # Decisions are 2,1,1: A/B exchange labels, and C retains its tied self vote.
    boxes = [(1, 1, 9, 7), (1, 7, 5, 13), (5, 7, 9, 13)]
    fixture = np.zeros((11, 15), dtype=np.int8)
    fixture[_walls_from_boxes(fixture.shape, boxes)] = WALL
    fixture[3, 3:6] = [1, 1, 2]
    fixture[2:4, 9:12] = [[2, 2, 2], [2, 2, 1]]
    fixture[7, 9:12] = [1, 2, 1]
    fixture[2:5, 7] = PORT
    fixture[5, 10] = PORT
    fixture[1, 4] = NOISE
    fixture[0, 0] = NOISE
    fixture[7, 3] = NOISE
    expected = fixture.copy()
    expected[1, 4] = WALL
    for box, color in zip(boxes, (2, 1, 1)):
        t, l, b, r = box
        block = expected[t + 1:b, l + 1:r]
        block[block == EMPTY] = color
    assert np.array_equal(solve(fixture), expected)
    info = _analyze(fixture)
    assert info["votes"].tolist() == [[1, 3], [4, 1], [1, 1]]
    assert all(_metrics(info)[name] for name in (
        "weighted_diff", "raw_diff", "ungated_diff", "sequential_diff", "tie",
        "partial", "recovery_diff", "open_ports_diff"))
    reduced = fixture.copy()
    reduced[2, 7] = reduced[4, 7] = WALL
    assert solve(reduced)[2, 2] == 1 and expected[2, 2] == 2

    measures, shapes, palettes, structures, signatures = [], set(), set(), set(), set()
    puzzles = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                m = _metrics(info)
                measures.append(m)
                shapes.add(pair["input"].shape)
                palettes.add(info["signals"])
                # Partition topology with room sizes and port weights; no colors,
                # panel patterns, noise positions, or exterior margins included.
                structures.add((tuple(m["room_dimensions"]),
                                tuple(tuple(map(int, row)) for row in info["weights"])))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(palettes) >= 10 and len(structures) >= 140
    assert set(range(3, 10)) <= {m["rooms"] for m in measures}
    assert len({tuple(m["gate_weights"]) for m in measures}) >= 35
    assert len({m["adjacency"] for m in measures}) >= 40
    assert any(m["isolated"] for m in measures)
    assert any(not m["partial"] for m in measures) and any(m["partial"] for m in measures)
    assert any(not m["damage"] for m in measures) and any(m["damage"] >= 5 for m in measures)
    assert any(m["edge"] for m in measures) and any(not m["edge"] for m in measures)
    assert max(max(m["panel_areas"]) for m in measures) >= 35
    sizes = [size for m in measures for size in m["room_dimensions"]]
    assert min(min(s) for s in sizes) == 3 and max(max(s) for s in sizes) >= 15
    assert max(m["ports"] for m in measures) >= 20
    for seed in (HOST_SEEDS[0], HOST_SEEDS[1]):
        a, b = generate_puzzle(seed), generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in a
                   for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123456 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4., 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    # No directional or numeric signal priority: all spatial symmetries commute
    # with solve, and arbitrary bijections of the six signal values commute too.
    recolor = np.arange(10, dtype=np.int8)
    recolor[list(SIGNAL_POOL)] = [7, 6, 4, 3, 2, 1]
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"] + puzzle["test"]:
            for transform in (lambda a: a.T, np.fliplr, np.flipud, lambda a: np.rot90(a)):
                assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
            assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])

    # Altering a label or a preserved minority pixel must be rejected.
    puzzle = puzzles[0]
    info = _analyze(puzzle["test"][0]["input"])
    for p in (next(p for p in info["rooms"][0] if info["grid"][p] == EMPTY),
              info["panels"][0][0]):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzle[s]]
               for s in puzzle}
        old = int(bad["test"][0]["output"][p])
        bad["test"][0]["output"][p] = next(c for c in info["signals"] if c != old)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong output accepted")

    # Moving panel pixels without changing counts cannot affect peer decisions.
    pair = puzzles[0]["train"][0]
    modified = pair["input"].copy()
    expected = pair["output"].copy()
    info = _analyze(modified)
    for t, l, b, r in info["panel_boxes"]:
        block = modified[t:b + 1, l:r + 1]
        block[:] = np.sort(block, axis=None).reshape(block.shape)
        expected[t:b + 1, l:r + 1] = block
    assert np.array_equal(solve(modified), expected)

    return {"seeds": len(HOST_SEEDS), "pairs": len(measures),
            "input_shapes": len(shapes), "signal_palettes": len(palettes),
            "partition_port_structures": len(structures),
            "contact_graphs": len({m["adjacency"] for m in measures}),
            "room_counts": sorted({m["rooms"] for m in measures}),
            "port_totals": sorted({m["ports"] for m in measures}),
            "legal_count_combinations": 30,
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
