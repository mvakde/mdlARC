"""Compass-filtered elections: rigidly compatible objects vote into cavities.

All inference uses visible pixels. A ballot votes once per cavity, regardless
of how many rigid alignments witness its eligibility. Tabs are eligibility
labels, not paint, and one ballot can serve several rooms.
"""
from collections import Counter
from functools import lru_cache

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _norm(points):
    points = tuple(points)
    y, x = min(r for r, c in points), min(c for r, c in points)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in points))


def _point(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return r, c


def _shape(points, t):
    return _norm(_point(p, t) for p in points)


@lru_cache(maxsize=4096)
def _variants(points):
    return tuple(sorted({_shape(points, t) for t in range(8)}))


def _canon(points):
    return min(_variants(_norm(points)))


def _map_transform(cells, t):
    raw = {_point(p, t): int(v) for p, v in cells.items()}
    y, x = min(r for r, c in raw), min(c for r, c in raw)
    return {(r-y, c-x): v for (r, c), v in raw.items()}


def _size(points):
    return max(r for r, c in points)+1, max(c for r, c in points)+1


def _components(mask, diagonal=False):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    steps = (tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if dr or dc) if diagonal else DIRS)
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r+dr, c+dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.add(q)
        result.append(cells)
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _ballot_features(cells, bands):
    colors = set(cells.values()) - set(bands)
    if len(colors) != 1:
        raise ValueError("each ballot needs exactly one non-compass body color")
    color = next(iter(colors))
    body = {p for p, v in cells.items() if v == color}
    h, w = _size(_norm(body))
    mask = np.zeros((h, w), dtype=bool)
    for p in _norm(body):
        mask[p] = True
    if len(_components(mask)) != 1:
        raise ValueError("disconnected ballot body")
    transforms = []
    for t in range(8):
        transformed = {_point(p, t): v for p, v in cells.items()}
        core = [p for p, v in transformed.items() if v == color]
        r0, r1 = min(r for r, c in core), max(r for r, c in core)
        c0, c1 = min(c for r, c in core), max(c for r, c in core)
        labels = []
        for (r, c), v in transformed.items():
            if v == color:
                continue
            facing = [d for d, condition in enumerate(
                (r < r0, c > c1, r > r1, c < c0)) if condition]
            if len(facing) != 1:
                raise ValueError("a tab must project beyond exactly one body side")
            d = facing[0]
            # Tabs occupy the immediately adjacent row or column, with body
            # support behind every tab; they never float or encode corners.
            dr, dc = DIRS[d]
            if transformed.get((r-dr, c-dc)) != color:
                raise ValueError("unsupported directional tab")
            labels.append((d, v))
        if not labels:
            raise ValueError("ballot without compass tabs")
        if any(len({v for dd, v in labels if dd == d}) > 1 for d in range(4)):
            raise ValueError("conflicting identities on a ballot side")
        transforms.append((_norm(core), tuple(sorted(set(labels)))))
    return {"color": color, "transforms": transforms,
            "canonical": _canon(body), "cells": cells}


def _eligible(voter, raw, frame, mode="full"):
    transforms = voter["transforms"]
    if mode == "rotation":
        transforms = transforms[:4]
    elif mode == "identity":
        transforms = transforms[:1]
    for body, labels in transforms:
        geometry = (len(body) == len(raw)) if mode == "area" else body == raw
        if geometry and (mode == "no_compass" or
                         all(frame[d] == v for d, v in labels)):
            return True
    return False


def _parse(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if np.count_nonzero(frequencies == frequencies[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    rectangles = set()
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for cells in _components(a == color):
            if len(cells) < 16:
                continue
            ys, xs = zip(*cells)
            y, x, b, e = min(ys), min(xs), max(ys)+1, max(xs)+1
            box = a[y:b, x:e]
            if (min(box.shape) >= 5 and len(np.unique(box)) == 6
                    and np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                rectangles.add((y, x, b, e, color))
    if len(rectangles) != 1:
        raise ValueError("expected one carrier rectangle enclosing four compass colors")
    y, x, b, e, carrier = next(iter(rectangles))
    box = a[y:b, x:e]
    bands = set(map(int, np.unique(box))) - {bg, carrier}
    if len(bands) != 4:
        raise ValueError("four distinct compass identities required")
    exterior = a.copy()
    exterior[y:b, x:e] = bg
    if np.any(exterior == carrier):
        raise ValueError("carrier is confined to the board")
    if np.any(exterior[max(0, y-1):min(a.shape[0], b+1),
                       max(0, x-1):min(a.shape[1], e+1)] != bg):
        raise ValueError("ballots must have a background margin from the board")
    rings = _components((box != bg) & (box != carrier))
    if len(rings) != len(_components((box != bg) & (box != carrier), diagonal=True)):
        raise ValueError("room rings must have a carrier separation margin")
    cavities, room_records = [], []
    allowed_bg = np.zeros(box.shape, dtype=bool)
    for room_id, cells in enumerate(rings):
        rs, cs = zip(*cells)
        r0, c0, r1, c1 = min(rs), min(cs), max(rs)+1, max(cs)+1
        room = box[r0:r1, c0:c1]
        if min(room.shape) < 3:
            raise ValueError("room too small")
        frame = tuple(map(int, (room[0, 0], room[1, -1], room[-1, 0], room[1, 0])))
        if (set(frame) != bands or not np.all(room[0] == frame[0])
                or not np.all(room[-1] == frame[2])
                or not np.all(room[1:-1, 0] == frame[3])
                or not np.all(room[1:-1, -1] == frame[1])
                or np.any(~np.isin(room[1:-1, 1:-1], (bg, carrier)))):
            raise ValueError("malformed compass room")
        perimeter = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                     if r in (r0, r1-1) or c in (c0, c1-1)}
        if cells != perimeter:
            raise ValueError("room bands must form an isolated closed ring")
        local_cavities = _components(room[1:-1, 1:-1] == bg)
        if len(local_cavities) != len(_components(room[1:-1, 1:-1] == bg, diagonal=True)):
            raise ValueError("cavities must have a separation margin")
        if not local_cavities:
            raise ValueError("empty room")
        for local in local_cavities:
            points = {(r+r0+1, c+c0+1) for r, c in local}
            for p in points:
                allowed_bg[p] = True
            cavities.append({"raw": _norm(points), "cells": points,
                             "frame": frame, "room": room_id,
                             "canonical": _canon(points)})
        room_records.append((r0, c0, r1, c1, frame))
    if not cavities or not np.array_equal(box == bg, allowed_bg):
        raise ValueError("all board background cells must be enclosed cavities")
    voters = []
    objects = _components(exterior != bg)
    if len(objects) != len(_components(exterior != bg, diagonal=True)):
        raise ValueError("ballots must have a background separation margin")
    for cells in objects:
        data = {p: int(a[p]) for p in cells}
        voter = _ballot_features(data, bands)
        if voter["color"] in (bg, carrier) or not any(
                _eligible(voter, cavity["raw"], cavity["frame"]) for cavity in cavities):
            raise ValueError("every exterior ballot must serve at least one cavity")
        voters.append(voter)
    if not voters:
        raise ValueError("missing voters")
    return dict(background=bg, carrier=carrier, bbox=(y, x, b, e), box=box,
                bands=bands, rooms=room_records, cavities=cavities, voters=voters)


def _predict(info, mode="full"):
    output, records = info["box"].copy(), []
    for cavity in info["cavities"]:
        eligible = [i for i, v in enumerate(info["voters"])
                    if _eligible(v, cavity["raw"], cavity["frame"], mode)]
        counts = Counter(info["voters"][i]["color"] for i in eligible)
        if not counts:
            raise ValueError("unsupported cavity")
        high = max(counts.values())
        winners = [c for c, n in counts.items() if n == high]
        if len(winners) != 1:
            raise ValueError("election has no unique plurality")
        winner = winners[0]
        for p in cavity["cells"]:
            output[p] = winner
        records.append({"winner": winner, "counts": counts, "eligible": eligible})
    return output, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from one input only; no palette, seed or training state is read."""
    return _predict(_parse(input_grid))[0]


def _alternative_fails(info, output, mode):
    try:
        return not np.array_equal(_predict(info, mode)[0], output)
    except ValueError:
        return True


def _pool(rng):
    shapes = set()
    for n in range(3, 9):
        for _ in range(70):
            cells = {(0, 0)}
            while len(cells) < n:
                frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS} - cells)
                p = _pick(rng, frontier)
                cells.add(p)
            norm = _canon(cells)
            h, w = _size(norm)
            # Avoid closed holes in a body, retaining a plainly connected shape.
            empty = np.ones((h+2, w+2), dtype=bool)
            for r, c in norm:
                empty[r+1, c+1] = False
            if max(h, w) <= 4 and len(_components(empty)) == 1:
                shapes.add(norm)
    return tuple(sorted(shapes))


def _tabbed(rng, raw, frame, color, sides):
    h, w = _size(raw)
    cells = {p: color for p in raw}
    for side in sides:
        edge = [p for p in raw if (p[0] == 0, p[1] == w-1,
                                   p[0] == h-1, p[1] == 0)[side]]
        count = int(rng.integers(1, min(2, len(edge))+1))
        dr, dc = DIRS[side]
        for i in rng.choice(len(edge), count, replace=False):
            r, c = edge[int(i)]
            cells[r+dr, c+dc] = frame[side]
    return cells


def _make_ballot(rng, raw, frames, color, membership):
    """Choose clear supported tabs with exactly the desired electorate mask."""
    base = frames[membership.index(True)]
    common = [d for d in range(4) if all(f[d] == base[d] for f, yes in zip(frames, membership) if yes)]
    for _ in range(80):
        max_sides = len(common)
        count = int(rng.integers(1, max_sides+1))
        sides = sorted(map(int, rng.choice(common, count, replace=False)))
        cells = _tabbed(rng, raw, base, color, sides)
        features = _ballot_features(cells, set(base))
        if tuple(_eligible(features, raw, frame) for frame in frames) == membership:
            return cells
    # All four identities distinguish two non-dihedrally-equivalent frames.
    if sum(membership) == 1:
        cells = _tabbed(rng, raw, base, color, range(4))
    else:
        cells = _tabbed(rng, raw, base, color, common[:1])
    features = _ballot_features(cells, set(base))
    assert tuple(_eligible(features, raw, frame) for frame in frames) == membership
    return cells


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _positions(available, points):
    h, w = _size(points)
    nr, nc = available.shape[0]-h+1, available.shape[1]-w+1
    if min(nr, nc) <= 0:
        return np.empty((0, 2), dtype=int)
    valid = np.ones((nr, nc), dtype=bool)
    for r, c in points:
        valid &= available[r:r+nr, c:c+nc]
    return np.argwhere(valid)


def _room(rng, shapes, transform, frame, carrier, bg, repeat):
    poses = [_shape(s, transform) for s in shapes]
    for _ in range(30):
        h, w = int(rng.integers(4, 8)), int(rng.integers(4, 9))
        available = np.ones((h, w), dtype=bool)
        placements = []
        order = list(range(len(poses)))
        rng.shuffle(order)
        for i in order:
            positions = _positions(available, poses[i])
            if not len(positions):
                break
            y, x = map(int, _pick(rng, positions))
            points = tuple((y+r, x+c) for r, c in poses[i])
            placements.append((i, points))
            _block(available, points)
        if len(placements) != len(shapes):
            continue
        if repeat:
            i = int(rng.integers(len(poses)))
            positions = _positions(available, poses[i])
            if len(positions):
                y, x = map(int, _pick(rng, positions))
                placements.append((i, tuple((y+r, x+c) for r, c in poses[i])))
        rs, cs = zip(*(p for _, cells in placements for p in cells))
        # Unequal margins retain irregular placements without wasting most
        # of the 30x30 canvas on unused carrier.
        top, left = min(rs), min(cs)
        inner_h = max(rs)-top+1+int(rng.integers(2))
        inner_w = max(cs)-left+1+int(rng.integers(2))
        room = np.full((inner_h+2, inner_w+2), carrier, dtype=np.int8)
        room[0], room[-1] = frame[0], frame[2]
        room[1:-1, 0], room[1:-1, -1] = frame[3], frame[1]
        targets = []
        for i, cells in placements:
            shifted = tuple((r-top+1, c-left+1) for r, c in cells)
            for p in shifted:
                room[p] = bg
            targets.append((i, shifted, poses[i], frame))
        return room, targets
    return None


def _frame_transform(frame, t):
    result = [None]*4
    for d, vector in enumerate(DIRS):
        result[DIRS.index(_point(vector, t))] = frame[d]
    return tuple(result)


def _arrange(rng, rooms, carrier):
    n = len(rooms)
    layout = int(rng.integers(4))
    gap = int(rng.integers(1, 3))
    origins = [(0, 0)]
    if n > 1:
        if layout == 0:
            for i in range(1, n):
                origins.append((int(rng.integers(3)), origins[-1][1]+rooms[i-1].shape[1]+gap))
        elif layout == 1:
            for i in range(1, n):
                origins.append((origins[-1][0]+rooms[i-1].shape[0]+gap, int(rng.integers(3))))
        else:
            origins.append((int(rng.integers(3)), rooms[0].shape[1]+gap))
            if n == 3:
                origins.append((max(rooms[0].shape[0], origins[1][0]+rooms[1].shape[0])+gap,
                                0 if layout == 2 else origins[1][1]))
    margin = int(rng.integers(1, 3))
    h = max(y+room.shape[0] for (y, x), room in zip(origins, rooms))+2*margin
    w = max(x+room.shape[1] for (y, x), room in zip(origins, rooms))+2*margin
    if max(h, w) > 27 or h*w > 420:
        return None
    board = np.full((h, w), carrier, dtype=np.int8)
    for (y, x), room in zip(origins, rooms):
        board[y+margin:y+margin+room.shape[0], x+margin:x+margin+room.shape[1]] = room
    return board, [(y+margin, x+margin) for y, x in origins]


def _construct(rng, shapes, room_count, teaching=None, previous_roles=None):
    palette = list(map(int, rng.permutation(10)))
    if previous_roles is not None:
        bg = _pick(rng, [c for c in palette if c != previous_roles[0]])
        carrier = _pick(rng, [c for c in palette if c not in (bg, previous_roles[1])])
        palette = [bg, carrier]+[c for c in palette if c not in (bg, carrier)]
    bg, carrier = palette[:2]
    frame0 = tuple(palette[2:6])
    frame1 = list(frame0)
    d = int(rng.integers(4))
    frame1[d], frame1[(d+1) % 4] = frame1[(d+1) % 4], frame1[d]
    contexts = [frame0] if room_count == 1 else [frame0, tuple(frame1)]
    raw_shapes = [_pick(rng, _variants(s)) for s in shapes]
    color_a, color_b, color_c, color_d = palette[6:10]
    specifications = []
    for i, raw in enumerate(raw_shapes):
        if room_count == 1:
            counts = (3, 2, 2) if i == 0 else _pick(rng, ((1,), (2, 1), (2, 1, 1)))
            for color, count in zip(palette[6:], counts):
                for _ in range(count):
                    specifications.append((i, color, (True,), False))
        elif i == 0 or (teaching is None and rng.random() < .25):
            # Shared voters participate in both elections. Teaching 0 has
            # 2/1/1 and 3/1; teaching 1 adds repeated losers: 3/2/2 and 3/2.
            if teaching == 1:
                win0, win1, shared_count, loser_count = 3, 3, 2, 2
            elif teaching == 0:
                win0, win1, shared_count, loser_count = 2, 3, 1, 1
            else:
                win0, win1 = int(rng.integers(2, 4)), int(rng.integers(2, 4))
                shared_count, loser_count = 1, int(rng.random() < .5)
            ca, cb = (color_a, color_b) if i == 0 else (color_b, color_c)
            shared_color = color_c if i == 0 else color_a
            specifications.extend((i, ca, (True, False), i == 0) for _ in range(win0))
            specifications.extend((i, cb, (False, True), False) for _ in range(win1))
            specifications.extend((i, shared_color, (True, True), False) for _ in range(shared_count))
            specifications.extend((i, color_d, (True, False), False) for _ in range(loser_count))
        else:
            for _ in range(2 if teaching is not None else int(rng.integers(1, 4))):
                specifications.append((i, color_c, (True, True), False))
    voters = []
    for i, color, membership, witness in specifications:
        cells = _make_ballot(rng, raw_shapes[i], contexts, color, membership)
        voters.append((cells, witness))
    for _ in range(90):
        rooms, target_groups, context_ids, transforms = [], [], [], []
        for j in range(room_count):
            context = j if j < min(2, room_count) else int(rng.integers(2))
            t = int(rng.integers(8))
            frame = _frame_transform(contexts[context], t)
            made = _room(rng, raw_shapes, t, frame, carrier, bg,
                         repeat=teaching is None and rng.random() < .65)
            if made is None:
                break
            room, targets = made
            rooms.append(room)
            target_groups.append(targets)
            context_ids.append(context)
            transforms.append(t)
        if len(rooms) != room_count:
            continue
        arrangement = _arrange(rng, rooms, carrier)
        if arrangement is None:
            continue
        board, origins = arrangement
        # Ground truth is constructed from planned ballots and their visible
        # membership, separately from parsing the completed input.
        output = board.copy()
        flat_targets = []
        for j, (origin, targets) in enumerate(zip(origins, target_groups)):
            for i, points, raw, frame in targets:
                counts = Counter(color for ii, color, member, witness in specifications
                                 if ii == i and member[context_ids[j]])
                win = max(counts, key=counts.get)
                assert list(counts.values()).count(counts[win]) == 1
                absolute = [(r+origin[0], c+origin[1]) for r, c in points]
                for p in absolute:
                    output[p] = win
                flat_targets.append((i, raw, frame, context_ids[j]))
        bh, bw = board.shape
        load = bh*bw + sum((_size(_norm(cells))[0]+1)*(_size(_norm(cells))[1]+1)
                           for cells, _ in voters)
        side = int(np.clip(np.sqrt(load)*1.15, 22, 30))
        height = int(rng.integers(max(bh, side-1), 31))
        width = int(rng.integers(max(bw, side-1), 31))
        if rng.random() < .5:
            height = width = 30
        y = _pick(rng, (0, height-bh, int(rng.integers(height-bh+1))))
        x = _pick(rng, (0, width-bw, int(rng.integers(width-bw+1))))
        inp = np.full((height, width), bg, dtype=np.int8)
        inp[y:y+bh, x:x+bw] = board
        available = np.ones(inp.shape, dtype=bool)
        available[max(0, y-1):min(height, y+bh+1), max(0, x-1):min(width, x+bw+1)] = False
        order = list(map(int, rng.permutation(len(voters))))
        order.sort(key=lambda k: len(voters[k][0]), reverse=True)
        success = True
        for index in order:
            cells, witness = voters[index]
            pose_ids = list(map(int, rng.permutation(8)))
            if witness and teaching is not None:
                # Certify an answer-relevant reflection in demonstration 0
                # and a pure rotation in demonstration 1, using current pixels.
                target = next(v for v in flat_targets if v[0] == 0 and v[3] == 0)
                eligible_poses = []
                for t in pose_ids:
                    f = _ballot_features(_map_transform(cells, t), frame0)
                    rotation = _eligible(f, target[1], target[2], "rotation")
                    identity = _eligible(f, target[1], target[2], "identity")
                    if (not rotation if teaching == 0 else rotation and not identity):
                        eligible_poses.append(t)
                pose_ids = eligible_poses
            placed = None
            for t in pose_ids:
                transformed = _map_transform(cells, t)
                positions = _positions(available, tuple(transformed))
                if not len(positions):
                    continue
                r0, c0 = map(int, _pick(rng, positions))
                placed = {(r+r0, c+c0): v for (r, c), v in transformed.items()}
                break
            if placed is None:
                success = False
                break
            for p, value in placed.items():
                inp[p] = value
            _block(available, placed)
        if not success:
            continue
        try:
            info = _parse(inp)
            prediction, records = _predict(info)
            if not np.array_equal(prediction, output):
                continue
            if teaching is not None and not all(_alternative_fails(info, output, m)
                    for m in ("area", "no_compass", "identity")):
                continue
            if teaching == 0 and not _alternative_fails(info, output, "rotation"):
                continue
        except ValueError:
            continue
        return {"input": inp, "output": output}
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" outside legal range")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pool = _pool(rng)
    eligible = [s for s in pool if 4 <= len(s) <= 6 and max(_size(s)) <= 3
                and _shape(s, 4) not in {_shape(s, t) for t in range(4)}
                and any(t != s and len(t) == len(s) and sorted(_size(t)) == sorted(_size(s)) for t in pool)]
    anchor = _pick(rng, eligible)
    companion = _pick(rng, [s for s in pool if s != anchor and len(s) == len(anchor)
                            and sorted(_size(s)) == sorted(_size(anchor))])
    shift = _pick(rng, ("rooms", "novel_geometry"))
    train_pool = [s for s in pool if len(s) <= 6]
    novel_pool = [s for s in pool if len(s) >= 7]
    puzzle, seen, previous_roles = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            teaching = i if split == "train" and i < 2 else None
            if teaching is not None:
                shapes, room_count = [anchor, companion], 2
            else:
                k = int(rng.choice([1, 2], p=[.3, .7]))
                available = train_pool if split == "train" or shift == "rooms" else novel_pool
                shapes = [_pick(rng, available)]
                if k == 2:
                    shapes.append(_pick(rng, [s for s in available if s != shapes[0]]))
                room_count = (3 if split == "test" and shift == "rooms" else
                              int(rng.integers(1, 3 if shift == "rooms" else 4)))
            for attempt in range(12):
                pair = _construct(rng, shapes, room_count, teaching,
                                  previous_roles if teaching == 1 else None)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction failed")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train" and i == 0:
                data = _parse(pair["input"])
                previous_roles = data["background"], data["carrier"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _parse(inp)
            predicted, records = _predict(info)
            assert np.array_equal(out, predicted), "wrong local election or crop"
            preserved = info["box"] != info["background"]
            assert out.shape == info["box"].shape
            assert np.array_equal(out[preserved], info["box"][preserved])
            assert all(record["winner"] in {v["color"] for v in info["voters"]}
                       for record in records)
            # One physical ballot may vote in several elections, but is counted
            # once in each even if the body or tabs have D4 symmetries.
            assert all(len(rec["eligible"]) == len(set(rec["eligible"])) for rec in records)
            assert all(rec["counts"].total() == len(rec["eligible"]) for rec in records)
            if split == "train":
                train_info.append((info, out, records))
    assert len({info["background"] for info, _, _ in train_info}) >= 2
    assert len({info["carrier"] for info, _, _ in train_info}) >= 2
    for mode in ("area", "no_compass", "rotation", "identity"):
        assert any(_alternative_fails(info, out, mode) for info, out, _ in train_info), mode
    assert any(len(rec["counts"]) >= 3 and 2*max(rec["counts"].values()) <= rec["counts"].total()
               for info, out, records in train_info for rec in records)
    assert any(len(rec["counts"]) >= 3 and sorted(rec["counts"].values(), reverse=True)[1] >= 2
               for info, out, records in train_info for rec in records)
    assert any(any(a["canonical"] == b["canonical"] and ra["winner"] != rb["winner"]
                   for j, (a, ra) in enumerate(zip(info["cavities"], records))
                   for b, rb in list(zip(info["cavities"], records))[j+1:])
               for info, out, records in train_info)
    assert any(any(sum(i in rec["eligible"] for rec in records) > 1
                   for i in range(len(info["voters"]))) for info, out, records in train_info)


def self_test():
    """Bounded 42-seed semantic/diversity audit; no external files required."""
    # Symmetry is not extra voting power: two adjacent-tab ballots each have
    # one valid D4 identifier; a single-tab ballot has two. The physical vote
    # is 2--1, while counting transformations would incorrectly produce 2--2.
    cross = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
    frame = (1, 2, 3, 4)
    two_tabs = {**{p: 8 for p in cross}, (-1, 1): 1, (1, 3): 2}
    one_tab = {**{p: 9 for p in cross}, (-1, 1): 1}
    voters = [_ballot_features(v, frame) for v in (two_tabs, two_tabs, one_tab)]
    box = np.full((3, 3), 7, dtype=np.int8)
    symmetric = dict(box=box, voters=voters,
                     cavities=[dict(raw=cross, frame=frame, cells=set(cross))])
    _, records = _predict(symmetric)
    assert records[0]["counts"] == Counter({8: 2, 9: 1})
    weighted = Counter()
    for voter in voters:
        weighted[voter["color"]] += sum(body == cross and all(frame[d] == color for d, color in labels)
                                        for body, labels in voter["transforms"])
    assert weighted == Counter({8: 2, 9: 2})
    puzzles, features, all_shapes, all_inputs = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                answer, records = _predict(info)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                assert np.array_equal(answer, pair["output"])
                all_inputs.add(_key(pair["input"]))
                all_shapes.update(c["canonical"] for c in info["cavities"])
                features.append((pair["input"].shape, pair["output"].shape,
                                 info["background"], len(info["rooms"]), len(info["voters"]),
                                 len(info["cavities"]),
                                 tuple(sorted(len(v["transforms"][0][1]) for v in info["voters"]))))
        train_data = [_parse(p["input"]) for p in puzzle["train"]]
        test_data = [_parse(p["input"]) for p in puzzle["test"]]
        seen_shapes = {c["canonical"] for d in train_data for c in d["cavities"]}
        assert (max(len(d["rooms"]) for d in test_data) > max(len(d["rooms"]) for d in train_data)
                or any(c["canonical"] not in seen_shapes for d in test_data for c in d["cavities"]))
    assert len(all_inputs) == 42*5
    assert len({f[0] for f in features}) >= 15
    assert len({f[1] for f in features}) >= 45
    assert {f[2] for f in features} == set(range(10))
    assert {f[3] for f in features} == {1, 2, 3}
    assert len({f[4] for f in features}) >= 6
    assert len({f[5] for f in features}) >= 5
    assert len({f[6] for f in features}) >= 25
    assert len(all_shapes) >= 40
    def puzzle_key(p):
        return tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s])
                     for s in ("train", "test"))
    assert puzzle_key(puzzles[0]) == puzzle_key(generate_puzzle(0))
    assert puzzle_key(puzzles[0]) != puzzle_key(puzzles[1])
    for nt, ne in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(800+nt*10+ne, nt, ne)
        assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted carrier accepted")


if __name__ == "__main__":
    self_test()
