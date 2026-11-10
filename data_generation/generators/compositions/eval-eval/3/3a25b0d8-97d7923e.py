"""Hidden-family room tournaments: correspondence, unary rank, and cavity area.

No parent code or data is imported. Generation paints from construction labels;
solve reconstructs every room and its correspondence from the visible input.
The module performs no file I/O, logging, or execution-trace retention.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must lie in 0..9")
    return a


def _parts(mask):
    """Four-connected components, returned in topmost/leftmost discovery order."""
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    points.append(q)
        result.append(np.asarray(points, dtype=int))
    return result


def _box(points):
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return tuple(map(int, (*lo, *hi)))


def _rooms(walls):
    padded = np.pad(~walls, 1, constant_values=True)
    return [p - 1 for p in _parts(padded)
            if bool(np.all(p > 0) and np.all(p < np.array(padded.shape) - 1))]


def _relations(rooms):
    """Qualitative above/below and left/right/overlap, independent of area."""
    boxes = [_box(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _match(left, right):
    """Require a unique orientation-preserving spatial-graph isomorphism."""
    _require(len(left) == len(right), "different compartment counts")
    a, b = _relations(left), _relations(right)
    n = len(a)
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    domains = [[j for j in range(n) if signatures_a[i] == signatures_b[j]] for i in range(n)]
    found, visits = [], 0

    def search(mapping, used):
        nonlocal visits
        visits += 1
        _require(visits <= 12000, "insufficient spatial landmarks")
        if len(found) >= 2:
            return
        if len(mapping) == n:
            found.append(dict(mapping))
            return
        options = {i: [j for j in domains[i] if j not in used and
                       all(a[i][k] == b[j][v] for k, v in mapping.items())]
                   for i in range(n) if i not in mapping}
        i = min(options, key=lambda k: (len(options[k]), k))
        for j in options[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]
    search({}, set())
    _require(len(found) == 1, "room correspondence is absent or ambiguous")
    return found[0]


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "one empty background border is required")
    candidates = []
    for wall in sorted(set(map(int, a.flat)) - {bg}):
        components = _parts(a == wall)
        if len(components) != 2:
            continue
        figures = []
        for p in components:
            r, c, R, C = _box(p)
            local_walls = np.zeros((R - r, C - c), dtype=bool)
            local_walls[p[:, 0] - r, p[:, 1] - c] = True
            rooms = _rooms(local_walls)
            if not 4 <= len(rooms) <= 12:
                break
            crop = a[r:R, c:C].copy()
            inside = np.zeros(crop.shape, dtype=bool)
            for room in rooms:
                inside[tuple(room.T)] = True
            if np.any((crop != bg) & ~inside & ~local_walls):
                break
            figures.append((crop, rooms, (r, c, R, C), local_walls))
        if len(figures) != 2:
            continue
        try:
            blank, colored = [], []
            for figure in figures:
                values = [np.unique(figure[0][tuple(p.T)]) for p in figure[1]]
                if all(len(v) == 1 and int(v[0]) == bg for v in values):
                    blank.append(figure)
                elif all(len(v) == 1 and int(v[0]) not in (bg, wall) for v in values):
                    colored.append(figure)
            _require(len(blank) == len(colored) == 1, "donor and recipient roles must be visible")
            donor, target = colored[0], blank[0]
            correspondence = _match(donor[1], target[1])
            colors = [int(donor[0][tuple(p.T)][0]) for p in donor[1]]
            outside = np.ones(a.shape, dtype=bool)
            for figure in figures:
                r, c, R, C = figure[2]
                outside[r:R, c:C] = False
            key_mask = outside & (a != bg)
            _require(set(map(int, a[key_mask])) == set(colors), "keys must exactly cover donor identities")
            ranks = {}
            for color in sorted(set(colors)):
                p = np.argwhere(key_mask & (a == color))
                r, c, R, C = _box(p)
                _require(len(_parts(key_mask & (a == color))) == 1, "split unary key")
                _require((R - r == 1 or C - c == 1) and len(p) == (R - r) * (C - c),
                         "unary key must be a straight solid run")
                rank = len(p)
                members = [i for i, v in enumerate(colors) if v == color]
                areas = [len(target[1][correspondence[i]]) for i in members]
                _require(2 <= len(members) <= 6 and 1 <= rank <= min(3, len(members)),
                         "invalid competitor population or ordinal")
                _require(len(set(areas)) == len(areas), "tied recipient areas in one family")
                ranks[color] = rank
            _require(1 <= len(ranks) <= 3, "invalid identity count")
            candidates.append(dict(background=bg, wall=wall, donor=donor, target=target,
                                   mapping=correspondence, colors=colors, ranks=ranks))
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input does not uniquely encode a room tournament")
    return candidates[0]


def _winners(info, measure="area", ranks=None):
    donor, target, mapping = info["donor"], info["target"], info["mapping"]
    winners = {}
    for color, rank in (info["ranks"] if ranks is None else ranks).items():
        members = [i for i, v in enumerate(info["colors"]) if v == color]
        def size(i):
            room = target[1][mapping[i]]
            r, c, R, C = _box(room)
            if measure == "donor_area":
                return len(donor[1][i])
            if measure == "box_area":
                return (R - r) * (C - c)
            if measure == "extent":
                return max(R - r, C - c)
            if measure == "width":
                return C - c
            if measure == "height":
                return R - r
            return len(room)
        members.sort(key=lambda i: (-size(i), i))
        winners[color] = members[rank - 1]
    return winners


def _answer(info, winners=None, mapping=None, all_rooms=False):
    result = info["target"][0].copy()
    mapping = info["mapping"] if mapping is None else mapping
    winners = _winners(info) if winners is None else winners
    for i, color in enumerate(info["colors"]):
        if all_rooms or winners[color] == i:
            result[tuple(info["target"][1][mapping[i]].T)] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the tight crop of the recipient with one winning room per key."""
    return _answer(_decode(input_grid))


def _abstract(rng, count, extra=False):
    shapes = [(r, c) for r in range(2, 5) for c in range(2, 5)
              if min(r, c) <= 3 and count + (4 if extra else 0) <= r * c <= min(15, count + (8 if extra else 5))]
    rows, cols = shapes[int(rng.integers(len(shapes)))]
    labels = np.arange(rows * cols, dtype=int).reshape(rows, cols)
    # Missing corners change the exterior and create asymmetric side pockets.
    for r, c in [((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1))[int(i)]
                 for i in rng.permutation(4)]:
        if np.count_nonzero(labels >= 0) > count and rng.random() < 0.28:
            labels[r, c] = -1
    if len(_parts(labels >= 0)) != 1:
        return None
    while len(set(labels.flat) - {-1}) > count:
        edges = [(int(labels[r, c]), int(labels[r + dr, c + dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < rows and c + dc < cols and labels[r, c] >= 0
                 and labels[r + dr, c + dc] >= 0 and labels[r, c] != labels[r + dr, c + dc]]
        if not edges:
            return None
        a, b = edges[int(rng.integers(len(edges)))]
        labels[labels == b] = a
    for new, old in enumerate(sorted(set(labels.flat) - {-1})):
        labels[labels == old] = new
    return labels


def _geometry(rng, plan, widths=None, heights=None, trim=True):
    if widths is None:
        widths = rng.integers(1, 4, size=plan.shape[1])
    if heights is None:
        heights = rng.integers(2, 4, size=plan.shape[0])
    xs = np.r_[0, np.cumsum(np.asarray(widths) + 1)]
    ys = np.r_[0, np.cumsum(np.asarray(heights) + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1]):
            v = plan[r, c]
            if v < 0:
                continue
            labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = v
            if r and plan[r - 1, c] == v:
                labels[ys[r], xs[c] + 1:xs[c + 1]] = v
            if c and plan[r, c - 1] == v:
                labels[ys[r] + 1:ys[r + 1], xs[c]] = v
            if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == v):
                labels[ys[r], xs[c]] = v
    original = _relations([np.argwhere(labels == i) for i in range(int(plan.max()) + 1)])
    if trim:
        # Shift room starts independently, and notch corners. A visible spatial
        # graph must survive; discovery/scan order need not survive.
        for i in rng.permutation(int(plan.max()) + 1):
            i = int(i)
            p = np.argwhere(labels == i)
            r, c, R, C = _box(p)
            proposal = labels.copy()
            kind = int(rng.integers(4))
            if kind == 0:
                proposal[r, proposal[r] == i] = -2
            elif kind == 1:
                proposal[R - 1, proposal[R - 1] == i] = -2
            else:
                corners = [(r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1)]
                rng.shuffle(corners)
                for q in corners[:int(rng.integers(1, 4))]:
                    if proposal[q] == i:
                        proposal[q] = -2
            if len(_parts(proposal == i)) != 1:
                continue
            rooms = [np.argwhere(proposal == j) for j in range(int(plan.max()) + 1)]
            if _relations(rooms) == original:
                labels = proposal
    occupied = labels >= 0
    p = np.pad(occupied, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= p[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    r, c, R, C = _box(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _paint(labels, bg, wall, colors=None):
    a = np.full(labels.shape, bg, dtype=np.int8)
    a[labels == -2] = wall
    if colors is not None:
        for i, color in enumerate(colors):
            a[labels == i] = color
    return a


def _embed(rng, donor, target, bg, colors, ranks, plan_shape):
    # Arrange along the coarse plan's shorter dimension, reserving a small
    # orthogonal key band. Rotate the whole scene back afterwards.
    transpose = plan_shape[0] < plan_shape[1]
    if transpose:
        donor, target = donor.T, target.T
    gap = int(rng.integers(1, 3))
    if donor.shape[1] + target.shape[1] + gap + 2 > 30:
        gap = 1
    width = donor.shape[1] + target.shape[1] + gap
    offset_budget = min(3, 24 - max(donor.shape[0], target.shape[0]))
    offsets = [int(rng.integers(offset_budget + 1)) for _ in range(2)]
    objects = [donor, target]
    rng.shuffle(objects)
    height = max(offsets[i] + obj.shape[0] for i, obj in enumerate(objects))
    key_height = max(ranks.values())
    core = np.full((height + key_height + 2, width), bg, dtype=np.int8)
    x = 0
    for i, obj in enumerate(objects):
        y = key_height + 2 + offsets[i]
        core[y:y + obj.shape[0], x:x + obj.shape[1]] = obj
        x += obj.shape[1] + gap
    order = [int(c) for c in rng.permutation(sorted(set(colors)))]
    key_width = 2 * len(order) - 1
    left = int(rng.integers(width - key_width + 1))
    for i, color in enumerate(order):
        core[:ranks[color], left + 2 * i] = color
    flip_key = bool(rng.integers(2))
    if flip_key:
        core = np.flipud(core)
    # Uniform empty border makes background role explicit, even at dense sizes.
    extra_h = int(rng.integers(min(3, 28 - core.shape[0]) + 1))
    extra_w = int(rng.integers(min(2, 28 - core.shape[1]) + 1))
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    core = np.pad(core, ((1 + top, 1 + extra_h - top), (1 + left, 1 + extra_w - left)),
                  constant_values=bg)
    if transpose:
        core = core.T
    rotation = int(rng.integers(4))
    reflected = bool(rng.integers(2))
    def transform(a):
        return np.ascontiguousarray(np.rot90(np.fliplr(a) if reflected else a, rotation))
    def transform_answer(a):
        if flip_key:
            a = np.fliplr(a) if transpose else np.flipud(a)
        return transform(a)
    return transform(core), transform_answer


def _rejected_shortcuts(info):
    """Reject a metric even allowing every way to resolve its tied values."""
    winners = _winners(info)
    rejected = set()
    for metric in ("donor_area", "box_area", "extent", "width", "height"):
        for color, chosen in winners.items():
            members = [i for i, v in enumerate(info["colors"]) if v == color]
            values = {}
            for i in members:
                room = info["donor"][1][i] if metric == "donor_area" else info["target"][1][info["mapping"][i]]
                r, c, R, C = _box(room)
                values[i] = (len(room) if metric == "donor_area" else
                             (R - r) * (C - c) if metric == "box_area" else
                             max(R - r, C - c) if metric == "extent" else
                             C - c if metric == "width" else R - r)
            better = sum(v > values[chosen] for v in values.values())
            equal = sum(v == values[chosen] for v in values.values())
            if not better < info["ranks"][color] <= better + equal:
                rejected.add(metric)
    scan_info = dict(info, mapping={i: i for i in range(len(info["colors"]))})
    if not np.array_equal(_answer(scan_info), _answer(info)):
        rejected.add("scan")
    return rejected


def _pair(rng, counts, rank_values, palette, required=()):
    count = sum(counts)
    bg, wall = palette[:2]
    identities = palette[2:2 + len(counts)]
    ranks = dict(zip(identities, rank_values))
    for attempt in range(192):
        plan = _abstract(rng, count, extra="box_area" in required)
        if plan is None:
            continue
        d, t = _geometry(rng, plan), _geometry(rng, plan)
        if len(_parts(d == -2)) != 1 or len(_parts(t == -2)) != 1:
            continue
        drooms, trooms = _rooms(d == -2), _rooms(t == -2)
        if len(drooms) != count or len(trooms) != count:
            continue
        try:
            _match(drooms, trooms)
        except AssertionError:
            continue
        # Local label identity is used ONLY for construction; the decoder does
        # not receive it. Distinct recipient areas are required within families.
        colors = [color for color, n in zip(identities, counts) for _ in range(n)]
        for _ in range(18):
            rng.shuffle(colors)
            if all(len({int(np.count_nonzero(t == i)) for i, c in enumerate(colors) if c == color}) == n
                   for color, n in zip(identities, counts)):
                break
        else:
            continue
        donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
        expected = target.copy()
        for color, rank in ranks.items():
            members = sorted((i for i, c in enumerate(colors) if c == color),
                             key=lambda i: -int(np.count_nonzero(t == i)))
            expected[t == members[rank - 1]] = color
        inp, transform = _embed(rng, donor, target, bg, colors, ranks, plan.shape)
        # _embed transposes individual figures when needed, then transposes the
        # whole core. These cancel for each figure's local orientation.
        out = transform(expected)
        try:
            info = _decode(inp)
            _require(np.array_equal(_answer(info), out), "construction/decode mismatch")
            if required:
                rejected = _rejected_shortcuts(info)
                _require(set(required) <= rejected,
                         "teaching contrast does not rule out geometric shortcuts")
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    return _fallback_pair(rng, counts, rank_values, palette, required)


def _fallback_pair(rng, counts, rank_values, palette, required):
    """Guaranteed fitting constructions after bounded random attempts.

    The two teaching geometries are labeled floorplans, not stored ARC pairs or
    stored outputs. All colors, keys, placements and winners are constructed.
    Ordinary examples use a procedural grid with distinct areas per family.
    """
    if required:
        if "box_area" in required:
            donor_rows = ("##############", "#aaa#bbbb#ccc#", "#####bbbb#ccc#", ".#########ccc#",
                          "##dddd##e#ccc#", "#dddddd#e#ccc#", "#####dd#######", "....#dd#######",
                          "....#dd#fffff#", "....##########")
            target_rows = ("...#####....", "####bbb#####", "#a#bbbb#ccc#", "#a#bbbb#ccc#",
                           "########ccc#", "##dd####ccc#", "#dddd#e#ccc#", "#dddd#e#ccc#",
                           "###dd#######", "..#dd#ffff##", "..#dd##ffff#", "..##########")
            roles = [0, 0, 1, 1, 1, 0]
        else:
            donor_rows = ("####............", "#aa#############", "#aaa#bbbbbb#ccc#", "#aaa#bbbbbb#ccc#",
                          "################", "######ee#ff#....", "#ddd#ee##ff#....", "############....")
            target_rows = ("############", "#aaa#bb##cc#", "#####bbb#cc#", "############",
                           "#dd####f#...", "#dd##e#f#...", "#########...")
            roles = [1, 1, 1, 0, 0, 0]
        def labels(rows):
            return np.array([[-2 if c == "#" else -1 if c == "." else ord(c) - 97
                              for c in row] for row in rows], dtype=int)
        d, t = labels(donor_rows), labels(target_rows)
        colors = [palette[2 + role] for role in roles]
        ranks = dict(zip(palette[2:4], rank_values))
    else:
        n = sum(counts)
        rows, cols = (n + 2) // 3, 3
        d = np.full((4 * rows + 1, 4 * cols + 1), -2, dtype=int)
        t = d.copy()
        for i in range(n):
            y, x = 4 * (i // cols) + 1, 4 * (i % cols) + 1
            d[y:y + 3, x:x + 3] = i
            # Row-major prefixes of a 3x3 cell are connected; every family
            # receives <=6 consecutive, hence distinct, recipient areas.
            for k in range(i % 6 + 1):
                t[y + k // 3, x + k % 3] = i
        colors = [color for color, n in zip(palette[2:], counts) for _ in range(n)]
        ranks = dict(zip(palette[2:2 + len(counts)], rank_values))
    bg, wall = palette[:2]
    donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
    expected = target.copy()
    for color, rank in ranks.items():
        members = sorted((i for i, c in enumerate(colors) if c == color),
                         key=lambda i: -int(np.count_nonzero(t == i)))
        expected[t == members[rank - 1]] = color
    plan_shape = (2, 3) if d.shape[0] + t.shape[0] < d.shape[1] + t.shape[1] else (3, 2)
    inp, transform = _embed(rng, donor, target, bg, colors, ranks, plan_shape)
    out = transform(expected)
    info = _decode(inp)
    _require(np.array_equal(out, _answer(info)), "fallback construction/decode mismatch")
    # Scan order is an absolute-canvas shortcut. Choose a geometric view that
    # exposes its failure; size-metric contrasts are dihedrally invariant.
    for reflection in (False, True):
        for rotation in range(4):
            a = np.ascontiguousarray(np.rot90(np.fliplr(inp) if reflection else inp, rotation))
            b = np.ascontiguousarray(np.rot90(np.fliplr(out) if reflection else out, rotation))
            if set(required) <= _rejected_shortcuts(_decode(a)):
                return {"input": a, "output": b}
    raise AssertionError("fallback teaching contrast failed")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = ("rank", "population", "families")[int(rng.integers(3))]
    base_palette = [int(c) for c in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index < 2:
                counts = ([3, 3] if mode == "population" else [3, 4] if mode == "rank" else
                          [3, int(rng.integers(3, 5))])
                ranks = [1, 2] if index == 0 else [2, 1]
                # The first two teaching pairs visibly reuse the key identities.
                palette = base_palette.copy()
                if index == 1:
                    palette[0], palette[1] = palette[1], palette[0]
            else:
                f = 3 if split == "test" and mode == "families" else int(rng.integers(1, 3))
                if split == "train" and mode == "population":
                    f = 2
                upper = 5 if split == "test" and mode != "population" else 4
                counts = [int(rng.integers(2, upper)) for _ in range(f)]
                if split == "test" and mode == "population":
                    counts[int(rng.integers(f))] = int(rng.integers(4, 6))
                if sum(counts) < 4:
                    counts[0] = 4
                ranks = [int(rng.integers(1, min(n, 2) + 1)) for n in counts]
                if split == "test" and mode == "rank":
                    k = int(rng.integers(f))
                    counts[k] = max(counts[k], 3)
                    ranks[k] = 3
                palette = [int(c) for c in rng.permutation(10)]
            for retry in range(16):
                required = (("donor_area", "scan") if split == "train" and index == 0 else
                            ("extent", "box_area") if split == "train" and index == 1 else ())
                pair = _pair(rng, counts, ranks, palette, required=required)
                if _signature(pair["input"]) not in seen:
                    break
            else:
                # At least 42 role renamings preserve every payload identity,
                # so one is unique among at most 13 earlier inputs.
                info = _decode(pair["input"])
                available = [c for c in range(10) if c not in info["colors"]]
                old = [info["background"], info["wall"]] + [c for c in available
                       if c not in (info["background"], info["wall"])]
                for bg in available:
                    for wall in available:
                        if wall == bg:
                            continue
                        new = [bg, wall] + [c for c in available if c not in (bg, wall)]
                        lookup = np.arange(10, dtype=np.int8)
                        lookup[old] = new
                        candidate = {k: lookup[v] for k, v in pair.items()}
                        if _signature(candidate["input"]) not in seen:
                            pair = candidate
                            break
                    else:
                        continue
                    break
                _require(_signature(pair["input"]) not in seen, "unique-role fallback failed")
            seen.add(_signature(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def measure_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    _require(np.array_equal(out, _answer(info)), "incorrect room tournament output")
    target, mapping, winners = info["target"], info["mapping"], _winners(info)
    _require(np.array_equal(out == info["wall"], target[3]), "recipient walls changed")
    filled = []
    for color, chosen in winners.items():
        filled.extend(map(tuple, target[1][mapping[chosen]]))
    changed = np.zeros(out.shape, dtype=bool)
    for p in filled:
        changed[p] = True
    _require(np.array_equal(out != target[0], changed), "wrong changed-cell set")
    _require(np.all(target[0][changed] == info["background"]), "only blank cavities may be filled")
    _require(np.array_equal(out[~changed], target[0][~changed]), "unselected rooms or exterior changed")
    colors = info["colors"]
    areas = [len(p) for p in target[1]]
    return {"rooms": len(areas), "families": len(info["ranks"]),
            "populations": [colors.count(c) for c in sorted(info["ranks"])],
            "ranks": [info["ranks"][c] for c in sorted(info["ranks"])],
            "background": info["background"], "wall": info["wall"],
            "input_shape": tuple(inp.shape), "output_shape": tuple(out.shape),
            "target_areas": sorted(areas),
            "nonrectangular": sum(len(p) != (_box(p)[2] - _box(p)[0]) * (_box(p)[3] - _box(p)[1])
                                  for p in target[1]),
            "scan_changed": any(i != j for i, j in mapping.items()),
            "rejected_shortcuts": sorted(_rejected_shortcuts(info)),
            "graph": tuple(tuple(row) for row in _relations(target[1]))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _signature(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            metrics[split].append(measure_pair(pair))
    observed = {rank for m in metrics["train"] for rank in m["ranks"]}
    _require({1, 2} <= observed, "missing ordinal teaching progression")
    _require(any(1 < r < n for m in metrics["train"]
                 for n, r in zip(m["populations"], m["ranks"])), "no non-extreme demonstration")
    rejected = set().union(*(set(m["rejected_shortcuts"]) for m in metrics["train"]))
    _require({"donor_area", "scan", "extent", "box_area"} <= rejected, "training leaves a geometric shortcut")
    histories = {}
    for pair in puzzle["train"]:
        for color, rank in _decode(pair["input"])["ranks"].items():
            histories.setdefault(color, set()).add(rank)
    _require(any({1, 2} <= values for values in histories.values()), "missing repeated-identity rank contrast")
    max_population = max(n for m in metrics["train"] for n in m["populations"])
    max_families = max(m["families"] for m in metrics["train"])
    for m in metrics["test"]:
        _require(all(r in observed or set(range(1, r)) <= observed for r in m["ranks"]),
                 "test ordinal lacks prior progression")
        _require(any(r not in observed for r in m["ranks"]) or
                 max(m["populations"]) > max_population or m["families"] > max_families,
                 "test has no intended ordinal, population or family shift")


def self_test():
    """Streaming 42-seed checks, bounded counterfactuals, and count boundaries."""
    records, shapes, graphs, puzzles, structures = [], set(), set(), set(), set()
    identity_interventions = key_interventions = invalid_swaps = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                m = measure_pair(pair)
                records.append(m)
                shapes.add(m["input_shape"])
                graphs.add(m["graph"])
                digest.update(pair["input"].tobytes())
                a = pair["input"]
                rename = {}
                norm = bytes(rename.setdefault(int(c), len(rename)) for c in a.flat)
                structures.add((a.shape, norm))
        puzzles.add(digest.hexdigest())
        # Removing B: filling every corresponding room always changes the answer.
        info = _decode(p["train"][0]["input"])
        _require(not np.array_equal(_answer(info, all_rooms=True), _answer(info)), "B is redundant")
        # Removing A's identity correspondence: exchange two differently colored
        # winning donor compartments, preserving geometry, keys and populations.
        original = p["train"][0]["input"]
        winners = _winners(info)
        r, c, _, _ = info["donor"][2]
        changed = False
        for ia, ca in enumerate(info["colors"]):
            for ib, cb in enumerate(info["colors"]):
                if ca == cb or (winners[ca] != ia and winners[cb] != ib):
                    continue
                edited = original.copy()
                for i, color in ((ia, cb), (ib, ca)):
                    q = info["donor"][1][i] + (r, c)
                    edited[tuple(q.T)] = color
                try:
                    alternate = solve(edited)
                except AssertionError:
                    invalid_swaps += 1
                    continue
                if not np.array_equal(alternate, _answer(info)):
                    identity_interventions += 1
                    changed = True
                    break
            if changed:
                break
        _require(changed, "no valid identity intervention changes the answer")
        # Remove one end cell from the rank-two key. This is a valid rank-one
        # query with all geometry and all family memberships unchanged.
        edited = original.copy()
        color = next(c for c, rank in info["ranks"].items() if rank == 2)
        outside = np.ones(edited.shape, dtype=bool)
        for fig in (info["donor"], info["target"]):
            y, x, Y, X = fig[2]
            outside[y:Y, x:X] = False
        q = tuple(np.argwhere(outside & (edited == color))[0])
        edited[q] = info["background"]
        _require(not np.array_equal(solve(edited), _answer(info)), "unary key is redundant")
        key_interventions += 1
    _require(len(puzzles) == 42 and len(structures) >= 180, "insufficient structural diversity")
    _require(len(shapes) >= 30 and len(graphs) >= 30, "insufficient geometry/topology diversity")
    _require({1, 2, 3} <= {m["families"] for m in records}, "family-count coverage")
    _require({1, 2, 3} <= {r for m in records for r in m["ranks"]}, "ordinal coverage")
    _require({2, 3, 4, 5} <= {n for m in records for n in m["populations"]}, "population coverage")
    _require(sum(m["nonrectangular"] > 0 for m in records) >= 100, "nonrectangular-room coverage")
    for nt, nv in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(907, nt, nv), generate_puzzle(907, nt, nv)
        validate_generated_puzzle(a)
        _require(all(_signature(x[k]) == _signature(y[k]) for s in a
                     for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(16, 2, 1)
    info = _decode(bad["test"][0]["input"])
    q = tuple(np.argwhere(info["target"][3])[0])
    bad["test"][0]["output"][q] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt wall accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "spatial_graphs": len(graphs),
            "color_normalized_structures": len(structures),
            "nonrectangular_pairs": sum(m["nonrectangular"] > 0 for m in records),
            "scan_reordered_pairs": sum(m["scan_changed"] for m in records),
            "valid_identity_interventions": identity_interventions,
            "key_interventions": key_interventions, "excluded_tied_swaps": invalid_swaps}
