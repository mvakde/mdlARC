"""Assemble a chamber jigsaw, then print its registered open component.

Only NumPy and the standard library are used. The solver does not consult the
generator, a seed, training examples, or a table of sampled layouts.
"""
from __future__ import annotations

import numpy as np


DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
CORNERS = ("TL", "TR", "BL", "BR")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= g) & (g <= 9))), "invalid grid colors")
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = [start]
        for r, c in queue:
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
        result.append(set(queue))
    return result


def _flood(mask, point):
    if not (0 <= point[0] < mask.shape[0] and 0 <= point[1] < mask.shape[1]) or not mask[point]:
        return set()
    visited = {point}
    queue = [point]
    for r, c in queue:
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if (0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1]
                    and mask[p] and p not in visited):
                visited.add(p)
                queue.append(p)
    return visited


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return {(r + dy, c), (r, c + dx), (r + dy, c + dx)}


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background mode is ambiguous")
    bg = int(counts.argmax())
    marker_options = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        cells = set(map(tuple, np.argwhere(g == color).tolist()))
        rr, cc = map(int, np.array(sorted(cells)).min(axis=0))
        if cells <= {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}:
            missing = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)} - cells
            if len(missing) != 1:
                continue
            point = missing.pop()
            if g[point] in (bg, color):
                continue
            corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
                (point[0] - rr, point[1] - cc)]
            components = _components((g != bg) & (g != color))
            tile_cells = [x for x in components if len(x) >= 9 and len({int(g[p]) for p in x}) >= 2]
            anchors = [i for i, x in enumerate(tile_cells) if point in x]
            if len(anchors) == 1:
                marker_options.append((color, point, corner, tile_cells, anchors[0]))
    _require(len(marker_options) == 1, "missing or ambiguous external L")
    marker, point, corner, tile_cells, anchor = marker_options[0]
    _require(3 <= len(tile_cells) <= 6, "expected three to six tiles")
    pieces, union = [], set()
    for cells in tile_cells:
        q = np.array(sorted(cells))
        r, c = q.min(axis=0)
        end_r, end_c = q.max(axis=0) + 1
        m = np.full((end_r - r, end_c - c), -1, dtype=np.int8)
        for y, x in cells:
            m[y - r, x - c] = g[y, x]
        pieces.append((m, (int(r), int(c))))
        union |= cells
    palette_counts = {color: sum(int(g[p]) == color for p in union) for color in range(10)}
    common = sorted((c for c in range(10) if palette_counts[c] > 1),
                    key=lambda c: -palette_counts[c])
    singleton = [c for c in range(10) if palette_counts[c] == 1]
    _require(len(common) == 2 and len(singleton) == 2, "tiles need two base colors and two labels")
    open_color, wall = common
    _require(palette_counts[open_color] > palette_counts[wall] >= 4, "open color must be the tile majority")
    for m, _ in pieces:
        _require(np.count_nonzero(m == open_color) >= 2 and np.count_nonzero(m == wall) >= 2,
                 "each tile needs both base colors")
    ar, ac = pieces[anchor][1]
    ah, aw = pieces[anchor][0].shape
    _require(point == (ar + (ah - 1 if corner[0] == "B" else 0),
                       ac + (aw - 1 if corner[1] == "R" else 0)), "L must mark a tile corner")
    for cells in _components((g != bg) & (g != marker)):
        if not cells & union:
            _require(len(cells) <= 8 and len({int(g[p]) for p in cells}) == 1,
                     "external content must be small monochrome groups")
    _require(not any(int(g[p]) in (open_color, wall) for p in
                     set(map(tuple, np.argwhere((g != bg) & (g != marker)).tolist())) - union),
             "base colors cannot occur outside tiles")
    return dict(bg=bg, marker=marker, point=point, corner=corner, pieces=pieces,
                anchor=anchor, tile_cells=union, open_color=open_color, wall=wall,
                label_colors=singleton)


def _assemble(info, canvas):
    """Complete search of translation-only rectangular tilings with equal seams.

    Color matching is checked against cells already placed, so false edge
    matches remain alternatives. An incomplete search is never accepted.
    """
    pieces = info["pieces"]
    area = sum(np.count_nonzero(m >= 0) for m, _ in pieces)
    point, corner, anchor = info["point"], info["corner"], info["anchor"]
    winner, winner_key, nodes = None, None, 0
    for h in range(1, canvas[0] + 1):
        if area % h:
            continue
        w = int(area // h)
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if w > canvas[1] or oy < 0 or ox < 0 or oy + h > canvas[0] or ox + w > canvas[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        configs = []
        for m, _ in pieces:
            coords = [(int(r), int(c), int(m[r, c])) for r, c in np.argwhere(m >= 0)]
            bit = sum(1 << (r * w + c) for r, c, _ in coords)
            configs.append((m.shape, coords, coords[0][1], bit))
        board = np.full((h, w), -1, dtype=np.int8)
        owners = np.full((h, w), -1, dtype=np.int8)
        positions = [None] * len(pieces)

        def place(i, r, c):
            ph, pw = configs[i][0]
            if r < 0 or c < 0 or r + ph > h or c + pw > w:
                return False
            for y, x, value in configs[i][1]:
                if board[r + y, c + x] >= 0:
                    return False
                for dy, dx in DIRS:
                    yy, xx = r + y + dy, c + x + dx
                    if 0 <= yy < h and 0 <= xx < w and board[yy, xx] >= 0 and board[yy, xx] != value:
                        return False
            for y, x, value in configs[i][1]:
                board[r + y, c + x] = value
                owners[r + y, c + x] = i
            positions[i] = (r, c)
            return True

        ar, ac = pieces[anchor][1][0] - oy, pieces[anchor][1][1] - ox
        if not place(anchor, ar, ac):
            continue
        occupied = configs[anchor][3] << (ar * w + ac)
        full = (1 << int(area)) - 1

        def visit(occ, remaining):
            nonlocal winner, winner_key, nodes
            nodes += 1
            _require(nodes <= 100000, "assembly search budget exceeded")
            if not remaining:
                key = (h, w, oy, ox, board.tobytes())
                _require(winner_key is None or key == winner_key, "distinct matching assemblies tie")
                winner_key = key
                winner = (board.copy(), owners.copy(), (oy, ox), tuple(positions))
                return
            free = full ^ occ
            index = (free & -free).bit_length() - 1
            r, c = divmod(index, w)
            for i in remaining:
                x = c - configs[i][2]
                ph, pw = configs[i][0]
                if x < 0 or r + ph > h or x + pw > w:
                    continue
                bits = configs[i][3] << (r * w + x)
                if bits & occ or not place(i, r, x):
                    continue
                visit(occ | bits, tuple(j for j in remaining if j != i))
                for y, z, _ in configs[i][1]:
                    board[r + y, x + z] = -1
                    owners[r + y, x + z] = -1
        visit(occupied, tuple(i for i in range(len(pieces)) if i != anchor))
    _require(winner is not None, "no matching rectangular assembly")
    return winner


def _instruction(board, info):
    colors = info["label_colors"]
    p = [tuple(map(int, np.argwhere(board == c)[0])) for c in colors]
    options = []
    for a, b, source, target in ((p[0], p[1], colors[0], colors[1]),
                                  (p[1], p[0], colors[1], colors[0])):
        d = (b[0] - a[0], b[1] - a[1])
        if d not in DIRS:
            continue
        q = (b[0] + d[0], b[1] + d[1])
        if 0 <= q[0] < board.shape[0] and 0 <= q[1] < board.shape[1] and board[q] == info["open_color"]:
            options.append((source, target, q, d))
    _require(len(options) == 1, "label ray must have one open registration")
    return options[0]


def _decode(g):
    info = _extract(g)
    board, owners, origin, positions = _assemble(info, g.shape)
    source, target, q, direction = _instruction(board, info)
    region = _flood(board == info["open_color"], q)
    _require(len(region) >= 5 and len({int(owners[p]) for p in region}) >= 2,
             "selected chamber must cross a tile seam")
    _require(len(region) < np.count_nonzero(board == info["open_color"]), "need an unselected chamber")
    anchors = [tuple(map(int, p)) for p in np.argwhere(g == source) if tuple(p) not in info["tile_cells"]]
    _require(bool(anchors), "no external source anchors")
    info.update(board=board, owners=owners, origin=origin, positions=positions,
                source=source, target=target, q=q, direction=direction, region=region, anchors=anchors,
                offsets={(r - q[0], c - q[1]) for r, c in region})
    return info


def _print(g, info, offsets=None):
    offsets = info["offsets"] if offsets is None else offsets
    out = g.copy()
    for p in info["tile_cells"] | _marker_cells(info["point"], info["corner"]) | set(info["anchors"]):
        out[p] = info["bg"]
    for ar, ac in info["anchors"]:
        for dr, dc in offsets:
            r, c = ar + dr, ac + dc
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                out[r, c] = info["target"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    return _print(g, _decode(g))


def _partition(rng, h, w, count, stepped):
    rects = [(0, 0, h, w)]
    for _ in range(count - 1):
        choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                   for axis, size in enumerate((ph, pw)) if size >= 6]
        if not choices:
            return None
        i, axis = choices[int(rng.integers(len(choices)))]
        r, c, ph, pw = rects.pop(i)
        cut = int(rng.integers(3, (ph if axis == 0 else pw) - 2))
        if axis == 0:
            rects += [(r, c, cut, pw), (r + cut, c, ph - cut, pw)]
        else:
            rects += [(r, c, ph, cut), (r, c + cut, ph, pw - cut)]
    labels = np.empty((h, w), dtype=np.int8)
    for i, (r, c, ph, pw) in enumerate(rects):
        labels[r:r + ph, c:c + pw] = i
    if stepped:
        options = [(r, c, int(labels[r + dr, c + dc])) for r in range(1, h - 1)
                   for c in range(1, w - 1) for dr, dc in DIRS
                   if labels[r, c] != labels[r + dr, c + dc]]
        for k in rng.permutation(len(options))[:12]:
            r, c, recipient = options[int(k)]
            donor = int(labels[r, c])
            before = labels.copy()
            labels[r, c] = recipient
            if any(np.count_nonzero(labels == j) < 9 or len(_components(labels == j)) != 1
                   for j in (donor, recipient)):
                labels = before
                continue
            # One to three little tabs; occupied masks, not boxes, will tile.
            if rng.random() < .55:
                break
        if all(np.count_nonzero(labels == i) == np.prod(np.ptp(np.argwhere(labels == i), axis=0) + 1)
               for i in range(len(rects))):
            return None
    return labels


def _seam_groups(labels):
    h, w = labels.shape
    roots = list(range(h * w))
    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    for r in range(h):
        for c in range(w):
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if rr < h and cc < w and labels[r, c] != labels[rr, cc]:
                    roots[root(rr * w + cc)] = root(r * w + c)
    groups = {}
    for i in range(h * w):
        groups.setdefault(root(i), []).append(divmod(i, w))
    return list(groups.values())


def _chambers(rng, labels, direction, smaller):
    h, w = labels.shape
    groups = _seam_groups(labels)
    for _ in range(100):
        walls = rng.random((h, w)) < float(rng.uniform(.04, .13))
        # Sealed compartments and partial barriers supply coherent negative space.
        axis = int(rng.integers(2))
        cut = int(rng.integers(2, (h if axis == 0 else w) - 2))
        if axis == 0:
            walls[cut, :] = True
        else:
            walls[:, cut] = True
        for _ in range(int(rng.integers(2, 6))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            length = int(rng.integers(2, 6))
            if rng.integers(2):
                walls[r, c:min(w, c + length)] = True
            else:
                walls[r:min(h, r + length), c] = True
        for group in groups:
            value = any(walls[p] for p in group)
            for p in group:
                walls[p] = value
        if walls.sum() >= h * w * .46:
            continue
        pic = walls.astype(np.int8)  # 0 is open; 1 is a wall.
        # Both colored labels must be strictly inside one occupied tile.
        options = []
        dr, dc = direction
        for r in range(h):
            for c in range(w):
                q, t, s, back = (r, c), (r - dr, c - dc), (r - 2 * dr, c - 2 * dc), (r - 3 * dr, c - 3 * dc)
                if pic[q] != 0 or not (0 <= s[0] < h and 0 <= s[1] < w):
                    continue
                tile = int(labels[s])
                if labels[t] != tile or labels[q] != tile:
                    continue
                if any(not (0 <= p[0] + dy < h and 0 <= p[1] + dx < w)
                       or labels[p[0] + dy, p[1] + dx] != tile for p in (s, t) for dy, dx in DIRS):
                    continue
                if 0 <= back[0] < h and 0 <= back[1] < w and pic[back] == 0:
                    continue
                candidate = pic.copy()
                candidate[s], candidate[t] = 2, 3
                if np.count_nonzero(candidate == 0) <= np.count_nonzero(candidate == 1):
                    continue
                if any(np.count_nonzero((labels == i) & (candidate == color)) < 2
                       for i in range(int(labels.max()) + 1) for color in (0, 1)):
                    continue
                region = _flood(candidate == 0, q)
                components = _components(candidate == 0)
                if not (5 <= len(region) <= 65 and len({int(labels[p]) for p in region}) >= 2
                        and len(region) < np.count_nonzero(candidate == 0)):
                    continue
                if smaller and len(region) == max(map(len, components)):
                    continue
                if sum(r - q[0] for r, c in region) == 0 and sum(c - q[1] for r, c in region) == 0:
                    continue
                options.append((candidate, q, region))
        if options:
            return options[int(rng.integers(len(options)))]
    return None


def _pieces(labels, pic):
    result = []
    for i in range(int(labels.max()) + 1):
        cells = np.argwhere(labels == i)
        r, c = cells.min(axis=0)
        y, x = cells.max(axis=0) + 1
        m = np.where(labels[r:y, c:x] == i, pic[r:y, c:x], -1).astype(np.int8)
        result.append((m, (int(r), int(c))))
    return result


def _scatter(rng, pieces, anchor, origin, point, corner, canvas, bg, marker):
    g = np.full(canvas, bg, dtype=np.int8)
    positions = [None] * len(pieces)
    halo = set()
    def place(i, r, c):
        cells = {(r + y, c + x) for y, x in np.argwhere(pieces[i][0] >= 0)}
        if cells & halo:
            return False
        for y, x in cells:
            g[y, x] = pieces[i][0][y - r, x - c]
        positions[i] = (r, c)
        halo.update((y + dy, x + dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        return True
    r, c = pieces[anchor][1]
    place(anchor, r + origin[0], c + origin[1])
    for p in _marker_cells(point, corner):
        g[p] = marker
        halo.update((p[0] + dy, p[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    order = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i][0].size)
    for i in order:
        h, w = pieces[i][0].shape
        locations = [(r, c) for r in range(canvas[0] - h + 1) for c in range(canvas[1] - w + 1)]
        for k in rng.permutation(len(locations)):
            if place(i, *locations[int(k)]):
                break
        else:
            return None
    return g, halo


def _stamp_cells(offsets, anchor, shape):
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets
            if 0 <= anchor[0] + dr < shape[0] and 0 <= anchor[1] + dc < shape[1]}


def _make_pair(rng, roles, count, direction, corner, smaller, clip, stepped):
    bg, marker, open_color, wall, source, target = roles
    for attempt in range(120):
        h, w = int(rng.integers(7, 13)), int(rng.integers(7, 13))
        labels = _partition(rng, h, w, count, stepped)
        if labels is None:
            continue
        drawn = _chambers(rng, labels, direction, smaller)
        if drawn is None:
            continue
        pic, q, region = drawn
        pic = np.array([open_color, wall, source, target], dtype=np.int8)[pic]
        pieces = _pieces(labels, pic)
        local_point = (h - 1 if corner[0] == "B" else 0, w - 1 if corner[1] == "R" else 0)
        anchor = int(labels[local_point])
        canvas = (int(rng.integers(max(19, h + 8), 31)), int(rng.integers(max(19, w + 8), 31)))
        origin = (int(rng.integers(1, canvas[0] - h)), int(rng.integers(1, canvas[1] - w)))
        point = (origin[0] + local_point[0], origin[1] + local_point[1])
        scattered = _scatter(rng, pieces, anchor, origin, point, corner, canvas, bg, marker)
        if scattered is None:
            continue
        g, halo = scattered
        try:
            info = _extract(g)
            board, owners, found_origin, positions = _assemble(info, canvas)
            if not np.array_equal(board, pic) or found_origin != origin:
                continue
        except ValueError:
            continue
        offsets = {(r - q[0], c - q[1]) for r, c in region}
        available = [(r, c) for r in range(canvas[0]) for c in range(canvas[1]) if (r, c) not in halo]
        rng.shuffle(available)
        full = [p for p in available if len(_stamp_cells(offsets, p, canvas)) == len(offsets)]
        if not full:
            continue
        anchors = [full[0]]
        if clip:
            clipped = [p for p in available if 0 < len(_stamp_cells(offsets, p, canvas)) < len(offsets)]
            if not clipped:
                continue
            anchors.append(clipped[0])
        desired = int(rng.integers(3, 7)) if clip else int(rng.integers(2, 5))
        if rng.random() < .45:
            # Adjacent anchors and print-through are allowed; parse as pixels.
            ar, ac = anchors[0]
            neighbors = [(ar + dr, ac + dc) for dr, dc in DIRS]
            anchors.extend(p for p in neighbors if p in available and p not in anchors)
        for p in available:
            if len(anchors) >= desired:
                break
            if p not in anchors:
                anchors.append(p)
        anchors = anchors[:desired]
        for p in anchors:
            g[p] = source
        paints = set().union(*(_stamp_cells(offsets, p, canvas) for p in anchors))
        inert_colors = [target] + [c for c in range(10) if c not in roles]
        for _ in range(int(rng.integers(0, 5))):
            cells = [p for p in available if g[p] == bg and p not in paints
                     and all(abs(p[0] - a[0]) + abs(p[1] - a[1]) > 1 for a in anchors)]
            if not cells:
                break
            p = cells[int(rng.integers(len(cells)))]
            g[p] = inert_colors[int(rng.integers(len(inert_colors)))]
            # Isolated inert pixels avoid inventing an extra tile or label ray.
            available = [a for a in available if abs(p[0] - a[0]) + abs(p[1] - a[1]) > 1]
        try:
            info = _decode(g)
            out = _print(g, info)
            _check_pair(g, out, info)
        except ValueError:
            continue
        return {"input": g, "output": out}
    raise RuntimeError("bounded chamber construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(name + " is outside its legal integer range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    _require(nt > ne, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen, train_geometry = set(), set()
    first_direction = int(rng.integers(4))
    first_corner = int(rng.integers(4))
    # The first two examples vary the local target for the same source color.
    role_pool = [int(c) for c in rng.permutation(10)]
    for split, total in (("train", nt), ("test", ne)):
        for index in range(total):
            if split == "train" and index < 2:
                source, target = role_pool[0], role_pool[1 + index]
                base = [c for c in rng.permutation(10) if c not in (source, target)][:4]
                roles = tuple(map(int, base)) + (source, target)
                count = 3 + index
                direction = DIRS[(first_direction + index) % 4]
                corner = CORNERS[(first_corner + 3 * index) % 4]
            else:
                roles = tuple(map(int, rng.permutation(10)[:6]))
                count = int(rng.integers(3, 7))
                direction, corner = DIRS[int(rng.integers(4))], CORNERS[int(rng.integers(4))]
            for _ in range(12):
                pair = _make_pair(rng, roles, count, direction, corner,
                                  smaller=split == "train" and index == 1,
                                  clip=index == 1 or split == "test" or rng.random() < .35,
                                  stepped=(index == 1 if split == "train" and index < 2
                                           else bool(rng.random() < .45)))
                key = (pair["input"].shape, pair["input"].tobytes())
                info = _decode(pair["input"])
                geometry = tuple(sorted(info["offsets"]))
                if key in seen or split == "test" and geometry in train_geometry:
                    continue
                seen.add(key)
                if split == "train":
                    train_geometry.add(geometry)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not produce a distinct new input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(g, out, info=None):
    info = _decode(g) if info is None else info
    _require(out.shape == g.shape and np.array_equal(out, _print(g, info)), "incorrect print")
    _require(not np.array_equal(g, out), "identity example")
    _require(not np.any(out == info["marker"]), "marker color remains")
    board, owners, q = info["board"], info["owners"], info["q"]
    all_open = {(int(r) - q[0], int(c) - q[1]) for r, c in np.argwhere(board == info["open_color"])}
    local_mask = (board == info["open_color"]) & (owners == owners[q])
    local = {(r - q[0], c - q[1]) for r, c in _flood(local_mask, q)}
    _require(not np.array_equal(out, _print(g, info, local)), "omitting assembly does not change this answer")
    _require(not np.array_equal(out, _print(g, info, all_open)), "ignoring chamber selection gives the same answer")
    r0 = min(r for r, c in info["region"])
    r1 = max(r for r, c in info["region"])
    c0 = min(c for r, c in info["region"])
    c1 = max(c for r, c in info["region"])
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    centered = {(r - center[0], c - center[1]) for r, c in info["region"]}
    _require(not np.array_equal(out, _print(g, info, centered)), "centering is an undistinguished shortcut")
    chambers = _components(board == info["open_color"])
    largest_size = max(map(len, chambers))
    if len(info["region"]) < largest_size:
        for chamber in chambers:
            if len(chamber) == largest_size:
                largest = {(r - q[0], c - q[1]) for r, c in chamber}
                _require(not np.array_equal(out, _print(g, info, largest)),
                         "largest-chamber shortcut is undistinguished")
    complete = sum(len(_stamp_cells(info["offsets"], p, g.shape)) == len(info["offsets"]) for p in info["anchors"])
    _require(complete >= 1, "need a complete stamp to expose exact geometry")
    # Every original tile cell appears unchanged in the reconstructed board.
    oy, ox = info["origin"]
    for i, ((m, old), (r, c)) in enumerate(zip(info["pieces"], info["positions"])):
        _require(np.array_equal(board[r:r + m.shape[0], c:c + m.shape[1]][m >= 0], m[m >= 0]),
                 "assembly changed tile content")
        if i == info["anchor"]:
            _require((r + oy, c + ox) == old, "anchor moved")
    unrelated = set(map(tuple, np.argwhere((g != info["bg"]) & (g != info["marker"])).tolist()))
    unrelated -= info["tile_cells"] | set(info["anchors"])
    paints = set().union(*(_stamp_cells(info["offsets"], p, g.shape) for p in info["anchors"]))
    _require(all(out[p] == g[p] for p in unrelated - paints), "unrelated cells changed")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            key = (g.shape, g.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(_check_pair(g, out))
    train = infos["train"]
    _require(len({x["direction"] for x in train}) >= 2, "training must show two label directions")
    _require(any(any(np.count_nonzero(m >= 0) != m.size for m, _ in x["pieces"]) for x in train),
             "training must demonstrate stepped occupancy")
    _require(any(any(len(_stamp_cells(x["offsets"], p, np.asarray(puzzle["train"][i]["input"]).shape)) < len(x["offsets"])
                     for p in x["anchors"]) for i, x in enumerate(train)),
             "training must demonstrate clipping")
    _require(any(len(x["region"]) < max(map(len, _components(x["board"] == x["open_color"])))
                 for x in train), "training must distinguish largest-chamber selection")
    mappings = {}
    for x in train:
        mappings.setdefault(x["source"], set()).add(x["target"])
    _require(any(len(v) >= 2 for v in mappings.values()), "training must distinguish memorized targets")
    familiar = {tuple(sorted(x["offsets"])) for x in train}
    for pair, x in zip(puzzle["test"], infos["test"]):
        _require(tuple(sorted(x["offsets"])) not in familiar, "test needs a new registered chamber")
        _require(any(len(_stamp_cells(x["offsets"], p, np.asarray(pair["input"]).shape)) < len(x["offsets"])
                     for p in x["anchors"]), "test must combine complete and clipped impressions")


def _semantic_regressions():
    # Three horizontal strips. The unequal boundary signatures force their
    # order; only the right-hand seven-cell chamber is registered for printing.
    # These grids and expected offsets are hand specified, not sampled.
    rows = ("222332222", "222333142", "223222332", "223222332", "333333333",
            "232222223", "232222223", "222332222", "322232232")
    board = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    g = np.zeros((20, 22), dtype=np.int8)
    for start, (r, c) in zip((0, 3, 6), ((2, 2), (12, 1), (7, 11))):
        g[r:r + 3, c:c + 9] = board[start:start + 3]
    for p in ((1, 1), (1, 2), (2, 1)):
        g[p] = 9
    anchors = ((2, 18), (3, 18), (18, 0))
    for p in anchors:
        g[p] = 1
    g[0, 0], g[19, 21] = 5, 4
    expected = np.zeros_like(g)
    expected[0, 0], expected[19, 21] = 5, 4
    offsets = {(-1, -3), (-1, -2), (-1, -1), (-1, 0), (0, 0), (1, 0), (2, 0)}
    for ar, ac in anchors:
        for dr, dc in offsets:
            r, c = ar + dr, ac + dc
            if 0 <= r < 20 and 0 <= c < 22:
                expected[r, c] = 4
    _require(np.array_equal(solve(g), expected), "hand-specified chamber regression failed")
    info = _check_pair(g, expected)
    _require(info["offsets"] == offsets, "wrong registered chamber")
    _require((3, 18) in _stamp_cells(offsets, (2, 18), g.shape), "print-through regression missing")
    # Segmentation order cannot select the tiling.
    reordered = _extract(g)
    reordered["pieces"] = list(reversed(reordered["pieces"]))
    reordered["anchor"] = len(reordered["pieces"]) - 1 - reordered["anchor"]
    _require(np.array_equal(_assemble(reordered, g.shape)[0], board), "piece-order dependence")
    # Distinct pictures with equal seams must be rejected, not chosen by order.
    pieces = []
    for i in range(3):
        m = np.full((3, 9), 2, dtype=np.int8)
        m[1, 2 + i] = 3
        pieces.append((m, (2, 2) if i == 0 else (12, 2 + i)))
    ambiguous = dict(pieces=pieces, point=(2, 2), corner="TL", anchor=0)
    try:
        _assemble(ambiguous, (20, 22))
    except ValueError as error:
        _require("tie" in str(error), "ambiguity regression failed for another reason")
    else:
        raise AssertionError("distinct ambiguous assemblies accepted")


def self_test():
    """Bounded semantic regressions and 42-seed structural measurements."""
    _semantic_regressions()
    shapes, boards, partitions, masks, corners, directions, counts, roles = [set() for _ in range(8)]
    clipped = overlap = stepped = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                               for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                x = _check_pair(g, out)
                shapes.add(g.shape)
                boards.add(x["board"].shape)
                partitions.add((x["owners"].shape, x["owners"].tobytes()))
                masks.add(tuple(sorted(x["offsets"])))
                corners.add(x["corner"])
                directions.add(x["direction"])
                counts.add(len(x["pieces"]))
                roles.add((x["bg"], x["marker"], x["open_color"], x["wall"], x["source"], x["target"]))
                footprints = [_stamp_cells(x["offsets"], p, g.shape) for p in x["anchors"]]
                clipped += any(len(f) < len(x["offsets"]) for f in footprints)
                overlap += sum(map(len, footprints)) > len(set().union(*footprints))
                stepped += any(np.count_nonzero(m >= 0) != m.size for m, _ in x["pieces"])
                _require(np.array_equal(solve(g), out), "solver mismatch")
    _require(len(fingerprints) == 42, "seed diversity missing")
    _require(len(shapes) >= 30 and len(boards) >= 15 and len(partitions) >= 100 and len(masks) >= 100,
             "structural variation missing")
    _require(corners == set(CORNERS) and directions == set(DIRS) and counts == {3, 4, 5, 6},
             "orientation or count variation missing")
    _require(len(roles) >= 100 and clipped >= 42 and overlap >= 20 and stepped >= 20,
             "role, overlap, clipping or occupancy variation missing")
    a, b = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p),
             "seed is nondeterministic")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit four-neighbor regression: diagonal open cells never connect.
    _require(_flood(np.array([[1, 0], [0, 1]], dtype=bool), (0, 0)) == {(0, 0)}, "diagonal leakage")
    damaged = {s: [{k: g.copy() for k, g in p.items()} for p in pairs] for s, pairs in a.items()}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), canvas_shapes=len(shapes),
                board_shapes=len(boards), partitions=len(partitions), registered_chambers=len(masks),
                clipped_pairs=clipped, overlapping_pairs=overlap, stepped_pairs=stepped)


if __name__ == "__main__":
    print(self_test())
