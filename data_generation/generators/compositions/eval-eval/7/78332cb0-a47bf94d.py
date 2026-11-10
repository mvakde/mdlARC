"""Conductive jigsaw: edge assembly and cable labels jointly determine order.

No parent modules are imported. Construction draws explicit paths; solve parses
the input, traces a directional graph, and tests every geometric assembly.
All routines are bounded: <=4 tiles, <=6 strands, <=2 geometric orders.
"""
from itertools import permutations, product
from collections import Counter
import hashlib
import numpy as np


DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 315834799, 691668884, 724059097, 171334592, 821686921,
              57852024, 418824077, 578140334, 233601399, 374031607,
              763177810, 424675705, 491253988, 950630815, 21678519,
              95213080, 442341985, 413327758, 687959589, 667359722,
              702192042, 807205375, 261805852, 663506999, 345290359,
              151722332, 325670903, 695631677, 564001910, 234594245,
              359132011, 430477754, 499329373, 314123767, 950576303,
              384037916, 102831527, 247626858, 763660042, 157642301,
              659932270)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid size")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _partition(grid):
    """Discover full separator lines and equal tile dimensions from cells."""
    found = []
    h, w = grid.shape
    for color in map(int, np.unique(grid)):
        rr = [-1] + list(map(int, np.flatnonzero(np.all(grid == color, axis=1)))) + [h]
        cc = [-1] + list(map(int, np.flatnonzero(np.all(grid == color, axis=0)))) + [w]
        if len(rr) == len(cc) == 2:
            continue
        heights = [b - a - 1 for a, b in zip(rr, rr[1:])]
        widths = [b - a - 1 for a, b in zip(cc, cc[1:])]
        if min(heights + widths) < 7 or len(set(heights)) != 1 or len(set(widths)) != 1:
            continue
        tiles = [grid[r + 1:rr[i + 1], c + 1:cc[j + 1]].copy()
                 for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
        if not 2 <= len(tiles) <= 4 or any(np.any(t == color) for t in tiles):
            continue
        corners = {int(t[r, c]) for t in tiles for r, c in
                   ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        found.append(dict(tiles=tiles, background=corners.pop(), divider=color,
                          layout=(len(heights), len(widths))))
    _need(len(found) == 1, "separator partition is not unique")
    return found[0]


def _signature(tile, carrier):
    return (tuple(map(int, np.flatnonzero(tile[:, 0] == carrier))),
            tuple(map(int, np.flatnonzero(tile[:, -1] == carrier))))


def _block_kind(block, bg, source):
    if np.all(block == bg):
        return "blank", None
    colors = set(map(int, np.unique(block))) - {bg}
    _need(len(colors) == 1, "terminal uses multiple labels")
    color = colors.pop()
    if source and np.all(block == color):
        return "solid", color
    pattern = DIAMOND if source else CHECKER
    _need(np.array_equal(block, np.where(pattern, color, bg)), "invalid terminal stencil")
    return ("diamond" if source else "checker"), color


def _cap(tile, rows, bg, carrier, source):
    h, w = tile.shape
    _need(2 <= len(rows) <= 6 and all(b - a >= 4 for a, b in zip(rows, rows[1:])),
          "overlapping terminal sites")
    _need(rows[0] >= 1 and rows[-1] < h - 1, "terminal exceeds tile")
    column = 1 if source else w - 4
    expected = np.full(tile.shape, bg, dtype=np.int8)
    clues, footprints = [], []
    for row in rows:
        site = (slice(row - 1, row + 2), slice(column, column + 3))
        clues.append(_block_kind(tile[site], bg, source))
        footprints.append(site)
        expected[row, 4:w] = carrier if source else bg
        if not source:
            expected[row, :w - 4] = carrier
        expected[site] = tile[site]
    _need(np.array_equal(expected, tile), "cap has extra cells or broken leads")
    return clues, footprints


def _trace_body(tile, bg, carrier, cross, incoming, outgoing):
    """Trace cells and incoming direction; crossings never join their strands."""
    _need(set(map(int, np.unique(tile))) <= {bg, carrier, cross}, "extra routing color")
    occupied = set(map(tuple, np.argwhere(tile != bg)))
    h, w = tile.shape
    starts, ends = {(r, 0) for r in incoming}, {(r, w - 1) for r in outgoing}
    neighbors = {}
    for r, c in occupied:
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRS)
                 if (r + dr, c + dc) in occupied}
        neighbors[r, c] = edges
        if (r, c) in starts | ends:
            _need(tile[r, c] == carrier and len(edges) == 1, "branched port")
        elif tile[r, c] == cross:
            _need(len(edges) == 4, "crossing must have four arms")
        else:
            _need(len(edges) == 2, "ordinary wire must not branch or stop")
    mapping, routes, all_seen = {}, [], set()
    for row in incoming:
        p, entry = (row, 0), 3
        states, route = set(), []
        while True:
            _need((p, entry) not in states, "wire cycle")
            states.add((p, entry))
            route.append(p)
            all_seen.add(p)
            if p in ends:
                mapping[row] = int(p[0])
                break
            _need(p == (row, 0) or p not in starts, "wire returns to entry edge")
            edges = neighbors[p]
            if p == (row, 0):
                direction = 1
            elif tile[p] == cross:
                direction = (entry + 2) % 4
            else:
                _need(entry in edges, "wire entered through a missing arm")
                direction = next(d for d in edges if d != entry)
            _need(direction in edges, "wire cannot continue")
            p, entry = edges[direction], (direction + 2) % 4
        routes.append(route)
    _need(sorted(mapping.values()) == list(outgoing), "nonbijective tile")
    _need(all_seen == occupied, "unattached wire fragments")
    return mapping, routes


def _orders(signatures):
    start = [i for i, (a, b) in enumerate(signatures) if not a and b]
    end = [i for i, (a, b) in enumerate(signatures) if a and not b]
    _need(len(start) == len(end) == 1, "one cap of each direction required")
    middle = [i for i in range(len(signatures)) if i not in start + end]
    _need(all(signatures[i][0] and signatures[i][1] for i in middle), "isolated tile")
    orders = []
    for p in permutations(middle):
        order = (start[0],) + p + (end[0],)
        if all(signatures[a][1] == signatures[b][0] for a, b in zip(order, order[1:])):
            orders.append(order)
    return orders


def _routing(order, signatures, maps):
    left, right = signatures[order[0]][1], signatures[order[-1]][0]
    result = []
    for row in left:
        for index in order[1:-1]:
            row = maps[index][row]
        result.append(right.index(row))
    return tuple(result)


def _labels_for(permutation, clues):
    labels = []
    for i, j in enumerate(permutation):
        labels_here = {v for _, v in (clues[0][i], clues[1][j]) if v is not None}
        if len(labels_here) != 1:
            return None
        labels.append(labels_here.pop())
    return labels


def _interpret(parsed, vertical, carrier):
    bg = parsed["background"]
    tiles = [t.T.copy() if vertical else t.copy() for t in parsed["tiles"]]
    h, w = tiles[0].shape
    _need(w >= 7 and len(tiles) * w <= 30, "assembled strip too long")
    _need(all(not np.any(t[0] == carrier) and not np.any(t[-1] == carrier) for t in tiles),
          "wire touches a transverse edge")
    signatures = [_signature(t, carrier) for t in tiles]
    orders = _orders(signatures)
    _need(bool(orders), "edge signatures cannot assemble")
    start, end = orders[0][0], orders[0][-1]
    counts = {len(s) for pair in signatures for s in pair if s}
    _need(len(counts) == 1, "strand count changes at a seam")
    k = counts.pop()
    _need(2 <= k <= 6, "strand count out of range")
    left_clues, left_sites = _cap(tiles[start], signatures[start][1], bg, carrier, True)
    right_clues, right_sites = _cap(tiles[end], signatures[end][0], bg, carrier, False)
    labels_used = {v for bank in (left_clues, right_clues) for _, v in bank if v is not None}
    _need(labels_used.isdisjoint({carrier, bg, parsed["divider"]}), "labels collide with roles")
    middle = orders[0][1:-1]
    extra = {int(v) for i in middle for v in np.unique(tiles[i])} - {bg, carrier}
    _need(len(extra) <= 1 and extra.isdisjoint(labels_used), "ambiguous crossover role")
    cross = next(iter(extra)) if extra else -1
    maps, routes = {}, {}
    for i in middle:
        maps[i], routes[i] = _trace_body(tiles[i], bg, carrier, cross, *signatures[i])
    matches = []
    for order in orders:
        perm = _routing(order, signatures, maps)
        labels = _labels_for(perm, (left_clues, right_clues))
        if labels is not None:
            matches.append((order, perm, labels))
    _need(len(matches) == 1, "endpoint clues must select exactly one assembly")
    order, permutation, labels = matches[0]
    return dict(parsed, tiles=tiles, vertical=vertical, carrier=carrier, cross=cross,
                signatures=signatures, geometric_orders=orders, order=order,
                permutation=permutation, labels=labels, clues=(left_clues, right_clues),
                sites=(left_sites, right_sites), maps=maps, routes=routes,
                strands=k, tile_shape=(h, w))


def _decode(input_grid):
    grid = _grid(input_grid)
    parsed = _partition(grid)
    found = []
    for vertical in (False, True):
        for carrier in sorted(set(map(int, np.unique(grid))) -
                              {parsed["background"], parsed["divider"]}):
            try:
                found.append(_interpret(parsed, vertical, carrier))
            except (ValueError, KeyError, StopIteration):
                continue
    _need(len(found) == 1, "input does not determine one interpretation")
    return found[0]


def _render(info, complete=True):
    ordered = [info["tiles"][i].copy() for i in info["order"]]
    if complete:
        for i, j in enumerate(info["permutation"]):
            color = info["labels"][i]
            for tile, site, pattern in ((ordered[0], info["sites"][0][i], DIAMOND),
                                        (ordered[-1], info["sites"][1][j], CHECKER)):
                tile[site] = np.where(pattern, color, info["background"])
    output = np.concatenate(ordered, axis=1)
    return output.T.copy() if info["vertical"] else output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover order, routes, label colors and output solely from input cells."""
    return _render(_decode(input_grid))


def _event_columns(rng, width, count):
    if not count:
        return []
    minimum = 2 * count + 1
    _need(width >= minimum, "tile too narrow for routing events")
    gaps = [1] + [2] * (count - 1) + [1]
    for _ in range(width - minimum):
        gaps[int(rng.integers(len(gaps)))] += 1
    result, c = [], 0
    for gap in gaps[:-1]:
        c += gap
        result.append(c)
    return result


def _draw_paths(paths, h, w):
    visits = Counter(p for path in paths for p in path)
    _need(max(visits.values()) <= 2, "more than two wires overlap")
    field = np.zeros((h, w), dtype=np.int8)
    for p, count in visits.items():
        field[p] = 2 if count == 2 else 1
    return field


def _stable_body(rng, rows, spare, h, w, operation=None):
    """Use a vacant row to route a swap/cycle with orthogonal paths."""
    k = len(rows)
    if operation is None:
        kinds = ["straight", "detour", "swap"] + (["cycle"] if k >= 3 and w >= 9 else [])
        kind = _choose(rng, kinds)
        indices = list(map(int, rng.permutation(k)))
        operation = (kind, indices)
    kind, indices = operation
    moves = []
    if kind == "detour":
        a = indices[0]
        moves = [(a, k), (k, a)]
    elif kind == "swap":
        a, b = indices[:2]
        moves = [(a, k), (b, a), (k, b)]
    elif kind == "cycle":
        a, b, c = indices[:3]
        moves = [(a, k), (b, a), (c, b), (k, c)]
    elif kind != "straight":
        raise ValueError("unknown routing operation")
    columns = _event_columns(rng, w, len(moves))
    tracks = list(rows) + [spare]
    state = list(range(k)) + [None]
    paths = [[(row, 0)] for row in rows]
    previous = 0
    for column, (old, new) in zip(columns, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((tracks[lane], c) for c in range(previous + 1, column + 1))
        route = state[old]
        _need(route is not None and state[new] is None, "occupied destination track")
        step = 1 if tracks[new] > tracks[old] else -1
        paths[route].extend((r, column) for r in range(tracks[old] + step, tracks[new] + step, step))
        state[new], state[old] = route, None
        previous = column
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((tracks[lane], c) for c in range(previous + 1, w))
    mapping = {int(path[0][0]): int(path[-1][0]) for path in paths}
    return _draw_paths(paths, h, w), mapping


def _changing_body(rng, incoming, outgoing, h, w, perm=None):
    """Disjoint port profiles permit independently located single-bend pairs."""
    k = len(incoming)
    if perm is None:
        perm = list(map(int, rng.permutation(k)))
    columns = _event_columns(rng, w, k)
    rng.shuffle(columns)
    paths = []
    for i, row in enumerate(incoming):
        col, dest = columns[i], outgoing[perm[i]]
        step = 1 if dest > row else -1
        path = [(row, c) for c in range(col + 1)]
        path.extend((r, col) for r in range(row + step, dest + step, step))
        path.extend((dest, c) for c in range(col + 1, w))
        paths.append(path)
    mapping = {int(path[0][0]): int(path[-1][0]) for path in paths}
    return _draw_paths(paths, h, w), mapping


def _profile(rng, k, max_height):
    minimum = 4 * k + 2
    _need(minimum <= max_height, "profile exceeds available height")
    h = int(rng.integers(minimum, max_height + 1))
    gaps, slack = [4] * (k - 1), h - minimum
    for i in map(int, rng.permutation(k - 1)):
        extra = int(rng.integers(min(2, slack) + 1))
        gaps[i] += extra
        slack -= extra
    offset = int(rng.integers(slack + 1))
    rows_list = [2 + offset]
    for gap in gaps:
        rows_list.append(rows_list[-1] + gap)
    rows = tuple(rows_list)
    other = tuple(r + 2 for r in rows)
    spare = rows[-1] + 2
    return h, rows, other, spare


def _cap_field(h, w, rows, source):
    field = np.zeros((h, w), dtype=np.int8)
    for row in rows:
        if source:
            field[row, 4:w] = 1
        else:
            field[row, :w - 4] = 1
    return field


def _layout(tiles, order, shape, divider):
    nr, nc = shape
    h, w = tiles[0].shape
    grid = np.full((nr * (h + 1) - 1, nc * (w + 1) - 1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        grid[r * (h + 1):r * (h + 1) + h,
             c * (w + 1):c * (w + 1) + w] = tiles[i]
    return grid


def _make_pair(rng, kind, vertical, lesson=None, repeated=False, max_k=3, min_k=2):
    joint = kind == "joint"
    if kind in ("geometry", "mixed", "joint"):
        n, w = 4, 7
        k = 3 if joint else int(rng.integers(2, min(3, max_k) + 1))
        h, rows, other, spare = _profile(rng, k, 14)
    else:
        n = 2 if kind == "simple" else 3
        kmax = min(max_k, 4 if kind == "bridge" else 6)
        k = int(rng.integers(min_k, kmax + 1))
        h, rows, other, spare = _profile(rng, k, 30)
        w = int(rng.integers(max(7, 2 * k + 1) if kind == "bridge" else 7,
                             15 if n == 2 else 10))
    fields, maps = [_cap_field(h, w, rows, True)], []
    current = rows
    if joint:
        a, b, c = map(int, rng.permutation(3))
        for indices in ([a, b], [b, c]):
            field, mapping = _stable_body(rng, rows, spare, h, w, ("swap", indices))
            fields.append(field)
            maps.append(mapping)
    elif kind == "geometry":
        p = list(map(int, rng.permutation(k)))
        if p == list(range(k)):
            p[0], p[-1] = p[-1], p[0]
        for dest, perm in ((other, p), (rows, list(range(k)))):
            field, mapping = _changing_body(rng, current, dest, h, w, perm)
            fields.append(field)
            maps.append(mapping)
            current = dest
    elif kind == "mixed":
        # Novel combination of the two lesson types: a repeated interface
        # profile alongside a changing one, rather than two of either type.
        first_stable = bool(rng.integers(2))
        for stable in (first_stable, not first_stable):
            if stable:
                local_spare = current[-1] + 2 if current == rows else rows[0]
                indices = list(map(int, rng.permutation(k)))
                field, mapping = _stable_body(rng, current, local_spare, h, w, ("swap", indices))
            else:
                field, mapping = _changing_body(rng, current, other, h, w)
                current = other
            fields.append(field)
            maps.append(mapping)
    elif n == 3:
        if kind == "bridge":
            field, mapping = _changing_body(rng, rows, other, h, w)
            current = other
        else:
            field, mapping = _stable_body(rng, rows, spare, h, w)
        fields.append(field)
        maps.append(mapping)
    fields.append(_cap_field(h, w, current, False))
    perm = []
    for row in rows:
        for mapping in maps:
            row = mapping[row]
        perm.append(current.index(row))
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, cross, divider = palette[:4]
    labels = list(palette[4:4 + k])
    if repeated and k >= 3:
        labels = [palette[4 + i % 2] for i in range(k)]
        rng.shuffle(labels)
    elif lesson is None and not joint and rng.random() < .4:
        count = int(rng.integers(1, min(k, 6) + 1))
        labels = list(map(int, rng.choice(palette[4:4 + count], k)))
    if lesson == "forward":
        modes = [("solid", "blank")] * k
        modes[int(rng.integers(k))] = ("diamond", "blank")
    elif joint and lesson == "joint":
        # All receiving clues plus two marked source clues expose the order;
        # one reverse-only route still requires genuine completion.
        indices = list(map(int, rng.permutation(k)))
        modes = [("solid", "checker")] * k
        modes[indices[0]] = ("diamond", "checker")
        modes[indices[1]] = ("blank", "checker")
    elif joint:
        # Vary which bank is incomplete. Of the 125 possible three-route
        # clue plans, retain those with an explicit color conflict under the
        # alternate assembly, a double clue, and a missing endpoint.
        alternate = []
        for row in rows:
            for mapping in reversed(maps):
                row = mapping[row]
            alternate.append(current.index(row))
        choices = (("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
                   ("solid", "checker"), ("diamond", "checker"))
        eligible = []
        for candidate in product(choices, repeat=k):
            if not any("blank" in mode for mode in candidate):
                continue
            if not any(a != "blank" and b != "blank" for a, b in candidate):
                continue
            left = [labels[i] if a != "blank" else None for i, (a, b) in enumerate(candidate)]
            right = [None] * k
            for i, (a, b) in enumerate(candidate):
                if b != "blank":
                    right[perm[i]] = labels[i]
            if any(left[i] is not None and right[j] is not None and left[i] != right[j]
                   for i, j in enumerate(alternate)):
                eligible.append(candidate)
        _need(bool(eligible), "no disambiguating clue plan")
        modes = _choose(rng, eligible)
    else:
        choices = (("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
                   ("solid", "checker"), ("diamond", "checker"))
        modes = [_choose(rng, choices) for _ in range(k)]
        if kind == "mixed":
            modes[0], modes[1] = choices[0], choices[2]
    tiles = [np.asarray([bg, carrier, cross], dtype=np.int8)[f] for f in fields]
    outputs = [t.copy() for t in tiles]
    for i, j in enumerate(perm):
        for index, row, col, pattern, mode in (
                (0, rows[i], 1, DIAMOND, modes[i][0]),
                (n - 1, current[j], w - 4, CHECKER, modes[i][1])):
            site = (slice(row - 1, row + 2), slice(col, col + 3))
            block = np.where(pattern, labels[i], bg).astype(np.int8)
            outputs[index][site] = block
            tiles[index][site] = (np.full((3, 3), bg if mode == "blank" else labels[i], dtype=np.int8)
                                  if mode in ("blank", "solid") else block)
    if rng.random() < .5:
        tiles = [t[::-1].copy() for t in tiles]
        outputs = [t[::-1].copy() for t in outputs]
    if vertical:
        tiles = [t.T.copy() for t in tiles]
    th, tw = tiles[0].shape
    layouts = [(nr, n // nr) for nr in range(1, n + 1) if n % nr == 0
               and nr * (th + 1) - 1 <= 30 and (n // nr) * (tw + 1) - 1 <= 30]
    _need(bool(layouts), "input tiling does not fit")
    order = list(map(int, rng.permutation(n)))
    if lesson and order == list(range(n)):
        order = order[1:] + order[:1]
    inp = _layout(tiles, order, _choose(rng, layouts), divider)
    out = np.concatenate(outputs, axis=1)
    if vertical:
        out = out.T.copy()
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # NumPy accepts nonnegative seeds; this also gives negative integer seeds
    # a deterministic interpretation while leaving all nonnegative seeds intact.
    seed = int(seed) if int(seed) >= 0 else int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    shift = _choose(rng, ("mixed_seams", "larger_bundle", "repeated_labels"))
    orientation = _choose(rng, ("same", "mixed", "shift"))
    base_vertical = bool(rng.integers(2))
    lessons = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = None
            repeated = False
            max_k = 3
            if split == "train":
                if index == lessons[0]:
                    kind, lesson = "geometry", "forward"
                elif index == lessons[1]:
                    kind, lesson = "joint", "joint"
                else:
                    kind = _choose(rng, ("simple", "bridge", "stable", "mixed", "joint"))
                    if shift == "repeated_labels":
                        lesson = "distinct"
            elif shift == "mixed_seams":
                kind = "mixed"
            elif shift == "larger_bundle":
                kind, max_k = "stable", 6
            else:
                kind, repeated = "joint", True
            vertical = (bool(rng.integers(2)) if orientation == "mixed" else
                        (not base_vertical if split == "test" and orientation == "shift" else base_vertical))
            # At most 16 attempts, holding only the current pair. Rejections
            # concern duplicate inputs, never a hidden
            # query answer. The constructor and decoder must otherwise agree.
            for attempt in range(16):
                min_k = 4 if split == "test" and shift == "larger_bundle" else 2
                pair = _make_pair(rng, kind, vertical, lesson, repeated, max_k, min_k)
                info = _decode(pair["input"])
                _need(np.array_equal(_render(info), pair["output"]), "construction/tracing disagreement")
                if _key(pair["input"]) not in seen:
                    break
            else:
                # The 40 affine palette permutations are distinct on any
                # >=3-color input. There are at most 13 earlier inputs.
                done = False
                for multiplier in (1, 3, 7, 9):
                    for offset in range(10):
                        lut = ((multiplier * np.arange(10) + offset) % 10).astype(np.int8)
                        fresh = {key: lut[value] for key, value in pair.items()}
                        if _key(fresh["input"]) not in seen:
                            pair, done = fresh, True
                            break
                    if done:
                        break
                _need(done, "finite palette fallback exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def inspect_pair(pair):
    """Compact, input-derived semantic measurements, without generation plans."""
    info = _decode(pair["input"])
    clues = [(info["clues"][0][i], info["clues"][1][j])
             for i, j in enumerate(info["permutation"])]
    geometry_count = len(info["geometric_orders"])
    port_sets = {s for pair_s in info["signatures"] for s in pair_s if s}
    return dict(tiles=len(info["tiles"]), strands=info["strands"],
                tile_shape=info["tile_shape"], layout=info["layout"],
                vertical=info["vertical"], order=info["order"],
                geometry_orders=geometry_count, interface_profiles=len(port_sets),
                permutation=info["permutation"],
                crossings=sum(int(np.count_nonzero(t == info["cross"])) for t in info["tiles"]),
                repeated_labels=len(set(info["labels"])) < info["strands"],
                forward=sum(a[0] != "blank" and b[0] == "blank" for a, b in clues),
                reverse=sum(a[0] == "blank" and b[0] != "blank" for a, b in clues),
                both=sum(a[0] != "blank" and b[0] != "blank" for a, b in clues),
                solid=sum(a[0] == "solid" for a, _ in clues),
                canonical=sum(a[0] == "diamond" for a, _ in clues),
                roles=(info["background"], info["carrier"], info["cross"], info["divider"]))


def validate_generated_puzzle(puzzle):
    """Check input-defined semantics, exact preservation, and teaching evidence."""
    try:
        _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
        _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        nt, nq = len(puzzle["train"]), len(puzzle["test"])
        _count(nt, "train count", 2, 10)
        _count(nq, "test count", 1, 4)
        _need(nt > nq, "invalid split counts")
        seen, teaching = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                _need(_key(inp) not in seen, "duplicate input")
                seen.add(_key(inp))
                info = _decode(inp)
                _need(np.array_equal(_render(info), out), "incorrect assembled/completed output")
                # Explicit conservation check, independent of the renderer:
                # split the snug output, zero ONLY the known terminal sites,
                # and compare every tile cell with its input counterpart.
                canonical = out.T if info["vertical"] else out
                h, w = info["tile_shape"]
                _need(canonical.shape == (h, len(info["tiles"]) * w), "incorrect output crop")
                for slot, index in enumerate(info["order"]):
                    actual = canonical[:, slot * w:(slot + 1) * w].copy()
                    original = info["tiles"][index].copy()
                    sites = info["sites"][0] if slot == 0 else info["sites"][1] if slot == len(info["tiles"]) - 1 else []
                    for site in sites:
                        actual[site] = original[site] = info["background"]
                    _need(np.array_equal(actual, original), "routing tile content changed")
                if split == "train":
                    m = inspect_pair(pair)
                    if m["geometry_orders"] == 2:
                        teaching.add("joint")
                    if m["geometry_orders"] == 1 and m["interface_profiles"] >= 2:
                        teaching.add("geometry")
                    if m["order"] != tuple(range(m["tiles"])):
                        teaching.add("reorder")
                    if m["permutation"] != tuple(range(m["strands"])):
                        teaching.add("routing")
                    for name in ("forward", "reverse", "both", "solid", "canonical"):
                        if m[name]:
                            teaching.add(name)
        _need({"joint", "geometry", "reorder", "routing", "forward", "reverse", "both", "solid", "canonical"}
              <= teaching, "demonstrations omit an essential mechanism")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise AssertionError(str(exc)) from exc


def self_test():
    """Stream 42 seeds, exercise ablations and mutations; never write artifacts."""
    stats = Counter()
    shapes, tile_shapes, structures, roles, orientations, permutations_seen = set(), set(), set(), set(), set(), set()
    puzzle_hashes = set()
    reference = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                predicted = solve(pair["input"].copy())
                assert np.array_equal(predicted, pair["output"])
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                m = inspect_pair(pair)
                shapes.add(pair["input"].shape)
                tile_shapes.add(m["tile_shape"])
                roles.add(m["roles"])
                orientations.add(m["vertical"])
                permutations_seen.add(m["permutation"])
                stats[("tiles", m["tiles"])] += 1
                stats[("strands", m["strands"])] += 1
                stats[("geometry_orders", m["geometry_orders"])] += 1
                stats[("repeated", m["repeated_labels"])] += 1
                stats[("profiles", m["interface_profiles"])] += 1
                stats[("crossings_present", m["crossings"] > 0)] += 1
                for tile in info["tiles"]:
                    mask = (tile == info["carrier"]) | (tile == info["cross"])
                    structures.add((mask.shape, mask.tobytes()))
                if len(info["geometric_orders"]) == 2:
                    # Omit B: geometric matching alone leaves two placements.
                    assert len({_render(dict(info, order=order), complete=False).tobytes()
                                for order in info["geometric_orders"]}) == 2
                    # Preserve endpoint clues but ignore internal cable routing:
                    # identity slot matching conflicts with at least one clue.
                    if not m["repeated_labels"]:
                        assert _labels_for(tuple(range(m["strands"])), info["clues"]) is None
                if split == "train" and m["geometry_orders"] == 1 and m["interface_profiles"] >= 2:
                    # Omit A: the two middle fragments cannot be swapped; an
                    # actual profile mismatch proves placement matters.
                    order = list(info["order"])
                    if len(order) == 4:
                        order[1], order[2] = order[2], order[1]
                        assert any(info["signatures"][a][1] != info["signatures"][b][0]
                                   for a, b in zip(order, order[1:]))
        puzzle_hashes.add(digest.digest())
        if seed == 0:
            reference = puzzle
    assert len(puzzle_hashes) == 42
    assert len(shapes) >= 15 and len(tile_shapes) >= 15 and len(structures) >= 100
    assert len(roles) >= 100 and orientations == {False, True}
    assert {k[1] for k in stats if k[0] == "tiles"} == {2, 3, 4}
    assert {2, 3, 4, 5, 6} <= {k[1] for k in stats if k[0] == "strands"}
    assert stats[("geometry_orders", 2)] >= 42 and stats[("geometry_orders", 1)] >= 42
    assert stats[("repeated", True)] and stats[("crossings_present", False)]
    assert len(permutations_seen) >= 12
    again = generate_puzzle(0)
    assert all(_key(p[r]) == _key(q[r]) for s in reference for p, q in zip(reference[s], again[s])
               for r in ("input", "output"))
    joint = next(_decode(p["input"]) for p in reference["train"]
                 if len(_decode(p["input"])["geometric_orders"]) == 2)
    # With all source clues removed, both geometric orders fit the receiving
    # bank. An input-only solver must reject the ambiguity rather than guess.
    blanked = [t.copy() for t in joint["tiles"]]
    for site in joint["sites"][0]:
        blanked[joint["order"][0]][site] = joint["background"]
    if joint["vertical"]:
        blanked = [t.T.copy() for t in blanked]
    ambiguous = _layout(blanked, list(range(len(blanked))), joint["layout"], joint["divider"])
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("missing order evidence accepted")
    # Recolor one true crossing as ordinary wire: treating a junction as an
    # unmarked crossing must fail even though all endpoint labels are intact.
    broken = [t.copy() for t in joint["tiles"]]
    body_index = joint["order"][1]
    p = tuple(np.argwhere(broken[body_index] == joint["cross"])[0])
    broken[body_index][p] = joint["carrier"]
    if joint["vertical"]:
        broken = [t.T.copy() for t in broken]
    invalid = _layout(broken, list(range(len(broken))), joint["layout"], joint["divider"])
    try:
        solve(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("unmarked branching crossing accepted")
    # Every legal explicit count combination; retain only one puzzle at a time.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(713, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in reference[s]] for s in reference}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("output mutation accepted")
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), tile_shapes=len(tile_shapes),
                routing_masks=len(structures), role_assignments=len(roles),
                endpoint_permutations=len(permutations_seen),
                joint_assemblies=stats[("geometry_orders", 2)])
