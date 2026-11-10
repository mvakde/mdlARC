"""Beam-constrained mosaics: finite rectangular inventory and spatial ray probes.

No files, generation metadata, or demonstrations are used by solve().
The sampler's labels are painted from its rectangles, then checked by a search
that reconstructs both the beam paths and the inventory from the input.
"""
from collections import Counter
import hashlib
import math
import numpy as np


_PORTS = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
               if r in (-1, 2) or c in (-1, 2))
_KINDS = ("diagonal", "perpendicular", "staircase")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
          "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _direction(offset):
    return tuple(-1 if x == -1 else 1 if x == 2 else 0 for x in offset)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        component, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    component.append(p)
                    stack.append(p)
        result.append(component)
    return result


def _unique_mode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    winners = np.flatnonzero(counts == counts.max())
    _need(len(winners) == 1, "background frequency tie")
    return int(winners[0])


def _launchers(board, background, vacancy, structure):
    launchers, accounted = [], set()
    footprint = np.zeros(board.shape, dtype=bool)
    for r in range(1, board.shape[0]-2):
        for c in range(1, board.shape[1]-2):
            color = int(board[r, c])
            if color in (background, vacancy, structure):
                continue
            if not np.all(board[r:r+2, c:c+2] == color):
                continue
            ring = [(r+dr, c+dc) for dr, dc in _PORTS]
            if any(int(board[p]) not in (background, structure) for p in ring):
                continue
            ports = [(p, _direction(o)) for p, o in zip(ring, _PORTS)
                     if board[p] == background]
            if not 1 <= len(ports) <= 4 or sum(board[p] == structure for p in ring) < 8:
                continue
            _need(not footprint[r-1:r+3, c-1:c+3].any(), "overlapping frames")
            footprint[r-1:r+3, c-1:c+3] = True
            launchers.append(dict(core=(r, c), color=color, ports=ports))
            accounted.update(p for p in ring if board[p] == structure)
            accounted.update((y, x) for y in range(r, r+2) for x in range(c, c+2))
    foreground = set(map(tuple, np.argwhere((board != background) & (board != vacancy))))
    _need(foreground == accounted and 2 <= len(launchers) <= 4,
          "unaccounted objects or invalid launcher count")
    return launchers, footprint


def _walk(start, heading, shape):
    scale = max(map(abs, heading))
    points = []
    for k in range(3*max(shape)+1):
        p = tuple(start[i] + (1 if heading[i] >= 0 else -1)
                  * (k*abs(heading[i])//scale) for i in (0, 1))
        if not _inside(p, shape):
            break
        points.append(p)
    return points


def _contact(paths_a, paths_b):
    hits = []
    for ia, a in enumerate(paths_a):
        cells = {p: k for k, p in enumerate(a)}
        mids = {(p[0]+q[0], p[1]+q[1]): k
                for k, (p, q) in enumerate(zip(a, a[1:]))
                if p[0] != q[0] and p[1] != q[1]}
        for ib, b in enumerate(paths_b):
            for j, p in enumerate(b):
                if p in cells:
                    hits.append((cells[p]-1, j-1, ia, ib, "cell"))
            for j, (p, q) in enumerate(zip(b, b[1:])):
                if p[0] == q[0] or p[1] == q[1]:
                    continue
                mid = (p[0]+q[0], p[1]+q[1])
                if mid in mids:
                    i = mids[mid]
                    if {a[i], a[i+1]} != {p, q}:
                        hits.append((i, j, ia, ib, "between"))
    return min(hits, key=lambda h: (sum(h[:2]), max(h[:2]), h[2:4])) if hits else None


def _beams(board, background, vacancy, launchers):
    """Spatial encounters, independent of travel times; no simulation clock."""
    groups, colors = [], []
    for launcher in launchers:
        for p, v in launcher["ports"]:
            strand = len(colors)
            colors.append(launcher["color"])
            groups.append(dict(v=v, members=[(strand, p)]))
    probes = np.full(board.shape, -1, dtype=np.int8)
    owners, segments, events = {}, set(), []

    def draw(strand, path):
        for k, p in enumerate(path):
            _need(board[p] in (background, vacancy), "beam hits a frame")
            _need(p not in owners or (k == 0 and owners[p] == strand), "beam overlap")
            owners[p] = strand
            probes[p] = colors[strand]
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1]:
                mid = (p[0]+q[0], p[1]+q[1])
                sign = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                _need((mid, -sign) not in segments, "unresolved crossing")
                segments.add((mid, sign))

    for _ in range(len(colors)+1):
        paths = [[_walk(p, g["v"], board.shape) for _, p in g["members"]] for g in groups]
        contacts = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hit = _contact(paths[i], paths[j])
                if hit is not None:
                    contacts.append((i, j, hit))
        if not contacts:
            for g, forecasts in zip(groups, paths):
                for (strand, _), path in zip(g["members"], forecasts):
                    draw(strand, path)
            return probes, events
        first = {}
        for i, j, hit in contacts:
            first[i] = min(first.get(i, 1000), hit[0])
            first[j] = min(first.get(j, 1000), hit[1])
        ready = [e for e in contacts if e[2][0] == first[e[0]] and e[2][1] == first[e[1]]]
        _need(bool(ready), "cyclic encounters")
        i, j, hit = min(ready, key=lambda e: (max(e[2][:2]), sum(e[2][:2]), e[:2]))
        for k, l, other in contacts:
            if (k, l) != (i, j):
                _need(not ((k in (i, j) and other[0] == first[k]) or
                           (l in (i, j) and other[1] == first[l])), "triple encounter")
        u, v = groups[i]["v"], groups[j]["v"]
        _need(max(map(abs, u)) == max(map(abs, v)) == 1, "staircase re-encounter")
        _need(u != v and tuple(-x for x in u) != v, "parallel or cancelling encounter")
        norms = sum(map(abs, u)), sum(map(abs, v))
        kind = "perpendicular" if norms == (1, 1) else "diagonal" if norms == (2, 2) else "staircase"
        if kind == "staircase":
            _need(sum(x*y for x, y in zip(u, v)) > 0, "obtuse encounter")
        total = tuple(x+y for x, y in zip(u, v))
        divisor = math.gcd(*map(abs, total))
        heading = tuple(x//divisor for x in total)
        _need(min(hit[:2]) >= 0, "encounter at a port or established path")
        members, outgoing = [], []
        for index, cut in ((i, hit[0]), (j, hit[1])):
            for (strand, _), path in zip(groups[index]["members"], paths[index]):
                _need(cut < len(path)-1, "partly clipped bundle")
                draw(strand, path[:cut+1])
                members.append((strand, path[cut]))
                outgoing.extend(_walk(path[cut], heading, board.shape)[1:])
        events.append(dict(kind=kind, intersection=hit[4], cuts=hit[:2],
                           heading=heading, outgoing=tuple(outgoing),
                           colors=tuple(colors[strand] for strand, _ in members)))
        groups = [g for k, g in enumerate(groups) if k not in (i, j)]
        groups.append(dict(v=heading, members=members))
    raise ValueError("too many beam events")


def _extract(tray, vacancy):
    tiles = []
    for comp in _components(tray != vacancy):
        rr, cc = zip(*comp)
        tile = tray[min(rr):max(rr)+1, min(cc):max(cc)+1]
        _need(tile.size == len(comp) and np.all(tile != vacancy), "tray tile is not a filled rectangle")
        tiles.append(tile.copy())
    _need(3 <= len(tiles) <= 12, "invalid tile count")
    return tiles


def _packing(mask, tiles, probes, max_nodes=None):
    """Enumerate at most two *rendered* completions with finite tile counts."""
    cells = list(map(tuple, np.argwhere(mask)))
    lookup = {p: i for i, p in enumerate(cells)}
    groups, group_index = [], {}
    for tile in tiles:
        key = _key(tile)
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append([tile, 0])
        groups[group_index[key]][1] += 1
    by_cell = [[] for _ in cells]
    for gi, (tile, _) in enumerate(groups):
        h, w = tile.shape
        for r in range(mask.shape[0]-h+1):
            for c in range(mask.shape[1]-w+1):
                if not mask[r:r+h, c:c+w].all():
                    continue
                required = probes[r:r+h, c:c+w]
                if np.any((required >= 0) & (tile != required)):
                    continue
                indices = tuple(lookup[(r+y, c+x)] for y in range(h) for x in range(w))
                bits = sum(1 << i for i in indices)
                candidate = (gi, bits, indices, r, c)
                for index in indices:
                    by_cell[index].append(candidate)
    remaining = [n for _, n in groups]
    full = (1 << len(cells))-1
    pixels = [-1]*len(cells)
    results, path, nodes = {}, [], 0

    def visit(covered):
        nonlocal nodes
        nodes += 1
        _need(max_nodes is None or nodes <= max_nodes, "exact-cover search budget exhausted")
        if covered == full:
            results.setdefault(tuple(pixels), tuple(path))
            return
        choices = None
        for i in range(len(cells)):
            if covered & (1 << i):
                continue
            options = [p for p in by_cell[i] if remaining[p[0]] and not (p[1] & covered)]
            if not options:
                return
            if choices is None or len(options) < len(choices):
                choices = options
                if len(choices) == 1:
                    break
        for gi, bits, indices, r, c in choices:
            remaining[gi] -= 1
            for index, color in zip(indices, groups[gi][0].flat):
                pixels[index] = int(color)
            path.append((gi, r, c))
            visit(covered | bits)
            path.pop()
            remaining[gi] += 1
            for index in indices:
                pixels[index] = -1
            if len(results) >= 2:
                return
    visit(0)
    return results, groups, nodes


def _analyze(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        for cut in range(1, a.shape[axis]-1):
            line = a[cut, :] if axis == 0 else a[:, cut]
            if not np.all(line == line[0]):
                continue
            structure = int(line[0])
            first, second = (a[:cut], a[cut+1:]) if axis == 0 else (a[:, :cut], a[:, cut+1:])
            for board, tray in ((first, second), (second, first)):
                try:
                    background, vacancy = _unique_mode(board), _unique_mode(tray)
                    _need(len({background, vacancy, structure}) == 3, "roles coincide")
                    _need(not np.isin(tray, [background, structure]).any(), "invalid tray colors")
                    tiles = _extract(tray, vacancy)
                    _need(all(not np.isin(t, [background, vacancy, structure]).any() for t in tiles),
                          "tile contains a reserved color")
                    launchers, footprint = _launchers(board, background, vacancy, structure)
                    mask = board == vacancy
                    _need(mask.any() and not (mask & footprint).any(), "invalid vacancies")
                    probes, events = _beams(board, background, vacancy, launchers)
                    outcomes, groups, nodes = _packing(mask, tiles, probes)
                    _need(len(outcomes) == 1, "packing has no unique rendered answer")
                    rendering, placements = next(iter(outcomes.items()))
                    output = board.copy()
                    output[probes >= 0] = probes[probes >= 0]
                    output[mask] = rendering
                    candidates.append(dict(board=board, tray=tray, background=background,
                                           vacancy=vacancy, structure=structure, mask=mask,
                                           tiles=tiles, launchers=launchers, footprint=footprint,
                                           probes=probes, events=events, output=output,
                                           groups=groups, placements=placements, nodes=nodes,
                                           axis=axis, board_first=board is first))
                except ValueError:
                    continue
    _need(len(candidates) == 1, "input must identify exactly one board and completion")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, probes and rectangular placements from the input alone."""
    return _analyze(input_grid)["output"].copy()


def _add_probe_launcher(rng, board, footprint, probes, events):
    """Occasional third emitter, close enough to share the same mosaic.

    It adds a new colored constraint, not another collision convention.
    """
    heading = events[0]["heading"]
    if max(map(abs, heading)) != 1:
        return board, footprint, probes, events
    offsets = [o for o in _PORTS if _direction(o) == heading]
    main_points = np.argwhere(probes >= 0)
    for _ in range(70):
        y, x = int(rng.integers(1, board.shape[0]-2)), int(rng.integers(1, board.shape[1]-2))
        if footprint[y-1:y+3, x-1:x+3].any():
            continue
        offset = offsets[int(rng.integers(len(offsets)))]
        p = y+offset[0], x+offset[1]
        path = _walk(p, heading, board.shape)
        if len(path) < 6 or not any(np.abs(main_points-q).sum(axis=1).min() <= 3 for q in path[2:]):
            continue
        candidate = board.copy()
        candidate[y-1:y+3, x-1:x+3] = 2
        candidate[y:y+2, x:x+2] = 5
        candidate[p] = 0
        try:
            launchers, fp = _launchers(candidate, 0, 1, 2)
            pp, ee = _beams(candidate, 0, 1, launchers)
            if len(ee) == 1 and set(ee[0]["colors"]) == {3, 4} and ee[0]["kind"] == events[0]["kind"]:
                return candidate, fp, pp, ee
        except ValueError:
            pass
    return board, footprint, probes, events


def _make_board(rng, kind, testing, contact):
    h = int(rng.integers(21 if testing else 17, 30 if testing else 26))
    w = int(rng.integers(16, 21))
    for _ in range(180):
        r, c = int(rng.integers(5, h-6)), int(rng.integers(6, w-5))
        da, db = (int(x) for x in rng.integers(2, 6, 2))
        if da == db:
            continue
        half = kind == "diagonal" and contact == "between"
        if kind == "diagonal":
            points = ((r+da+int(half), c-da), (r+db+int(half), c+db+int(half)))
            offsets = ((-1, 2), (-1, -1))
        elif kind == "perpendicular":
            points = ((r, c-da), (r+db, c))
            offsets = ((int(rng.integers(2)), 2), (-1, int(rng.integers(2))))
        else:
            points = ((r+da, c-da), (r+db, c))
            offsets = ((-1, 2), (-1, int(rng.integers(2))))
        board = np.zeros((h, w), dtype=np.int8)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True
        for color, p, offset in zip((3, 4), points, offsets):
            y, x = p[0]-offset[0], p[1]-offset[1]
            if y < 1 or x < 1 or y+2 >= h or x+2 >= w or occupied[y-1:y+3, x-1:x+3].any():
                ok = False
                break
            occupied[y-1:y+3, x-1:x+3] = True
            board[y-1:y+3, x-1:x+3] = 2
            board[y:y+2, x:x+2] = color
            board[p] = 0
        if not ok:
            continue
        try:
            launchers, footprint = _launchers(board, 0, 1, 2)
            probes, events = _beams(board, 0, 1, launchers)
            if len(events) == 1 and events[0]["kind"] == kind:
                if rng.random() < .4:
                    board, footprint, probes, events = _add_probe_launcher(rng, board, footprint, probes, events)
                return board, footprint, probes, events
        except ValueError:
            pass
    raise ValueError("could not place a beam encounter")


def _rectangles(rng, board, footprint, probes, count):
    points = np.argwhere(probes >= 0)
    mask = np.zeros(board.shape, dtype=bool)
    rectangles = []
    common = tuple(int(x) for x in rng.integers(2, 5, 2))
    for index in range(count):
        if index < 2:
            th, tw = common
        elif rng.random() < .3:
            th, tw = rectangles[int(rng.integers(len(rectangles)))][2:]
        else:
            th, tw = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        found = False
        for _ in range(350):
            p = points[int(rng.integers(len(points)))]
            y, x = int(p[0])-int(rng.integers(th)), int(p[1])-int(rng.integers(tw))
            if y < 0 or x < 0 or y+th > board.shape[0] or x+tw > board.shape[1]:
                continue
            if footprint[y:y+th, x:x+tw].any() or mask[y:y+th, x:x+tw].any():
                continue
            # Every new piece joins the same merged silhouette, by an edge.
            adjacent = (mask[max(0,y-1):y, x:x+tw].any() or
                        mask[y+th:y+th+1, x:x+tw].any() or
                        mask[y:y+th, max(0,x-1):x].any() or
                        mask[y:y+th, x+tw:x+tw+1].any())
            if rectangles and not adjacent:
                continue
            mask[y:y+th, x:x+tw] = True
            rectangles.append((y, x, th, tw))
            found = True
            break
        if not found:
            raise ValueError("could not grow a connected mosaic")
    _need(mask.mean() < .42, "mosaic obscures background role")
    return mask, rectangles


def _scatter(rng, tiles, height, width):
    for _ in range(16):
        tray = np.ones((height, width), dtype=np.int8)
        occupied = np.zeros_like(tray, dtype=bool)
        order = sorted(rng.permutation(len(tiles)), key=lambda i: -tiles[i].size)
        for i in order:
            tile = tiles[i]
            th, tw = tile.shape
            legal = [(y, x) for y in range(1, height-th) for x in range(1, width-tw)
                     if not occupied[y-1:y+th+1, x-1:x+tw+1].any()]
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            tray[y:y+th, x:x+tw] = tile
            occupied[y:y+th, x:x+tw] = True
        else:
            if _unique_mode(tray) == 1:
                return tray
    raise ValueError("tray packing failed")


def _without_bend(probes, events):
    result = probes.copy()
    for event in events:
        for p in event["outgoing"]:
            result[p] = -1
    return result


def _build_pair(rng, kind, testing, surplus, contact):
    for _ in range(220):
        try:
            board, footprint, probes, events = _make_board(rng, kind, testing, contact)
            count = int(rng.integers(6, 9) if testing else rng.integers(3, 6))
            mask, rects = _rectangles(rng, board, footprint, probes, count)
            _need(any(mask[p] for p in events[0]["outgoing"]), "no outgoing mosaic probe")
            launchers, _ = _launchers(board, 0, 1, 2)
            core_colors = sorted({l["color"] for l in launchers})
            _need({int(c) for c in probes[mask] if c >= 0} == set(core_colors),
                  "every emitter must constrain the same mosaic")
            accents = list(range(max(core_colors)+1, int(rng.integers(8, 11))))
            palette = np.array(core_colors + accents, dtype=np.int8)
            probabilities = np.array([.16/len(core_colors)]*len(core_colors) + [.84/len(accents)]*len(accents))
            tiles, completed = [], board.copy()
            completed[probes >= 0] = probes[probes >= 0]
            for y, x, h, w in rects:
                tile = rng.choice(palette, size=(h, w), p=probabilities).astype(np.int8)
                required = probes[y:y+h, x:x+w]
                tile[required >= 0] = required[required >= 0]
                tiles.append(tile)
                completed[y:y+h, x:x+w] = tile
            # Same-size, different matrices deliberately defeat geometry alone.
            _need(not np.array_equal(tiles[0], tiles[1]), "identical probe signatures")
            if surplus:
                for _ in range(int(rng.integers(1, 3))):
                    h, w = tiles[int(rng.integers(len(tiles)))].shape
                    decoy = rng.choice(palette, size=(h, w), p=probabilities).astype(np.int8)
                    tiles.append(decoy)
            outcomes, groups, _ = _packing(mask, tiles, probes, max_nodes=20000)
            expected = tuple(int(x) for x in completed[mask])
            _need(len(outcomes) == 1 and expected in outcomes, "ambiguous probe-constrained mosaic")
            used = outcomes[expected]
            _need(len(used) == count, "unexpected alternative decomposition")
            relaxed, _, _ = _packing(mask, tiles, _without_bend(probes, events), max_nodes=20000)
            _need(len(relaxed) >= 2, "bend does not disambiguate tile placement")
            # Beam-color texture off the paths teaches that these are constraints,
            # not a command to erase every other occurrence of a payload color.
            _need(np.any(mask & (probes < 0) & np.isin(completed, core_colors)), "missing off-path texture witness")
            max_width = 29-board.shape[1]
            minimum = max(max(t.shape[1] for t in tiles)+2, max_width-3)
            tray_width = int(rng.integers(minimum, max_width+1))
            tray = _scatter(rng, tiles, board.shape[0], tray_width)
            board[mask] = 1
            _need(_unique_mode(board) == 0, "board background not identifiable")
            separator = np.full((board.shape[0], 1), 2, dtype=np.int8)
            a = np.concatenate((board, separator, tray), axis=1)
            rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
            a, completed = np.rot90(a, rotation), np.rot90(completed, rotation)
            if reflection:
                a, completed = np.fliplr(a), np.fliplr(completed)
            recolor = rng.permutation(10).astype(np.int8)
            a, completed = recolor[a].copy(), recolor[completed].copy()
            analysis = _analyze(a)
            _need(np.array_equal(analysis["output"], completed), "reconstruction disagrees")
            return {"input": a, "output": completed}
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} is outside its integer range")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    kinds = list(rng.permutation(_KINDS))[:min(num_train, 3)]
    allow_surplus = bool(rng.integers(2))
    modes = [(kind, "between" if kind == "diagonal" and rng.integers(2) else "cell") for kind in kinds]
    if num_train >= 4:
        modes.append(("diagonal", "cell" if dict(modes)["diagonal"] == "between" else "between"))
    training = (modes * ((num_train+len(modes)-1)//len(modes)))[:num_train]
    rng.shuffle(training)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind, contact = training[i] if split == "train" else modes[int(rng.integers(len(modes)))]
            surplus = allow_surplus and ((split == "train" and i == 0) or rng.random() < .5)
            for _ in range(5):
                pair = _build_pair(rng, kind, split == "test", surplus, contact)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    return puzzle


def _features(a):
    used = [a["groups"][gi][0] for gi, _, _ in a["placements"]]
    shapes = [t.shape for t in used]
    corners = sum(1 for r in range(a["mask"].shape[0]-1) for c in range(a["mask"].shape[1]-1)
                  if int(a["mask"][r:r+2, c:c+2].sum()) == 3)
    return dict(shape=a["board"].shape, side=(a["axis"], a["board_first"]),
                kind=a["events"][0]["kind"], contact=a["events"][0]["intersection"],
                used=len(used), supplied=len(a["tiles"]),
                dimensions=tuple(sorted(shapes)), corners=corners,
                probe_cells=int((a["mask"] & (a["probes"] >= 0)).sum()),
                texture_cells=int((a["mask"] & (a["probes"] < 0)).sum()),
                repeated_shapes=any(n > 1 for n in Counter(shapes).values()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(a["output"], out)
            mask, probes, board = a["mask"], a["probes"], a["board"]
            assert len(_components(mask)) == 1
            assert np.array_equal(out[(~mask) & (probes < 0)], board[(~mask) & (probes < 0)])
            assert np.array_equal(out[probes >= 0], probes[probes >= 0])
            assert not np.any(out == a["vacancy"])
            assert np.any(mask & (probes < 0) & np.isin(out, [l["color"] for l in a["launchers"]]))
            assert len(a["events"]) == 1
            assert 2 <= len(a["launchers"]) <= 3 and all(len(l["ports"]) == 1 for l in a["launchers"])
            assert len(a["events"][0]["colors"]) == len(set(a["events"][0]["colors"])) == 2
            assert a["events"][0]["cuts"][0] != a["events"][0]["cuts"][1]
            assert any(mask[p] for p in a["events"][0]["outgoing"])
            assert set(int(c) for c in probes[mask] if c >= 0) == {l["color"] for l in a["launchers"]}
            # Omitting B leaves at least two different valid A-only completions.
            unprobed, _, _ = _packing(mask, a["tiles"], np.full(board.shape, -1, dtype=np.int8))
            assert len(unprobed) >= 2
            relaxed, _, _ = _packing(mask, a["tiles"], _without_bend(probes, a["events"]))
            assert len(relaxed) >= 2
            # Omitting A leaves genuinely unpainted cells, not just a ray picture.
            assert np.any(mask & (probes < 0))
            assert sum(t.size for t in a["tiles"]) >= int(mask.sum())
            used_counts = Counter(gi for gi, _, _ in a["placements"])
            assert all(n <= a["groups"][gi][1] for gi, n in used_counts.items())
            features[split].append(_features(a))
    assert {(f["kind"], f["contact"]) for f in features["test"]} <= {
        (f["kind"], f["contact"]) for f in features["train"]}
    assert min(f["used"] for f in features["test"]) > max(f["used"] for f in features["train"])
    assert not any(f["supplied"] > f["used"] for f in features["test"]) or any(
        f["supplied"] > f["used"] for f in features["train"])


def self_test():
    """42 streamed seeds, small analytic fixtures, corruption and count checks."""
    signatures, shapes, dimensions, sides, kinds, contacts, counts = (set() for _ in range(7))
    corner_counts, emitter_counts, surplus_counts, tile_sizes = (set() for _ in range(4))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                signature.update(pair["input"].tobytes())
                a = _analyze(pair["input"])
                f = _features(a)
                shapes.add(f["shape"]); dimensions.add(f["dimensions"])
                sides.add(f["side"]); kinds.add(f["kind"])
                contacts.add(f["contact"]); counts.add(f["used"])
                corner_counts.add(f["corners"])
                emitter_counts.add(len(a["launchers"]))
                surplus_counts.add(f["supplied"]-f["used"])
                tile_sizes.update(f["dimensions"])
        signatures.add(signature.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 20 and len(dimensions) >= 70
    assert len(sides) == 4 and kinds == set(_KINDS) and contacts == {"cell", "between"}
    assert {3, 4, 5, 6, 7, 8} <= counts and len(corner_counts) >= 4
    assert emitter_counts == {2, 3} and {0, 1, 2} <= surplus_counts and len(tile_sizes) >= 20
    # Full headings, major-axis-first phase, unequal travel distances.
    assert _walk((5, 2), (-2, 1), (10, 10))[:5] == [(5, 2), (4, 2), (3, 3), (2, 3), (1, 4)]
    assert _contact([[(3, 0), (2, 1), (1, 2), (0, 3)]],
                    [[(2, 3), (1, 2), (0, 1)]])[:2] == (1, 0)
    hit = _contact([[(3, 0), (2, 1), (1, 2), (0, 3)]],
                   [[(4, 4), (3, 3), (2, 2), (1, 1), (0, 0)]])
    assert hit[:2] == (1, 2) and hit[4] == "between"
    # A complete, hand-written unequal-distance staircase fixture, including
    # ports and the exact first outgoing pixels. No generated labels are used.
    board = np.zeros((15, 18), dtype=np.int8)
    for r, c, color, port in ((10, 3, 3, (9, 5)), (9, 10, 4, (8, 10))):
        board[r-1:r+3, c-1:c+3] = 2
        board[r:r+2, c:c+2] = color
        board[port] = 0
    launchers, _ = _launchers(board, 0, 1, 2)
    actual, events = _beams(board, 0, 1, launchers)
    expected = np.full(board.shape, -1, dtype=np.int8)
    for p in ((9,5),(8,6),(7,7),(6,8),(5,9),(4,9),(3,10),(2,10),(1,11),(0,11)):
        expected[p] = 3
    for p in ((8,10),(7,10),(6,10),(5,10),(4,10),(3,11),(2,11),(1,12),(0,12)):
        expected[p] = 4
    assert np.array_equal(actual, expected) and events[0]["heading"] == (-2, 1)
    mask = np.ones((1, 4), dtype=bool)
    probes = np.full(mask.shape, -1, dtype=np.int8)
    tile = np.array([[3, 4]], dtype=np.int8)
    assert not _packing(mask, [tile], probes)[0]
    assert len(_packing(mask, [tile, tile], probes)[0]) == 1
    assert len(_packing(mask, [tile, np.array([[5, 6]], dtype=np.int8)], probes)[0]) == 2
    for nt, ne in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(117, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(5, 2, 1), generate_puzzle(5, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("corrupted label accepted")
