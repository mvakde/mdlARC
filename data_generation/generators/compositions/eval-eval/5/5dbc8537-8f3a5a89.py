"""Connectivity-locked masonry: finite rectangle packing constrained by a flood.

The public solver reads only the query array.  All roles are fixed: 0 floor,
1 separator, 2 seed/rim, 3 checkpoint, 4..8 masonry, and 9 missing/tray.
No parent generator is imported or called.
"""

from collections import Counter
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 620417855, 570828847, 570182424, 191693345, 993959710,
              905879267, 394261712, 963673246, 156586305, 950693427,
              346880323, 543187749, 834202881, 248400865, 892953210,
              15739236, 278983667, 537748114, 465230471, 499460206,
              94732532, 728310084, 830254525, 84152085, 675751043,
              831364120, 478296912, 251868288, 598592251, 978270398,
              624594528, 606489370, 197127072, 158055852, 343982730,
              589414147, 107901996, 967799517, 950542599, 114958489,
              842744393)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    stack = [start]
    h, w = allowed.shape
    while stack:
        y, x = stack.pop()
        for dy, dx in N4:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and allowed[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                stack.append((yy, xx))
    return reached


def _components(mask):
    rest = mask.copy()
    result = []
    while np.any(rest):
        start = tuple(int(v) for v in np.argwhere(rest)[0])
        component = _flood(rest, start)
        result.append(component)
        rest[component] = False
    return result


def _dilate(mask, diagonal=False):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = np.zeros_like(mask)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)] if diagonal else N4
    for dy, dx in offsets:
        result |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return result


def _read_input(grid):
    a = _grid(grid)
    cuts = []
    for axis in (0, 1):
        lines = np.all(a == 1, axis=1 - axis)
        for cut in np.flatnonzero(lines):
            if cut in (0, a.shape[axis] - 1):
                continue
            panels = (a[:cut], a[cut + 1:]) if axis == 0 else (a[:, :cut], a[:, cut + 1:])
            for board, tray in (panels, panels[::-1]):
                if np.count_nonzero(board == 2) != 1 or not np.any(board == 3):
                    continue
                if np.any((tray == 1) | (tray == 2) | (tray == 3)) or np.any(board == 1):
                    continue
                if not np.any(board == 9) or not np.any(tray == 9):
                    continue
                tiles = []
                valid = True
                for component in _components(tray != 9):
                    yy, xx = np.where(component)
                    y0, y1, x0, x1 = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
                    tile = tray[y0:y1, x0:x1]
                    if np.any(tile == 9) or tile.size != int(component.sum()):
                        valid = False
                        break
                    tiles.append(tile.copy())
                if valid and tiles:
                    cuts.append((board.copy(), tray.copy(), tiles, axis))
    if len(cuts) != 1:
        raise ValueError("Input needs exactly one separator, map, and rectangular inventory")
    return cuts[0]


def _pack(board, tiles, connectivity=True, max_nodes=None):
    """Enumerate up to two distinct completed maps, quotienting identical copies.

    Search uses rectangle footprints AND optimistic checkpoint reachability.
    Unfilled cells are provisionally floor for pruning; they become concrete
    floor/masonry on placement. No source coordinates enter the search.
    """
    holes = board == 9
    coords = np.argwhere(holes)
    n = len(coords)
    index = np.full(board.shape, -1, int)
    index[holes] = np.arange(n)
    groups, lookup = [], {}
    for tile in tiles:
        key = _key(tile)
        if key in lookup:
            groups[lookup[key]][1] += 1
        else:
            lookup[key] = len(groups)
            groups.append([tile, 1])
    candidates, by_cell = [], [[] for _ in range(n)]
    h, w = board.shape
    for g, (tile, _) in enumerate(groups):
        th, tw = tile.shape
        for y in range(h - th + 1):
            for x in range(w - tw + 1):
                cells = index[y:y + th, x:x + tw]
                if np.any(cells < 0):
                    continue
                bits = sum(1 << int(c) for c in cells.flat)
                k = len(candidates)
                candidates.append((g, y, x, bits))
                for c in cells.flat:
                    by_cell[int(c)].append(k)
    seed = tuple(int(v) for v in np.argwhere(board == 2)[0])
    goals = board == 3
    remaining = [count for _, count in groups]
    working = board.copy()
    full = (1 << n) - 1
    answers, path = {}, []
    stats = dict(nodes=0, rejected_links=0, dead_ends=0, placements=0)

    def reachable():
        allowed = (working == 0) | (working == 2) | (working == 3) | (working == 9)
        return bool(np.all(_flood(allowed, seed)[goals]))

    def visit(covered):
        if len(answers) >= 2:
            return
        stats['nodes'] += 1
        if max_nodes is not None and stats['nodes'] > max_nodes:
            raise ValueError("Exact-cover search budget exceeded; uniqueness unresolved")
        if connectivity and not reachable():
            stats['rejected_links'] += 1
            return
        if covered == full:
            key = working.tobytes()
            answers.setdefault(key, (working.copy(), tuple(path)))
            return
        options = None
        for c in range(n):
            if covered & (1 << c):
                continue
            choices = [k for k in by_cell[c] if remaining[candidates[k][0]]
                       and not (covered & candidates[k][3])]
            if not choices:
                stats['dead_ends'] += 1
                return
            if options is None or len(choices) < len(options):
                options = choices
                if len(options) == 1:
                    break
        for k in options:
            g, y, x, bits = candidates[k]
            tile = groups[g][0]
            th, tw = tile.shape
            remaining[g] -= 1
            working[y:y + th, x:x + tw] = tile
            path.append((g, y, x))
            visit(covered | bits)
            path.pop()
            working[y:y + th, x:x + tw] = 9
            remaining[g] += 1
            if len(answers) >= 2:
                break

    visit(0)
    stats['placements'] = len(candidates)
    return list(answers.values()), groups, stats


def _finish(completed):
    seed = tuple(int(v) for v in np.argwhere(completed == 2)[0])
    selected = _flood(np.isin(completed, (0, 2, 3)), seed)
    complement = np.pad(~selected, 1, constant_values=True)
    outside = _flood(complement, (0, 0))
    rim = _dilate(outside, True)[1:-1, 1:-1] & selected
    orth = _dilate(outside)[1:-1, 1:-1] & selected
    wall = (completed >= 4) & (completed <= 8)
    touching = _dilate(selected) & wall
    keep = np.zeros_like(wall)
    for component in _components(wall):
        if np.any(component & touching):
            keep |= component
    output = completed.copy()
    output[wall & ~keep] = 0
    output[rim & (completed == 0)] = 2
    return output, dict(selected=selected, outside=outside[1:-1, 1:-1], rim=rim,
                        diagonal=rim & ~orth, wall=wall, keep=keep)


def _analyze(input_grid):
    board, tray, tiles, axis = _read_input(input_grid)
    answers, groups, stats = _pack(board, tiles)
    if len(answers) != 1:
        raise ValueError("Need one checkpoint-connected completed map")
    completed, placements = answers[0]
    output, regions = _finish(completed)
    return dict(board=board, tray=tray, tiles=tiles, axis=axis,
                completed=completed, placements=placements, groups=groups,
                output=output, stats=stats, **regions)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely connected reconstruction's exterior-rim map."""
    return _analyze(input_grid)['output'].copy()


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _neighbors(cell, nr, nc):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in N4
            if 0 <= r + dr < nr and 0 <= c + dc < nc]


def _connected(cells, nr, nc):
    if not cells:
        return False
    reached = {next(iter(sorted(cells)))}
    stack = list(reached)
    while stack:
        for q in _neighbors(stack.pop(), nr, nc):
            if q in cells and q not in reached:
                reached.add(q)
                stack.append(q)
    return reached == cells


def _partition(rng, nr, nc, count):
    all_cells = {(r, c) for r in range(nr) for c in range(nc)}
    active = {tuple(int(x) for x in (rng.integers(nr), rng.integers(nc)))}
    while len(active) < count:
        choices = sorted({q for p in active for q in _neighbors(p, nr, nc)} - active)
        rng.shuffle(choices)
        for q in choices:
            if _connected(all_cells - active - {q}, nr, nc):
                active.add(q)
                break
        else:
            return None
    return active, all_cells - active


def _tree(rng, cells, nr, nc):
    root = sorted(cells)[int(rng.integers(len(cells)))]
    seen, stack, edges = {root}, [root], []
    while stack:
        p = stack[-1]
        choices = [q for q in _neighbors(p, nr, nc) if q in cells and q not in seen]
        if not choices:
            stack.pop()
            continue
        q = choices[int(rng.integers(len(choices)))]
        edges.append((p, q))
        seen.add(q)
        stack.append(q)
    return edges


def _tray(rng, tiles, height, max_width):
    """Scatter components independently, with edge-or-corner separating gaps."""
    if max_width < max(t.shape[1] for t in tiles):
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for attempt in range(16):
        width = max_width if attempt < 8 else int(rng.integers(max(t.shape[1] for t in tiles), max_width + 1))
        occupied = np.zeros((height, width), bool)
        positions = {}
        for i in order:
            th, tw = tiles[i].shape
            legal = []
            for y in range(height - th + 1):
                for x in range(width - tw + 1):
                    if not np.any(occupied[max(0, y - 1):min(height, y + th + 1),
                                           max(0, x - 1):min(width, x + tw + 1)]):
                        legal.append((y, x))
            if not legal:
                break
            y, x = legal[int(rng.integers(len(legal)))]
            positions[i] = (y, x)
            occupied[y:y + th, x:x + tw] = True
        if len(positions) == len(tiles):
            tray = np.full((height, width), 9, np.int8)
            for i, (y, x) in positions.items():
                th, tw = tiles[i].shape
                tray[y:y + th, x:x + tw] = tiles[i]
            # Keep independent scatter coordinates, but vary unused outer margins.
            columns = np.flatnonzero(np.any(tray != 9, axis=0))
            left = max(0, int(columns[0]) - int(rng.integers(3)))
            right = min(width, int(columns[-1]) + 1 + int(rng.integers(3)))
            return tray[:, left:right].copy()
    return None


def _scene(rng, test, witness, fallback=False):
    nr, nc = (4 if test else 3), 3
    if fallback:
        heights, widths = [7] * nr, [5] * nc
        snake = [(r, c) for r in range(nr) for c in (range(nc) if r % 2 == 0 else range(nc - 1, -1, -1))]
        size = 9 if test else (4 if witness == 1 else 6)
        # Four-room witness has a turn on both sides of the two adjoining holes.
        if witness == 1:
            snake = [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
        active, inactive = set(snake[:size]), set(snake[size:])
        aedges = list(zip(snake[:size - 1], snake[1:size]))
        bedges = list(zip(snake[size:-1], snake[size + 1:]))
        desired = 4 if test else (2 if witness == 1 else 3)
        missing = set(snake[1:1 + desired])
        palette = [int(rng.integers(4, 9))]
    else:
        # Unequal row/column extents vary the room and rectangle geometry.
        heights = [int(rng.integers(5, 8)) for _ in range(nr)]
        widths = [int(rng.integers(4, 7)) for _ in range(nc)]
        if witness:
            heights = [7] * nr
            widths = [5] * nc
        size = (int(rng.integers(7, 11)) if test else
                4 if witness == 1 else 6 if witness == 2 else int(rng.integers(5, 8)))
        partition = _partition(rng, nr, nc, size)
        if partition is None:
            return None
        active, inactive = partition
        aedges, bedges = _tree(rng, active, nr, nc), _tree(rng, inactive, nr, nc)
        degree = Counter(p for edge in aedges for p in edge)
        choices = sorted(p for p in active if degree[p] >= 2)
        if test and len(choices) < 4:
            return None
        desired = int(rng.integers(4, min(6, len(choices)) + 1)) if test else 2 if witness == 1 else 3
        if len(choices) < desired:
            return None
        rng.shuffle(choices)
        missing = set(choices[:desired])
        if witness == 1 and not any(q in missing for p in missing for q in _neighbors(p, nr, nc)):
            return None
        palette = [int(v) for v in rng.choice(np.arange(4, 9), int(rng.integers(1, 5)), replace=False)]
        # Sometimes leave several excluded compartments, rather than just one.
        if not witness and len(bedges) > 1 and rng.random() < .45:
            rng.shuffle(bedges)
            bedges = bedges[int(rng.integers(1, min(3, len(bedges)) + 1)):]
        # Extra edges give selected networks cycles as well as branches and paths.
        if not witness and rng.random() < .35:
            unused = [(p, q) for p in sorted(active) for q in _neighbors(p, nr, nc)
                      if q in active and p < q and (p, q) not in aedges and (q, p) not in aedges]
            rng.shuffle(unused)
            aedges.extend(unused[:int(rng.integers(1, 3))])
    ys, xs = np.cumsum([0] + heights), np.cumsum([0] + widths)
    h, w = int(ys[-1]), int(xs[-1])
    completed = rng.choice(palette, size=(h, w)).astype(np.int8)
    boxes = {}
    for r in range(nr):
        for c in range(nc):
            y, x, th, tw = int(ys[r]), int(xs[c]), heights[r], widths[c]
            boxes[r, c] = (y, x, th, tw)
            style = 0 if fallback else int(rng.integers(5))
            local_y, local_x = np.indices((th, tw))
            if style == 0:
                completed[y:y + th, x:x + tw] = int(rng.choice(palette))
            elif style in (1, 2, 3):
                local_index = local_x if style == 1 else local_y if style == 2 else local_x + local_y
                colors = np.array(palette, np.int8)
                completed[y:y + th, x:x + tw] = colors[local_index % len(colors)]
            completed[y + 1:y + th - 1, x + 1:x + tw - 1] = 0
    centered_ports = fallback or rng.random() < .25
    for p, q in aedges + bedges:
        r, c = p
        rr, cc = q
        if r == rr:
            cut = int(xs[max(c, cc)])
            offset = heights[r] // 2 if centered_ports else int(rng.integers(1, heights[r] - 1))
            y = int(ys[r]) + offset
            completed[y, cut - 1:cut + 1] = 0
        else:
            cut = int(ys[max(r, rr)])
            offset = widths[c] // 2 if centered_ports else int(rng.integers(1, widths[c] - 1))
            x = int(xs[c]) + offset
            completed[cut - 1:cut + 1, x] = 0
    # A floor opening may end against a neighbor's wall; lining up every opening
    # is deliberately not an extra rule. Only global checkpoint reachability is.
    if not fallback and (witness == 2 or rng.random() < .5):
        candidates = []
        for p in sorted(missing):
            y, x, th, tw = boxes[p]
            for yy, xx, dy, dx in ((y, x + tw // 2, -1, 0),
                                  (y + th - 1, x + tw // 2, 1, 0),
                                  (y + th // 2, x, 0, -1),
                                  (y + th // 2, x + tw - 1, 0, 1)):
                ny, nx = yy + dy, xx + dx
                if (0 <= ny < h and 0 <= nx < w and completed[yy, xx] >= 4
                        and completed[ny, nx] >= 4):
                    candidates.append((yy, xx))
        if candidates:
            completed[candidates[int(rng.integers(len(candidates)))]] = 0
    # Islands are walls surrounded by floor, not extra outline targets.
    selected_islands, erased_islands = [], []
    for cells, store in ((active, selected_islands), (inactive, erased_islands)):
        roomy = sorted(p for p in cells if heights[p[0]] >= 7 and widths[p[1]] >= 5
                       and (not fallback or p not in missing))
        if witness and not roomy:
            return None
        if roomy and (witness or fallback or rng.random() < .75):
            rng.shuffle(roomy)
            for p in roomy[:int(rng.integers(1, min(3, len(roomy)) + 1))]:
                y, x, th, tw = boxes[p]
                cell = (y + th // 2, x + tw // 2)
                completed[cell] = int(rng.choice(palette))
                store.append(cell)
    fixed = sorted(active - missing)
    if len(fixed) < 2:
        return None
    rng.shuffle(fixed)
    seed_room = fixed[0]
    for p in fixed[1:]:
        y, x, th, tw = boxes[p]
        candidates = np.argwhere(completed[y + 1:y + th - 1, x + 1:x + tw - 1] == 0)
        local = candidates[len(candidates) // 2]
        completed[y + 1 + int(local[0]), x + 1 + int(local[1])] = 3
    y, x, th, tw = boxes[seed_room]
    floor = np.argwhere(completed[y:y + th, x:x + tw] == 0)
    # Choose a visibly interior marker in witness one; a rim marker in witness two.
    trial_seed = (y + int(floor[0, 0]), x + int(floor[0, 1]))
    provisional = completed.copy()
    provisional[trial_seed] = 2
    _, info = _finish(provisional)
    candidates = [(y + int(dy), x + int(dx)) for dy, dx in floor
                  if witness != 1 or not info['rim'][y + int(dy), x + int(dx)]]
    if witness == 2:
        candidates = [p for p in candidates if info['rim'][p]]
    if not candidates:
        return None
    seed = candidates[int(rng.integers(len(candidates)))]
    completed[seed] = 2
    board, tiles = completed.copy(), []
    for p in sorted(missing):
        y, x, th, tw = boxes[p]
        tiles.append(completed[y:y + th, x:x + tw].copy())
        board[y:y + th, x:x + tw] = 9
    # Repeated border strips expose finite-copy use inside a merged vacancy.
    # The strips keep the visible material pattern; no pattern is recolored here.
    if witness == 2 or (not fallback and witness != 1 and rng.random() < .45):
        repeatable = [i for i, tile in enumerate(tiles)
                      if np.all(tile[0] >= 4) and np.array_equal(tile[0], tile[-1])]
        if witness == 2 and not repeatable:
            return None
        if repeatable:
            i = repeatable[int(rng.integers(len(repeatable)))]
            tile = tiles.pop(i)
            tiles.extend([tile[:1].copy(), tile[1:-1].copy(), tile[-1:].copy()])
    # A geometrically plausible but sealed fragment teaches optional surplus.
    if witness == 1 or (not fallback and rng.random() < .35):
        th, tw = tiles[int(rng.integers(len(tiles)))].shape
        decoy = np.full((th, tw), int(rng.choice(palette)), np.int8)
        decoy[1:-1, 1:-1] = 0
        tiles.append(decoy)
    # A cut can erase piece boundaries inside a broad vacancy, beyond room-size matching.
    if not witness and not fallback and rng.random() < .35 and len(tiles) < 6:
        i = int(rng.integers(len(tiles)))
        tile = tiles.pop(i)
        axis = int(rng.integers(2))
        cut = int(rng.integers(1, tile.shape[axis]))
        tiles.extend([t.copy() for t in np.split(tile, [cut], axis=axis)])
    if not np.all(info['selected'][completed == 3]):
        return None
    answers, _, _ = _pack(board, tiles, max_nodes=16000)
    if len(answers) != 1 or not np.array_equal(answers[0][0], completed):
        return None
    geometric, _, _ = _pack(board, tiles, connectivity=False, max_nodes=8000)
    if len(geometric) < 2:
        return None
    output, info = _finish(completed)
    if witness:
        regions = _components(np.isin(completed, (0, 2, 3)))
        sizes = [int(r.sum()) for r in regions]
        area = int(info['selected'].sum())
        if witness == 1 and area >= max(sizes):
            return None
        if witness == 2 and area != max(sizes):
            return None
        if not selected_islands or not erased_islands or not np.any(info['diagonal'] & (completed == 0)):
            return None
    tray = _tray(rng, tiles, h, 29 - w)
    if tray is None:
        return None
    inp = np.concatenate((board, np.ones((h, 1), np.int8), tray), axis=1)
    return {'input': inp, 'output': output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    rotation = int(rng.integers(4))
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            witness = i + 1 if split == 'train' and i < 2 else 0
            for attempt in range(140):
                try:
                    pair = _scene(rng, split == 'test', witness, fallback=attempt >= 100)
                except ValueError:
                    continue
                if pair is None:
                    continue
                rot = (rotation + int(rng.integers(4))) % 4
                reflect = bool(rng.integers(2))
                for role in ('input', 'output'):
                    grid = np.rot90(pair[role], rot)
                    pair[role] = (np.fliplr(grid) if reflect else grid).copy()
                if witness == 1:
                    # Check reading order after the whole-scene isometry, since
                    # it changes both the first vacancy and tray scan order.
                    board, _, visible_tiles, _ = _read_input(pair['input'])
                    ordered, _, _ = _pack(board, visible_tiles, connectivity=False, max_nodes=8000)
                    if np.array_equal(_finish(ordered[0][0])[0], pair['output']):
                        continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('Bounded construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _features(a):
    completed, selected = a['completed'], a['selected']
    seed = tuple(int(v) for v in np.argwhere(completed == 2)[0])
    sizes = [int(c.sum()) for c in _components(np.isin(completed, (0, 2, 3)))]
    holes = a['board'] == 9
    islands = a['wall'] & ~a['outside']
    ignored_neighbors = _dilate(islands, True) & selected & ~a['rim'] & (completed == 0)
    used = Counter(g for g, _, _ in a['placements'])
    blocked_ports = 0
    h, w = completed.shape
    for g, y, x in a['placements']:
        tile = a['groups'][g][0]
        th, tw = tile.shape
        for yy, xx in np.argwhere(tile == 0):
            yy, xx = int(yy), int(xx)
            for dy, dx in N4:
                ly, lx = yy + dy, xx + dx
                ny, nx = y + ly, x + lx
                if (not (0 <= ly < th and 0 <= lx < tw) and 0 <= ny < h and 0 <= nx < w
                        and 4 <= completed[ny, nx] <= 8 and selected[y + yy, x + xx]):
                    blocked_ports += 1
    return dict(shape=a['board'].shape, axis=a['axis'], tiles=len(a['tiles']),
                used=len(a['placements']), surplus=len(a['tiles']) - len(a['placements']),
                pieces=tuple(sorted(t.shape for t in a['tiles'])),
                merged=max(int(c.sum()) for c in _components(holes)),
                vacancy_components=len(_components(holes)),
                compartments=len(sizes), selected_area=int(selected.sum()),
                small=int(selected.sum()) < max(sizes),
                large=int(selected.sum()) == max(sizes),
                marker_rim=bool(a['rim'][seed]),
                island_evidence=bool(np.any(ignored_neighbors)),
                islands=len(_components(islands)),
                erasures=int(np.count_nonzero(a['wall'] & ~a['keep'])),
                diagonal=int(np.count_nonzero(a['diagonal'] & (completed == 0))),
                colors=len(set(int(v) for t in a['tiles'] for v in t.flat) - {0}),
                repeats=any(v > 1 for v in used.values()),
                checkpoints=int(np.count_nonzero(completed == 3)), blocked_ports=blocked_ports,
                rejected_links=a['stats']['rejected_links'], nodes=a['stats']['nodes'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, training, first_fit_failures = set(), [], 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(a['output'], out)
            assert not np.any(np.isin(out, (1, 9)))
            assert np.all(a['selected'][a['completed'] == 3])
            assert np.all(out[a['completed'] == 3] == 3)
            assert np.all(out[a['completed'] == 2] == 2)
            assert np.all(out[a['wall'] & a['keep']] == a['completed'][a['wall'] & a['keep']])
            assert np.all(out[a['wall'] & ~a['keep']] == 0)
            assert np.all(out[(a['completed'] == 0) & ~a['rim']] == 0)
            features = _features(a)
            if split == 'train':
                geometry, _, _ = _pack(a['board'], a['tiles'], connectivity=False)
                assert len(geometry) == 2, 'Training must show geometry alone is insufficient'
                # At least one geometric completion loses checkpoint reachability.
                assert any(not np.all(_finish(g)[1]['selected'][g == 3]) for g, _ in geometry)
                first_fit_failures += not np.array_equal(_finish(geometry[0][0])[0], out)
                optimistic = a['board'].copy()
                optimistic[optimistic == 9] = 0
                assert not np.array_equal(_finish(optimistic)[0], out), 'Packing ablation must change answer'
                training.append(features)
    assert any(f['small'] and not f['marker_rim'] for f in training)
    assert any(f['large'] and f['marker_rim'] for f in training)
    assert any(f['island_evidence'] and f['erasures'] and f['diagonal'] for f in training)
    assert any(f['surplus'] for f in training)
    assert any(f['repeats'] and f['blocked_ports'] for f in training)
    assert first_fit_failures, 'Training must refute first geometric fit in tray order'


def self_test():
    signatures, features = set(), []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        features.extend(_features(_analyze(pair['input'])) for s in ('train', 'test') for pair in p[s])
    assert len(signatures) == 42
    assert {f['axis'] for f in features} == {0, 1}
    assert len({f['shape'] for f in features}) >= 15
    assert len({f['pieces'] for f in features}) >= 15
    assert len({f['tiles'] for f in features}) >= 3
    assert len({f['vacancy_components'] for f in features}) >= 2
    assert len({f['colors'] for f in features}) >= 3
    assert any(f['repeats'] for f in features)
    assert max(f['used'] for f in features) >= 4
    assert any(f['surplus'] == 0 for f in features)
    assert all(f['rejected_links'] > 0 for f in features)
    p = generate_puzzle(17, 2, 1)
    q = generate_puzzle(17, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    inp = p['test'][0]['input'].copy()
    original = inp.copy()
    assert np.array_equal(solve(inp), p['test'][0]['output'])
    assert np.array_equal(inp, original)
    # Moving same-size source objects, or the whole tray, supplies no tile order.
    witness = next(pair for pair in p['train'] if _features(_analyze(pair['input']))['small'])
    board, tray, tiles, axis = _read_input(witness['input'])
    reordered = tray.copy()
    for component, tile in zip(_components(tray != 9), reversed(tiles)):
        yy, xx = np.where(component)
        y, x = int(yy.min()), int(xx.min())
        reordered[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    separator = np.ones((1, board.shape[1]) if axis == 0 else (board.shape[0], 1), np.int8)
    for panels in ((board, separator, reordered), (reordered, separator, board)):
        assert np.array_equal(solve(np.concatenate(panels, axis=axis)), witness['output'])
    for nt, ne in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(101, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Finite inventory, same-size distinct patterns, and a connectivity lock.
    board = np.full((5, 10), 4, np.int8)
    board[2, 0], board[2, -1] = 2, 3
    board[1:4, 1:9] = 9
    straight = np.full((3, 4), 4, np.int8)
    straight[1, :] = 0
    assert len(_pack(board, [straight])[0]) == 0
    assert len(_pack(board, [straight, straight])[0]) == 1
    sealed = np.full((3, 4), 4, np.int8)
    assert len(_pack(board, [straight, sealed], connectivity=False)[0]) == 2
    assert len(_pack(board, [straight, sealed])[0]) == 0
    # Explicit hole exclusion and external-complement corner behavior.
    room = np.full((9, 12), 4, np.int8)
    room[1:-1, 1:-1] = 0
    room[4, 5], room[4, 2], room[4, 9] = 6, 2, 3
    result, info = _finish(room)
    expected = room.copy()
    expected[1, 1:-1] = expected[-2, 1:-1] = 2
    expected[1:-1, 1] = expected[1:-1, -2] = 2
    assert np.array_equal(result, expected)
    assert result[3, 5] == result[4, 4] == 0 and result[4, 5] == 6
    attached = room.copy()
    attached[:5, 5] = 6
    changed, _ = _finish(attached)
    assert changed[5, 5] == 2 and changed[5, 4] == 2
    two_rooms = np.full((9, 17), 4, np.int8)
    two_rooms[1:-1, 1:-1] = 0
    two_rooms[:, 8] = 4
    two_rooms[4, 2], two_rooms[4, 5] = 2, 3
    two_rooms[4, 4], two_rooms[4, 12] = 7, 6
    expected = two_rooms.copy()
    expected[1, 1:8] = expected[-2, 1:8] = 2
    expected[1:-1, 1] = expected[1:-1, 7] = 2
    expected[4, 12] = 0
    assert np.array_equal(_finish(two_rooms)[0], expected)
    moved = two_rooms.copy()
    moved[4, 2], moved[1, 2] = 0, 2
    expected[4, 2] = 0
    assert np.array_equal(_finish(moved)[0], expected)
    across = two_rooms.copy()
    across[4, 2], across[4, 11] = 0, 2
    assert not np.all(_finish(across)[1]['selected'][across == 3])
    assert not _pack(across, [straight])[0]
    # Whole-scene isometries preserve both packing and the contour operation.
    for rot in range(4):
        for flip in (False, True):
            query, expected = np.rot90(inp, rot), np.rot90(p['test'][0]['output'], rot)
            if flip:
                query, expected = np.fliplr(query), np.fliplr(expected)
            assert np.array_equal(solve(query), expected)
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupt['test'][0]['output'][0, 0] = 9
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')
    return dict(seeds=42, pairs=len(features), shapes=len({f['shape'] for f in features}),
                rectangle_inventories=len({f['pieces'] for f in features}),
                tile_counts=sorted({f['tiles'] for f in features}),
                vacancy_component_counts=sorted({f['vacancy_components'] for f in features}),
                max_search_nodes=max(f['nodes'] for f in features))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
