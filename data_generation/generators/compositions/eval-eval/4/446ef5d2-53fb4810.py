"""Phase-locked loom: periodic predictions choose and complete an anchored jigsaw.

Only standard-library modules and NumPy are used. The solver reconstructs all
roles, source words, possible packings, and phase constraints from input pixels.
"""
from __future__ import annotations

import itertools
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNERS = ("TL", "TR", "BL", "BR")


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer ARC grid of shape 1..30 and colors 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        group, queue = [first], [first]
        for r, c in queue:
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
                    group.append(p)
        groups.append(group)
    return groups


def _box(cells):
    a = np.asarray(list(cells))
    r, c = a.min(axis=0)
    h, w = a.max(axis=0) - (r, c) + 1
    return int(r), int(c), int(h), int(w)


def _marker_cells(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r + dr, c), (r, c + dc), (r + dr, c + dc))


def _clipped(h, w):
    mask = np.ones((h, w), dtype=bool)
    mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    return mask


def _is_hub(mask):
    h, w = mask.shape
    return 3 <= h <= 7 and 3 <= w <= 7 and np.array_equal(mask, _clipped(h, w))


def _coords(box, direction, width, distance):
    r, c, h, w = box
    if direction < 2:
        y = r - distance if direction == 0 else r + h - 1 + distance
        return tuple((y, c + (w - width) // 2 + j) for j in range(width))
    x = c - distance if direction == 2 else c + w - 1 + distance
    return tuple((r + (h - width) // 2 + j, x) for j in range(width))


def _maximum(shape, box, direction):
    r, c, h, w = box
    return (r, shape[0] - r - h, c, shape[1] - c - w)[direction]


def _whole_period(word):
    # A shorter prefix may be compatible with ABA, but it is not a whole copy.
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0 and all(row == word[i % p] for i, row in enumerate(word)))


def _extract(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background frequency tie")
    marks = []
    for color in range(10):
        q = np.argwhere(grid == color)
        if color == bg or len(q) != 3:
            continue
        r, c, h, w = _box(q)
        if (h, w) != (2, 2):
            continue
        missing = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} - set(map(tuple, q.tolist()))
        point = missing.pop()
        if int(grid[point]) in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[(point[0]-r, point[1]-c)]
        marks.append((color, point, corner))
    if len(marks) != 1:
        raise ValueError("missing or ambiguous corner marker")
    marker, point, corner = marks[0]
    pieces, anchor = [], None
    for cells in _components((grid != bg) & (grid != marker)):
        r, c, h, w = _box(cells)
        matrix = np.full((h, w), -1, dtype=np.int8)
        for y, x in cells:
            matrix[y-r, x-c] = grid[y, x]
        if point in cells:
            anchor = len(pieces)
        pieces.append((matrix, (r, c)))
    if not 3 <= len(pieces) <= 6 or anchor is None:
        raise ValueError("expected 3..6 connected fragments and one anchor")
    m, (r, c) = pieces[anchor]
    expected = (r + (m.shape[0]-1 if corner[0] == "B" else 0),
                c + (m.shape[1]-1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("marker does not pin a fragment corner")
    payload = np.concatenate([m[m >= 0] for m, _ in pieces])
    frequencies = np.bincount(payload, minlength=10)
    carrier = int(frequencies.argmax())
    if np.count_nonzero(frequencies == frequencies[carrier]) != 1:
        raise ValueError("carrier frequency tie")
    interpretations = []
    for color in sorted(set(map(int, payload)) - {carrier}):
        try:
            sources = _sources(pieces, carrier, color)
        except ValueError:
            continue
        interpretations.append((color, sources))
    if len(interpretations) != 1:
        raise ValueError("missing or ambiguous hub color")
    hub_color, sources = interpretations[0]
    return dict(background=bg, marker=marker, point=point, corner=corner,
                carrier=carrier, hub_color=hub_color, pieces=pieces, anchor=anchor, sources=sources)


def _sources(pieces, carrier, hub_color):
    sources = []
    for index, (matrix, _) in enumerate(pieces):
        for cells in _components(matrix == hub_color):
            box = _box(cells)
            r, c, h, w = box
            if not _is_hub(matrix[r:r+h, c:c+w] == hub_color):
                raise ValueError("hub-color component is not a clipped solid hub")
            choices = []
            for direction in range(4):
                face_width = w if direction < 2 else h
                adjacent = _coords(box, direction, face_width, 1)
                if not all(0 <= y < matrix.shape[0] and 0 <= x < matrix.shape[1] for y, x in adjacent):
                    continue
                values = [int(matrix[p]) for p in adjacent]
                indices = [j for j, v in enumerate(values) if v not in (-1, carrier, hub_color)]
                width = len(indices)
                start = (face_width-width)//2
                if (not width or width > 3 or width > face_width-2 or (face_width-width) % 2
                        or indices != list(range(start, start+width))):
                    continue
                choices.append((direction, width))
            if len(choices) != 1:
                raise ValueError("hub requires one centered attached word")
            direction, width = choices[0]
            word = []
            for distance in range(1, _maximum(matrix.shape, box, direction)+1):
                points = _coords(box, direction, width, distance)
                values = tuple(int(matrix[p]) for p in points)
                if all(v not in (-1, carrier, hub_color) for v in values):
                    word.append(values)
                elif all(v == carrier for v in values):
                    break
                else:
                    raise ValueError("word is split or has a partial slice")
            else:
                raise ValueError("the whole word must end before a carrier slice on its source fragment")
            if not word or len(word) > 6:
                raise ValueError("expected a complete word of 1..6 slices")
            sources.append(dict(piece=index, box=box, cells=tuple(cells), direction=direction,
                                width=width, word=tuple(word), period=_whole_period(word)))
    if not 1 <= len(sources) <= 3:
        raise ValueError("expected 1..3 source hubs")
    return sources


def _packings(info, canvas):
    """Complete translation-only exact cover, including complementary notches."""
    pieces = info["pieces"]
    masks = [m >= 0 for m, _ in pieces]
    area = sum(int(m.sum()) for m in masks)
    anchor = info["anchor"]
    ar, ac = pieces[anchor][1]
    pr, pc = info["point"]
    corner = info["corner"]
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area // h
        oy = pr - (h-1 if corner[0] == "B" else 0)
        ox = pc - (w-1 if corner[1] == "R" else 0)
        if w > canvas[1] or oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m in masks):
            continue
        configs = []
        for mask in masks:
            coords = tuple(map(tuple, np.argwhere(mask).tolist()))
            bit = sum(1 << (y*w+x) for y, x in coords)
            configs.append((mask.shape, coords[0][1], bit))
        r, c = ar-oy, ac-ox
        ah, aw = masks[anchor].shape
        if r < 0 or c < 0 or r+ah > h or c+aw > w:
            continue
        positions = [None]*len(pieces)
        positions[anchor] = (r, c)
        full = (1 << area)-1
        occupied = configs[anchor][2] << (r*w+c)

        def visit(occ, remaining):
            if not remaining:
                yield (h, w), (oy, ox), tuple(positions)
                return
            free = full ^ occ
            pos = (free & -free).bit_length()-1
            y, x = divmod(pos, w)
            for i in remaining:
                (ph, pw), first, bit = configs[i]
                left = x-first
                if left < 0 or y+ph > h or left+pw > w:
                    continue
                placed = bit << (y*w+left)
                if not placed & occ:
                    positions[i] = (y, left)
                    yield from visit(occ | placed, tuple(j for j in remaining if j != i))
        yield from visit(occupied, tuple(i for i in range(len(pieces)) if i != anchor))


def _assemble(info, shape, positions):
    board = np.full(shape, -1, dtype=np.int8)
    labels = np.full(shape, -1, dtype=np.int8)
    for i, ((matrix, _), (r, c)) in enumerate(zip(info["pieces"], positions)):
        h, w = matrix.shape
        mask = matrix >= 0
        board[r:r+h, c:c+w][mask] = matrix[mask]
        labels[r:r+h, c:c+w][mask] = i
    return board, labels


def _complete(info, board, positions, check_echoes=True):
    """Existing colored cells constrain placement; carrier is a wildcard."""
    predictions, hubs = {}, set()
    rays = []
    for s in info["sources"]:
        r, c = positions[s["piece"]]
        hubs.update((r+y, c+x) for y, x in s["cells"])
    for s in info["sources"]:
        r, c = positions[s["piece"]]
        y, x, h, w = s["box"]
        box = (r+y, c+x, h, w)
        maximum = _maximum(board.shape, box, s["direction"])
        ray = []
        for d in range(1, maximum+1):
            points = _coords(box, s["direction"], s["width"], d)
            values = s["word"][(d-1) % len(s["word"])]
            for p, v in zip(points, values):
                if p in predictions or p in hubs:
                    return None
                predictions[p] = v
            ray.append(points)
        rays.append((s, tuple(ray)))
    if check_echoes:
        for p in np.ndindex(board.shape):
            v = int(board[p])
            if p in hubs:
                if v != info["hub_color"]:
                    return None
            elif v != info["carrier"] and (p not in predictions or predictions[p] != v):
                return None
        completed = board.copy()
    else:
        # A diagnostic counterfactual: discard echoes and predict only the hubs'
        # field. It is never used to choose a label or accept an assembly.
        completed = np.full_like(board, info["carrier"])
        for p in hubs:
            completed[p] = info["hub_color"]
    for p, v in predictions.items():
        completed[p] = v
    return completed, rays, predictions


def _infer(grid):
    info = _extract(grid)
    unique = {}
    rejected_fields = set()
    geometric_count = valid_count = 0
    for shape, origin, positions in _packings(info, grid.shape):
        geometric_count += 1
        board, labels = _assemble(info, shape, positions)
        result = _complete(info, board, positions)
        if result is None:
            projected = _complete(info, board, positions, check_echoes=False)
            if projected is not None:
                rejected_fields.add((shape, origin, projected[0].tobytes()))
            continue
        valid_count += 1
        key = (shape, origin, board.tobytes(), result[0].tobytes())
        if key not in unique:
            unique[key] = (origin, positions, board, labels, result)
    if len(unique) != 1:
        raise ValueError(f"phase evidence leaves {len(unique)} distinct assemblies")
    origin, positions, board, labels, (completed, rays, predictions) = next(iter(unique.values()))
    field_key = (board.shape, origin, completed.tobytes())
    output = np.full_like(grid, info["background"])
    r, c = origin
    output[r:r+board.shape[0], c:c+board.shape[1]] = completed
    info.update(origin=origin, positions=positions, board=board, labels=labels,
                completed=completed, rays=rays, predictions=predictions,
                geometric_count=geometric_count, valid_count=valid_count,
                phase_changes_field=any(key != field_key for key in rejected_fields))
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer only from this input; there is no generation state."""
    return _infer(_grid(input_grid))[0]


def _tile(rng, width, period, palette, whole_word=False):
    tile = rng.choice(palette, size=(period, width)).astype(np.int8)
    if whole_word:
        tile[2] = tile[0]
        tile[1, 0] = next(c for c in palette if c != int(tile[0, 0]))
    elif period > 1 and _whole_period(tuple(map(tuple, tile.tolist()))) != period:
        tile[-1, 0] = next(c for c in palette if c != int(tile[0, 0]))
    if width > 1:
        tile[0, -1] = next(c for c in palette if c != int(tile[0, 0]))
        if whole_word:
            tile[2] = tile[0]
    if not whole_word and period > 1 and _whole_period(tuple(map(tuple, tile.tolist()))) != period:
        tile[:] = tile[0]
        tile[-1, 0] = next(c for c in palette if c != int(tile[0, 0]))
    return tile


def _scene(rng, shape, primary, carrier, hub_color, palette, hub_count):
    specs = []
    for i in range(hub_count):
        width, period, whole = primary if i == 0 else (int(rng.integers(1, 3)), int(rng.integers(1, 5)), False)
        for _ in range(140):
            direction = int(rng.integers(4))
            transverse = width+2
            longitudinal = int(rng.integers(3, 5))
            h, w = (longitudinal, transverse) if direction < 2 else (transverse, longitudinal)
            if h > shape[0] or w > shape[1]:
                continue
            r = int(rng.integers(shape[0]-h+1))
            c = int(rng.integers(shape[1]-w+1))
            box = (r, c, h, w)
            maximum = _maximum(shape, box, direction)
            if maximum < period+4:
                continue
            hub = {(r+int(y), c+int(x)) for y, x in np.argwhere(_clipped(h, w))}
            path = {p for d in range(1, maximum+1) for p in _coords(box, direction, width, d)}
            protected = hub | {p for d in range(1, period+2) for p in _coords(box, direction, width, d)}
            protect_box = _box(protected)
            # Hubs get an empty collar apart from their attached word.
            collar = {(y+dy, x+dx) for y, x in hub for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
            if any(path & (s["path"] | s["hub"] | s["collar"]) or hub & (s["path"] | s["collar"])
                   or collar & s["path"] for s in specs):
                continue
            tile = _tile(rng, width, period, palette, whole)
            specs.append(dict(box=box, direction=direction, width=width, tile=tile,
                              hub=hub, path=path, collar=collar, protect_box=protect_box,
                              maximum=maximum))
            break
        else:
            return None
    return specs


def _partition(rng, shape, count, specs, stepped, mode):
    rects = [(0, 0, *shape)]
    protects = [s["protect_box"] for s in specs]
    for _ in range(count-1):
        choices = []
        for i, (r, c, h, w) in enumerate(rects):
            for axis, length in enumerate((h, w)):
                for cut in range(2, length-1):
                    absolute = (r if axis == 0 else c)+cut
                    if any((pr < absolute < pr+ph if axis == 0 else pc < absolute < pc+pw)
                           and not (pr+ph <= r or r+h <= pr or pc+pw <= c or c+w <= pc)
                           for pr, pc, ph, pw in protects):
                        continue
                    weight = h*w
                    if mode == "bands" and axis == 0:
                        weight *= .25
                    choices.append((i, axis, cut, weight))
        if not choices:
            return None
        weights = np.array([x[3] for x in choices], dtype=float)
        i, axis, cut, _ = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        r, c, h, w = rects.pop(i)
        if axis == 0:
            rects.extend(((r, c, cut, w), (r+cut, c, h-cut, w)))
        else:
            rects.extend(((r, c, h, cut), (r, c+cut, h, w-cut)))
    labels = np.full(shape, -1, dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rects):
        labels[r:r+h, c:c+w] = i
    if stepped:
        protected = np.zeros(shape, dtype=bool)
        for r, c, h, w in protects:
            protected[r:r+h, c:c+w] = True
        changes = 0
        for _ in range(90):
            r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            if protected[r, c] or r in (0, shape[0]-1) or c in (0, shape[1]-1):
                continue
            dr, dc = _DIRS[int(rng.integers(4))]
            old, new = int(labels[r, c]), int(labels[r+dr, c+dc])
            if old == new:
                continue
            labels[r, c] = new
            if len(_components(labels == old)) != 1 or len(_components(labels == new)) != 1:
                labels[r, c] = old
                continue
            changes += 1
            if changes >= int(rng.integers(1, 4)):
                break
        if changes == 0:
            return None
    return labels


def _paint(rng, labels, specs, carrier, hub_color):
    board = np.full(labels.shape, carrier, dtype=np.int8)
    full = board.copy()
    for s in specs:
        for p in s["hub"]:
            board[p] = full[p] = hub_color
        length = len(s["tile"])
        for d in range(1, s["maximum"]+1):
            points = _coords(s["box"], s["direction"], s["width"], d)
            values = s["tile"][(d-1) % length]
            show = d <= length or (d > length+1 and (rng.random() < .55 or d == s["maximum"]))
            for p, v in zip(points, values):
                full[p] = v
                if show:
                    board[p] = v
        # Expose consecutive colors at some real cuts; a valid join need not match.
        for d in range(length+3, s["maximum"]+1):
            prev = _coords(s["box"], s["direction"], s["width"], d-1)
            points = _coords(s["box"], s["direction"], s["width"], d)
            if any(labels[p] != labels[q] for p, q in zip(prev, points)):
                for p in prev+points:
                    board[p] = full[p]
        # At least one visible echo in every crossed receiver fragment.
        source_label = int(labels[next(iter(s["hub"]))])
        receiving = {}
        for d in range(length+2, s["maximum"]+1):
            points = _coords(s["box"], s["direction"], s["width"], d)
            for p in points:
                if labels[p] != source_label:
                    receiving.setdefault(int(labels[p]), []).append(p)
        for cells in receiving.values():
            p = cells[int(rng.integers(len(cells)))]
            board[p] = full[p]
    return board, full


def _fragments(labels, board):
    pieces = []
    for i in range(int(labels.max())+1):
        cells = np.argwhere(labels == i)
        r, c, h, w = _box(cells)
        m = np.where(labels[r:r+h, c:c+w] == i, board[r:r+h, c:c+w], -1).astype(np.int8)
        pieces.append((m, (r, c)))
    return pieces


def _scatter(rng, pieces, anchor, origin, point, corner, canvas):
    markers = _marker_cells(point, corner)
    remaining = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i][0].size)
    for _ in range(12):
        positions = [None]*len(pieces)
        positions[anchor] = tuple(origin[j]+pieces[anchor][1][j] for j in (0, 1))
        r, c = positions[anchor]
        rectangles = [(r, c, *pieces[anchor][0].shape)]
        for i in remaining:
            h, w = pieces[i][0].shape
            options = [(r, c) for r in range(canvas[0]-h+1) for c in range(canvas[1]-w+1)]
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                if any(r-1 <= y <= r+h and c-1 <= x <= c+w for y, x in markers):
                    continue
                if all(r+h < y or y+ph < r or c+w < x or x+pw < c for y, x, ph, pw in rectangles):
                    positions[i] = (r, c)
                    rectangles.append((r, c, h, w))
                    break
            else:
                break
        else:
            return positions
    return None


def _features(info):
    board, labels = info["board"], info["labels"]
    carrier = info["carrier"]
    changed_other = echo_other = unequal_seams = reset_difference = 0
    whole_witness = False
    for s, ray in info["rays"]:
        word = s["word"]
        for p in range(1, len(word)):
            if len(word) % p and all(v == word[i % p] for i, v in enumerate(word)):
                whole_witness |= any(word[(d-1) % len(word)] != word[(d-1) % p]
                                     for d in range(len(word)+1, len(ray)+1))
        entry = {}
        for d, points in enumerate(ray, 1):
            for j, point in enumerate(points):
                label = int(labels[point])
                if label != s["piece"]:
                    echo_other += int(board[point] != carrier)
                    changed_other += int(board[point] == carrier)
                    key = (label, j)
                    entry.setdefault(key, d)
                    reset_difference += int(word[(d-1) % len(word)][j] != word[(d-entry[key]) % len(word)][j])
                if d > 1:
                    prev = ray[d-2][j]
                    unequal_seams += int(labels[prev] != label and board[prev] != carrier
                                         and board[point] != carrier and board[prev] != board[point])
    return dict(shape=board.shape,
                count=len(info["pieces"]), corner=info["corner"], hubs=len(info["sources"]),
                widths=tuple(sorted({s["width"] for s in info["sources"]})),
                periods=tuple(sorted({s["period"] for s in info["sources"]})),
                directions=tuple(sorted({s["direction"] for s in info["sources"]})),
                stepped=sum(int(np.any(m < 0)) for m, _ in info["pieces"]),
                geometric_count=info["geometric_count"], echo_other=echo_other,
                phase_rejects=info["geometric_count"]-info["valid_count"],
                phase_changes_field=info["phase_changes_field"],
                moving_source=any(s["piece"] != info["anchor"] and
                                  info["pieces"][s["piece"]][1] != tuple(
                                      info["positions"][s["piece"]][j]+info["origin"][j] for j in (0, 1))
                                  for s in info["sources"]),
                changed_other=changed_other, unequal_seams=unequal_seams,
                reset_difference=reset_difference, whole_witness=whole_witness,
                background=info["background"], carrier=carrier,
                clipped=any(len(ray) % s["period"] for s, ray in info["rays"]))


def _make_pair(rng, primary, count, corner, stepped, mode, witness=False):
    for attempt in range(180):
        colors = [int(v) for v in rng.permutation(10)]
        bg, marker, carrier, hub_color = colors[:4]
        palette = colors[4:4+int(rng.integers(2, 7))]
        width, period, _ = primary
        h = int(rng.integers(max(7, width+4), 15))
        w = int(rng.integers(max(12, period+7), 20))
        hub_count = int(rng.choice((1, 1, 2, 2, 3))) if count >= 5 else int(rng.choice((1, 1, 2)))
        specs = _scene(rng, (h, w), primary, carrier, hub_color, palette, hub_count)
        if specs is None:
            continue
        labels = _partition(rng, (h, w), count, specs, stepped, mode)
        if labels is None:
            continue
        board, completed = _paint(rng, labels, specs, carrier, hub_color)
        # Geometry and the words may rotate together; scattered fragments never do.
        turns = int(rng.integers(4))
        labels, board, completed = (np.rot90(a, turns).copy() for a in (labels, board, completed))
        h, w = board.shape
        pieces = _fragments(labels, board)
        local_point = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        anchor = int(labels[local_point])
        area = h*w
        base_h = max(h+2, int(np.ceil(np.sqrt(area*2.4))))
        base_w = max(w+2, int(np.ceil(area*2.4/base_h)))
        canvas = (min(30, base_h+int(rng.integers(1, 6))), min(30, base_w+int(rng.integers(1, 6))))
        origin = (int(rng.integers(1, canvas[0]-h)), int(rng.integers(1, canvas[1]-w)))
        point = tuple(origin[j]+local_point[j] for j in (0, 1))
        positions = _scatter(rng, pieces, anchor, origin, point, corner, canvas)
        if positions is None:
            continue
        inp = np.full(canvas, bg, dtype=np.int8)
        for (matrix, _), (r, c) in zip(pieces, positions):
            ph, pw = matrix.shape
            sub = inp[r:r+ph, c:c+pw]
            sub[matrix >= 0] = matrix[matrix >= 0]
        for p in _marker_cells(point, corner):
            inp[p] = marker
        expected = np.full(canvas, bg, dtype=np.int8)
        expected[origin[0]:origin[0]+h, origin[1]:origin[1]+w] = completed
        try:
            output, info = _infer(inp)
        except ValueError:
            continue
        f = _features(info)
        if (not np.array_equal(output, expected) or info["background"] != bg or info["carrier"] != carrier
                or not f["echo_other"] or not f["changed_other"]
                or (witness and (not f["phase_changes_field"] or not f["moving_source"]
                                 or not f["phase_rejects"] or not f["reset_difference"]
                                 or not f["unequal_seams"] or not f["whole_witness"]))):
            continue
        if stepped and not f["stepped"]:
            continue
        return {"input": inp, "output": expected}
    # A deterministic construction retains the requested width/word and corner.
    return _fallback(rng, primary, count, corner, stepped, witness)


def _fallback(rng, primary, count, corner, stepped, witness):
    """Finite phase-coded repair, explicitly exercised by self_test.

    Dense echoes fix the ray field. A tab fixes the receiver with one missing
    slice; a three-piece ABA witness instead has two distinct phase signatures.
    Identical receiver permutations are allowed only when their raw images agree.
    """
    width, period, whole = primary
    for _ in range(240):
        palette = [4, 5, 6, 7, 8, 9]
        tile = _tile(rng, width, period, palette, whole)
        start = period+5
        h, w = width+4, start+2*(count-1)
        labels = np.zeros((h, w), dtype=np.int8)
        cuts = [start+2*i for i in range(count-1)]+[w]
        for i, (a, b) in enumerate(zip([0]+cuts[:-1], cuts)):
            labels[:, a:b] = i
        # A single tab fixes the receiver with the deliberately missing slice.
        # With two receivers and ABA their phases already distinguish the slots.
        if not witness or count >= 4:
            labels[1, start] = 0
        box = (1, 0, width+2, 3)
        specs = [dict(box=box, direction=3, width=width, tile=tile,
                      hub={(1+int(y), int(x)) for y, x in np.argwhere(_clipped(width+2, 3))},
                      maximum=w-3)]
        board, completed = _paint(rng, labels, specs, 2, 3)
        # Make the long-range evidence dense in the fallback.
        for d in range(period+2, w-2):
            for p in _coords(box, 3, width, d):
                board[p] = completed[p]
        for p in _coords(box, 3, width, start-1):
            board[p] = 2
        if corner[1] == "L":
            labels, board, completed = (a[:, ::-1].copy() for a in (labels, board, completed))
        if corner[0] == "B":
            labels, board, completed = (a[::-1].copy() for a in (labels, board, completed))
        pieces = _fragments(labels, board)
        origin = (10, 1) if corner[1] == "L" else (10, 29-w)
        local = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        anchor = int(labels[local])
        point = tuple(origin[j]+local[j] for j in (0, 1))
        positions = _scatter(rng, pieces, anchor, origin, point, corner, (30, 30))
        if positions is None:
            continue
        inp = np.zeros((30, 30), dtype=np.int8)
        for (m, _), (r, c) in zip(pieces, positions):
            ph, pw = m.shape
            inp[r:r+ph, c:c+pw][m >= 0] = m[m >= 0]
        for p in _marker_cells(point, corner):
            inp[p] = 1
        expected = np.zeros_like(inp)
        r, c = origin
        expected[r:r+h, c:c+w] = completed
        try:
            out, info = _infer(inp)
        except ValueError:
            continue
        f = _features(info)
        if (np.array_equal(out, expected) and f["changed_other"] and f["echo_other"]
                and (not witness or (f["phase_changes_field"] and f["moving_source"]
                                     and f["phase_rejects"] and f["reset_difference"]
                                     and f["whole_witness"] and f["unequal_seams"]))):
            mapping = rng.permutation(10).astype(np.int8)
            return {"input": mapping[inp], "output": mapping[expected]}
    raise RuntimeError("finite fallback could not establish unique phase evidence")


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _unique(pair, used):
    if _key(pair["input"]) not in used:
        return pair
    colors = list(map(int, np.unique(pair["input"])))
    for values in itertools.permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _key(candidate["input"]) not in used:
            return candidate
    raise RuntimeError("recoloring space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed") % (1 << 64)
    nt = _integer(num_train, "num_train", 2, 10)
    ne = _integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("width", "pieces", "corner")))
    corners = tuple(str(v) for v in rng.choice(_CORNERS, 2, replace=False))
    test_corners = tuple(c for c in _CORNERS if c not in corners) if regime == "corner" else _CORNERS
    puzzle, used = {"train": [], "test": []}, set()
    for i in range(nt):
        primary = (1, 3, True) if i == 0 else (2, 2, False) if i == 1 else (int(rng.integers(1, 3)), int(rng.integers(2, 5)), False)
        count = int(rng.integers(3, 5 if regime == "pieces" else 7))
        pair = _make_pair(rng, primary, count, corners[i % 2], i == 1 or bool(rng.integers(2)),
                          str(rng.choice(("recursive", "bands"))), witness=i == 0)
        pair = _unique(pair, used)
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    rng.shuffle(puzzle["train"])
    for _ in range(ne):
        primary = (3 if regime == "width" else int(rng.integers(1, 3)), int(rng.integers(3, 6)), False)
        count = int(rng.integers(5, 7)) if regime == "pieces" else int(rng.integers(3, 7))
        pair = _make_pair(rng, primary, count, str(rng.choice(test_corners)), bool(rng.integers(2)),
                          str(rng.choice(("recursive", "bands"))))
        pair = _unique(pair, used)
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    used, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            assert _key(inp) not in used
            used.add(_key(inp))
            predicted, info = _infer(inp)
            assert np.array_equal(predicted, out), "answer violates phase-constrained assembly"
            bg, marker = info["background"], info["marker"]
            assert not np.any(out == marker)
            assert all(out[p] == bg for p in _marker_cells(info["point"], info["corner"]))
            assert np.count_nonzero(inp != bg)-3 == np.count_nonzero(out != bg)
            r, c = info["origin"]
            h, w = info["board"].shape
            assert np.all(out[r:r+h, c:c+w] != bg)
            assert all((info["positions"][info["anchor"]][j]+info["origin"][j]
                        == info["pieces"][info["anchor"]][1][j]) for j in (0, 1))
            for p in np.ndindex(info["board"].shape):
                if info["board"][p] != info["carrier"]:
                    assert info["completed"][p] == info["board"][p], "existing colored evidence changed"
                elif p not in info["predictions"]:
                    assert info["completed"][p] == info["carrier"]
            f = _features(info)
            assert f["echo_other"] and f["changed_other"]
            features[split].append(f)
    training = features["train"]
    assert any(f["whole_witness"] and f["unequal_seams"] and f["reset_difference"]
               and f["phase_rejects"] and f["phase_changes_field"] and f["moving_source"]
               for f in training)
    assert any(max(f["widths"]) >= 2 for f in training)
    assert any(f["stepped"] for f in training)
    assert len({f["corner"] for f in training}) >= 2
    testing = features["test"]
    wider = min(max(f["widths"]) for f in testing) > max(max(f["widths"]) for f in training)
    more_pieces = min(f["count"] for f in testing) > max(f["count"] for f in training)
    new_corners = not ({f["corner"] for f in training} & {f["corner"] for f in testing})
    assert wider or more_pieces or new_corners, "expected an evidenced structural train/test shift"


def self_test():
    assert _whole_period(((4,), (5,), (4,))) == 3
    assert _whole_period(((4,), (5,), (4,), (5,))) == 2
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(pair[k]) for split in ("train", "test")
                             for pair in puzzle[split] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, info = _infer(pair["input"])
                assert np.array_equal(out, pair["output"])
                # Removing B leaves the correctly assembled board unfinished.
                raw = np.full_like(out, info["background"])
                r, c = info["origin"]
                h, w = info["board"].shape
                raw[r:r+h, c:c+w] = info["board"]
                assert not np.array_equal(raw, out)
                # Removing A retains separated pieces rather than the rectangle.
                assert any(i != info["anchor"] and old != (r+y, c+x)
                           for i, ((_, old), (y, x)) in enumerate(zip(info["pieces"], info["positions"])))
                f = _features(info)
                f["canvas_shape"] = pair["input"].shape
                records.append(f)
    assert len(signatures) == 42
    assert {f["corner"] for f in records} == set(_CORNERS)
    assert {d for f in records for d in f["directions"]} == set(range(4))
    assert {w for f in records for w in f["widths"]} == {1, 2, 3}
    assert {p for f in records for p in f["periods"]} >= {1, 2, 3, 4, 5}
    assert {f["count"] for f in records} == {3, 4, 5, 6}
    assert len({f["shape"] for f in records}) >= 35
    assert len({f["canvas_shape"] for f in records}) >= 35
    assert len({f["background"] for f in records}) == 10
    assert any(f["hubs"] >= 2 for f in records)
    assert any(f["stepped"] for f in records) and any(not f["stepped"] for f in records)
    assert any(f["clipped"] for f in records) and any(not f["clipped"] for f in records)
    # Determinism, count boundaries, and strict invalid-value handling. The host
    # verifier separately exercises every legal count combination.
    p = generate_puzzle(123456, 7, 3)
    q = generate_puzzle(123456, 7, 3)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for a, b in ((2, 1), (10, 1), (10, 4)):
        q = generate_puzzle(101+a*7+b, a, b)
        assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # These bypass the random scene sampler and exercise the finite repair path.
    for corner in _CORNERS:
        for primary, count in (((1, 3, True), 3), ((1, 3, True), 6),
                               ((2, 2, False), 3), ((3, 5, False), 6)):
            pair = _fallback(np.random.default_rng(58), primary, count, corner, True, primary[2])
            assert np.array_equal(solve(pair["input"]), pair["output"])
    # Palette permutations and reflection must preserve the geometry-based rule.
    pair = p["test"][0]
    mapping = np.array([7, 2, 0, 9, 3, 8, 6, 4, 1, 5], dtype=np.int8)
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
    for corruption in ("phase", "marker", "anchor", "duplicate"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        _, info = _infer(pair["input"])
        if corruption == "phase":
            point = next(q for q in info["predictions"] if info["board"][q] == info["carrier"])
            y, x = (point[j]+info["origin"][j] for j in (0, 1))
            pair["output"][y, x] = info["carrier"]
        elif corruption == "marker":
            pair["output"][_marker_cells(info["point"], info["corner"])[0]] = info["marker"]
        elif corruption == "anchor":
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        else:
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("corruption accepted: "+corruption)
