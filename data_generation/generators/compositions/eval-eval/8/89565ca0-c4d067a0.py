"""Register and complete a magnified, room-colored rectilinear blueprint.

The large fragment matches room interiors, not dot coordinates or a regular
swatch lattice. Wall pixels become background gutters in the enlarged image.
Only standard Python and NumPy are used; solve reads input pixels exclusively.
"""

from itertools import combinations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        todo, cells = [(y, x)], []
        seen[y, x] = True
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= a < mask.shape[0] and 0 <= b < mask.shape[1]
                        and mask[a, b] and not seen[a, b]):
                    seen[a, b] = True
                    todo.append((a, b))
        result.append(cells)
    return result


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:  # Inclusive wall-coordinate bounds.
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _rooms(mask):
    """Return elementary room interiors as half-open rectangles, or reject."""
    if not (mask[0].all() and mask[-1].all()
            and mask[:, 0].all() and mask[:, -1].all()):
        raise ValueError("Blueprint outer frame is not closed")
    rooms = []
    for cells in _components(~mask):
        p = np.array(cells)
        t, l = map(int, p.min(axis=0))
        b, r = map(int, p.max(axis=0) + 1)
        if len(cells) != (b - t) * (r - l):
            raise ValueError("Every elementary room must have a rectangular interior")
        rooms.append((t, l, b, r))
    if not rooms:
        raise ValueError("Blueprint has no rooms")
    rebuilt = _walls(mask.shape, [(t - 1, l - 1, b, r) for t, l, b, r in rooms])
    if not np.array_equal(rebuilt, mask):
        raise ValueError("Unexplained or dangling wall")
    return sorted(rooms)


def _repair_sites(mask):
    """One-cell straight-wall gaps supported by direct collinear evidence.

    A pair across a room is insufficient: that row/column must also contain
    an adjacent wall pair. Dots are forbidden at these potential repair sites.
    """
    hs = np.any(mask[:, :-1] & mask[:, 1:], axis=1)
    vs = np.any(mask[:-1, :] & mask[1:, :], axis=0)
    sites = np.zeros_like(mask)
    sites[:, 1:-1] |= mask[:, :-2] & mask[:, 2:] & hs[:, None]
    sites[1:-1, :] |= mask[:-2, :] & mask[2:, :] & vs[None, :]
    return sites & ~mask


def _adjacent(a, b):
    # Half-open interior rectangles share a one-cell-wide dividing wall.
    t, l, d, r = a
    u, v, e, w = b
    return (((d + 1 == u or e + 1 == t) and max(l, v) < min(r, w))
            or ((r + 1 == v or w + 1 == l) and max(t, u) < min(d, e)))


def _registrations(rooms, colors, rectangles):
    """Color, dimensions, relative positions, adjacency; no canvas filtering.

    Rectangles are (top,left,bottom,right,color), half-open in global pixels.
    Returns every exact integer-scale registration of the two-room fragment.
    """
    matches = []
    first = rectangles[0]
    for i, room in enumerate(rooms):
        if colors[i] != first[4]:
            continue
        t, l, b, r = room
        sh, sw = first[2] - first[0], first[3] - first[1]
        h, w = b - t, r - l
        if sh % h or sw % w or sh // h != sw // w or sh // h < 2:
            continue
        scale = sh // h
        base = (first[0] - scale * t, first[1] - scale * l)
        indices = [i]
        for rect in rectangles[1:]:
            found = [j for j, q in enumerate(rooms)
                     if colors[j] == rect[4]
                     and tuple(base[k % 2] + scale * q[k] for k in range(4)) == rect[:4]]
            if len(found) != 1:
                break
            indices.append(found[0])
        else:
            if len(set(indices)) == len(indices) and _adjacent(rooms[indices[0]], rooms[indices[1]]):
                matches.append((scale, base, tuple(indices)))
    return matches


def _decode(input_grid):
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if frequencies[bg] <= np.partition(frequencies, -2)[-2]:
        raise ValueError("Background must be uniquely most frequent")
    candidates = []
    for wall in set(map(int, grid.flat)) - {bg}:
        p = np.argwhere(grid == wall)
        t, l = map(int, p.min(axis=0))
        b, r = map(int, p.max(axis=0) + 1)
        if min(b - t, r - l) < 5:
            continue
        local = grid[t:b, l:r]
        visible = local == wall
        repairs = _repair_sites(visible) & (local != bg)
        mask = visible | repairs
        try:
            rooms = _rooms(mask)
        except ValueError:
            continue
        repair_colors = set(map(int, local[repairs]))
        if len(repair_colors) > 1:
            continue
        # The damage ink is not a room swatch or a large block.
        if any(np.count_nonzero(grid == c) != np.count_nonzero(local[repairs] == c)
               for c in repair_colors):
            continue
        colors, dots = [], []
        good = True
        for y, x, d, e in rooms:
            points = np.argwhere(local[y:d, x:e] != bg)
            if len(points) > 1:
                good = False
                break
            if len(points):
                yy, xx = map(int, points[0] + (y, x))
                colors.append(int(local[yy, xx]))
                dots.append((yy, xx))
            else:
                colors.append(bg)
                dots.append(None)
        if not good:
            continue
        outside = grid.copy()
        outside[t:b, l:r] = bg
        rectangles = []
        for color in set(map(int, outside.flat)) - {bg}:
            for cells in _components(outside == color):
                q = np.array(cells)
                y, x = map(int, q.min(axis=0))
                d, e = map(int, q.max(axis=0) + 1)
                if min(d - y, e - x) < 2 or len(cells) != (d - y) * (e - x):
                    good = False
                    break
                rectangles.append((y, x, d, e, color))
        if not good or len(rectangles) != 2:
            continue
        rectangles.sort()
        matches = _registrations(rooms, colors, rectangles)
        if len(matches) != 1:
            continue
        scale, base, indices = matches[0]
        ey, ex = base[0] + scale * mask.shape[0], base[1] + scale * mask.shape[1]
        if not (0 <= base[0] < ey <= grid.shape[0] and 0 <= base[1] < ex <= grid.shape[1]):
            continue
        if not (ey <= t or b <= base[0] or ex <= l or r <= base[1]):
            continue
        # Every non-seed pixel of the enlarged bounding rectangle is empty.
        reserved = grid[base[0]:ey, base[1]:ex].copy()
        for y, x, d, e, _ in rectangles:
            reserved[y - base[0]:d - base[0], x - base[1]:e - base[1]] = bg
        if np.any(reserved != bg):
            continue
        added = [j for j, color in enumerate(colors) if color != bg and j not in indices]
        if not added:
            continue
        candidates.append(dict(background=bg, wall=wall, damage=tuple(repair_colors),
                               miniature=(t, l, b, r), mask=mask, rooms=rooms,
                               colors=colors, dots=dots, rectangles=rectangles,
                               scale=scale, base=base, seed=indices, added=added))
    if len(candidates) != 1:
        raise ValueError("Input must have one reconstructable, uniquely registered blueprint")
    return grid, candidates[0]


def _render(grid, meta):
    out = grid.copy()
    scale, (y, x) = meta["scale"], meta["base"]
    for (t, l, b, r), color in zip(meta["rooms"], meta["colors"]):
        if color != meta["background"]:
            out[y + scale * t:y + scale * b, x + scale * l:x + scale * r] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, meta = _decode(input_grid)
    return _render(grid, meta)


def _topology(rooms):
    ys = sorted({v for q in rooms for v in (q[0] - 1, q[2])})
    xs = sorted({v for q in rooms for v in (q[1] - 1, q[3])})
    return tuple(sorted((ys.index(t - 1), xs.index(l - 1), ys.index(b), xs.index(r))
                        for t, l, b, r in rooms))


def _partial(mask):
    hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(mask[:-1, :] & mask[1:, :], axis=0))
    return any(not mask[y].all() for y in hs) or any(not mask[:, x].all() for x in vs)


def _directions(meta):
    seeds = [meta["rooms"][i] for i in meta["seed"]]
    t, l = min(q[0] for q in seeds), min(q[1] for q in seeds)
    b, r = max(q[2] for q in seeds), max(q[3] for q in seeds)
    added = [meta["rooms"][i] for i in meta["added"]]
    return (any(q[0] < t for q in added), any(q[2] > b for q in added),
            any(q[1] < l for q in added), any(q[3] > r for q in added))


def _color_matches(meta):
    seq = tuple(q[4] for q in meta["rectangles"])
    return sum(_adjacent(a, b) and (meta["colors"][i], meta["colors"][j]) == seq
               for (i, a), (j, b) in combinations(enumerate(meta["rooms"]), 2))


def _cycle_count(mask):
    # Euler cycle rank of the connected wall graph, excluding solid 2x2 faces
    # that have no room interior. This does not use the room flood fill.
    edges = int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    edges += int(np.count_nonzero(mask[:-1, :] & mask[1:, :]))
    solid_faces = int(np.count_nonzero(mask[:-1, :-1] & mask[1:, :-1]
                                      & mask[:-1, 1:] & mask[1:, 1:]))
    return edges - int(mask.sum()) + 1 - solid_faces


def _origin_shortcut(grid, meta):
    """Ablation: assume the fragment starts at the blueprint's upper left."""
    s = meta["scale"]
    first = meta["rectangles"][0]
    wrong = dict(meta, base=(first[0] - s, first[1] - s))
    out = grid.copy()
    y, x = wrong["base"]
    for (t, l, b, r), color in zip(wrong["rooms"], wrong["colors"]):
        if color != wrong["background"]:
            a, c = max(0, y + s * t), max(0, x + s * l)
            d, e = min(grid.shape[0], y + s * b), min(grid.shape[1], x + s * r)
            if a < d and c < e:
                out[a:d, c:e] = color
    return out


def _lattice_shortcut(grid, meta):
    """Ablation: extend every divider across the entire blueprint."""
    mask = meta["mask"]
    hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(mask[:-1, :] & mask[1:, :], axis=0))
    lattice = np.zeros_like(mask)
    lattice[hs, :] = True
    lattice[:, vs] = True
    out = grid.copy()
    s, (y, x) = meta["scale"], meta["base"]
    # Extending staggered dividers can create adjacent wall coordinates and
    # zero-width bands. Keep the remaining positive-area elementary cells.
    for cells in _components(~lattice):
        points = np.array(cells)
        t, l = map(int, points.min(axis=0))
        b, r = map(int, points.max(axis=0) + 1)
        for color, dot in zip(meta["colors"], meta["dots"]):
            if dot is not None and t <= dot[0] < b and l <= dot[1] < r:
                out[y + s * t:y + s * b, x + s * l:x + s * r] = color
    return out


def _capacity(q):
    t, l, b, r = q
    return ((b - t) // 2) * ((r - l) // 2)


def _partition(rng, h, w, count):
    boxes = [(0, 0, h - 1, w - 1)]
    for _ in range(count - 1):
        options = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, stop in ((0, t, b), (1, l, r)):
                for cut in range(start + 2, stop - 1):
                    children = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else (
                        (t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(q) for j, q in enumerate(boxes) if j != i)
                    if capacity + sum(_capacity(q) for q in children) >= count:
                        options.append((i, children))
        if not options:
            raise ValueError("Requested partition does not fit")
        i, children = options[int(rng.integers(len(options)))]
        boxes[i:i + 1] = children
    return _walls((h, w), boxes)


def _place(rng, shape, scale):
    big = tuple(scale * n for n in shape)
    axes = [k for k in (0, 1) if shape[k] + big[k] + 1 <= 30]
    axis = int(rng.choice(axes))
    cross = 1 - axis
    gap = int(rng.integers(1, min(4, 30 - shape[axis] - big[axis]) + 1))
    needed = [max(shape[k], big[k]) for k in (0, 1)]
    needed[axis] = shape[axis] + big[axis] + gap
    canvas = tuple(int(rng.integers(n, min(30, n + 4) + 1)) for n in needed)
    if rng.random() < .15:
        canvas = (max(canvas), max(canvas))
    mini, base = [0, 0], [0, 0]
    margin = int(rng.integers(canvas[axis] - needed[axis] + 1))
    if rng.integers(2):
        mini[axis], base[axis] = margin, margin + shape[axis] + gap
    else:
        base[axis], mini[axis] = margin, margin + big[axis] + gap
    mini[cross] = int(rng.integers(canvas[cross] - shape[cross] + 1))
    base[cross] = int(rng.integers(canvas[cross] - big[cross] + 1))
    return canvas, tuple(mini), tuple(base)


def _make_pair(rng, witness=None, forbidden_topologies=(), avoid_scale=None,
               allow_blank=True, require_blank=False):
    # Retry geometry/registration together, never alter the semantic rule.
    for attempt in range(1200):
        scale = int(rng.choice([s for s in (2, 3, 4) if s != avoid_scale]))
        minimum_rooms = 5 if witness == "shape" or require_blank else 4 if witness else 3
        dimensions = [(h, w) for h in range(5, 11) for w in range(5, 11)
                      if max(scale * h, scale * w) <= 30
                      and any((scale + 1) * n + 1 <= 30 for n in (h, w))
                      and _capacity((0, 0, h - 1, w - 1)) >= minimum_rooms]
        if not dimensions:
            continue
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        cap = min(8, _capacity((0, 0, h - 1, w - 1)))
        count = int(rng.integers(minimum_rooms, cap + 1))
        mask = _partition(rng, h, w, count)
        rooms = _rooms(mask)
        if _topology(rooms) in forbidden_topologies:
            continue
        if witness == "shape" and not _partial(mask):
            continue
        ink = list(map(int, rng.permutation(10)))
        bg, wall, damage = ink[:3]
        palette = ink[3:3 + int(rng.integers(2, 5))]
        adjacent = [(i, j) for i, j in combinations(range(count), 2) if _adjacent(rooms[i], rooms[j])]
        i, j = adjacent[int(rng.integers(len(adjacent)))]
        colors = list(map(int, rng.choice(palette, size=count)))
        if witness == "shape":
            # Repeated color evidence deliberately cannot identify the fragment.
            colors = [palette[0]] * count
            remaining = [k for k in range(count) if k not in (i, j)]
            colors[int(rng.choice(remaining))] = palette[1]
        elif allow_blank and rng.random() < .6:
            for k in range(count):
                if k not in (i, j) and rng.random() < .22:
                    colors[k] = bg
        if require_blank and bg not in colors:
            colors[int(rng.choice([k for k in range(count) if k not in (i, j)]))] = bg
        if any(colors[k] == bg for k in (i, j)):
            continue
        rectangles = [tuple(scale * v for v in rooms[k]) + (colors[k],) for k in (i, j)]
        if len(_registrations(rooms, colors, rectangles)) != 1:
            continue
        added = [k for k in range(count) if colors[k] != bg and k not in (i, j)]
        meta = dict(rooms=rooms, colors=colors, seed=(i, j), added=added, rectangles=rectangles)
        if not added:
            continue
        if witness in ("shape", "transfer") and len({colors[k] for k in added}) < 2:
            continue
        if witness == "shape" and _color_matches(meta) < 2:
            continue
        dirs = _directions(meta)
        if witness in ("transfer", "test") and not ((dirs[0] and dirs[1]) or (dirs[2] and dirs[3])):
            continue
        # Dots may approach walls, but cannot impersonate a repair marker.
        forbidden_dots = _repair_sites(mask)
        dots = []
        possible = True
        for k, (t, l, b, r) in enumerate(rooms):
            sites = [(y, x) for y in range(t, b) for x in range(l, r)
                     if not forbidden_dots[y, x]]
            if colors[k] != bg and not sites:
                possible = False
                break
            dots.append(sites[int(rng.integers(len(sites)))] if colors[k] != bg else None)
        if not possible:
            continue
        local = np.full((h, w), bg, dtype=int)
        local[mask] = wall
        for color, dot in zip(colors, dots):
            if dot is not None:
                local[dot] = color
        # Supported one-cell damage; corners remain visible. A junction may
        # be damaged if its through-going segment still supplies the repair.
        damage_sites = []
        for y, x in np.argwhere(mask):
            y, x = int(y), int(x)
            if y in (0, h - 1) and x in (0, w - 1):
                continue
            trial = local.copy()
            trial[y, x] = damage
            visible = trial == wall
            repaired = visible | (_repair_sites(visible) & (trial != bg))
            if np.array_equal(repaired, mask):
                damage_sites.append((y, x))
        rng.shuffle(damage_sites)
        want_damage = witness == "shape" or rng.random() < .75
        if want_damage:
            if not damage_sites:
                continue
            for y, x in damage_sites[:int(rng.integers(1, min(4, len(damage_sites)) + 1))]:
                trial = local.copy()
                trial[y, x] = damage
                visible = trial == wall
                if np.array_equal(visible | (_repair_sites(visible) & (trial != bg)), mask):
                    local = trial
        canvas, mini, base = _place(rng, (h, w), scale)
        grid = np.full(canvas, bg, dtype=int)
        grid[mini[0]:mini[0] + h, mini[1]:mini[1] + w] = local
        for k in (i, j):
            t, l, b, r = rooms[k]
            grid[base[0] + scale * t:base[0] + scale * b,
                 base[1] + scale * l:base[1] + scale * r] = colors[k]
        # Labels are drawn directly from sampled clean rooms, independently of
        # input-only decoding and irrespective of where the swatch sits inside.
        output = grid.copy()
        for (t, l, b, r), color in zip(rooms, colors):
            if color != bg:
                output[base[0] + scale * t:base[0] + scale * b,
                       base[1] + scale * l:base[1] + scale * r] = color
        try:
            _, recovered = _decode(grid)
        except ValueError:
            continue
        if not np.array_equal(recovered["mask"], mask) or recovered["colors"] != colors:
            continue
        if not np.array_equal(solve(grid), output):
            raise AssertionError("Sampled geometry and input recovery disagree")
        if witness == "shape" and np.array_equal(_lattice_shortcut(grid, recovered), output):
            continue
        return {"input": grid, "output": output}, recovered
    raise RuntimeError("No valid room-fragment registration after 1200 bounded attempts")


def _check_counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    use_blanks = bool(rng.random() < .65)
    puzzle = {"train": [], "test": []}
    seen, topologies, first_scale = set(), set(), None
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            witness = ("shape" if index == 0 else "transfer" if index == 1 else None) if split == "train" else "test"
            for _ in range(16):
                pair, meta = _make_pair(rng, witness, topologies if split == "test" else (),
                                        first_scale if split == "train" and index == 1 else None,
                                        allow_blank=use_blanks,
                                        require_blank=use_blanks and split == "train" and index == 1)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                topologies.add(_topology(meta["rooms"]))
                if index == 0:
                    first_scale = meta["scale"]
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(g) not in seen, "Duplicate input"
            seen.add(_key(g))
            assert out.shape == g.shape and not np.array_equal(out, g)
            _, m = _decode(g)
            assert np.array_equal(out, _render(g, m)), "Incorrect room fill or changed preserved cell"
            assert _cycle_count(m["mask"]) == len(m["rooms"]), "Flood fill and wall graph count disagree"
            added_area = sum((m["rooms"][i][2] - m["rooms"][i][0])
                             * (m["rooms"][i][3] - m["rooms"][i][1]) for i in m["added"])
            assert np.count_nonzero(out != g) == added_area * m["scale"] ** 2
            # Check actual room count and shape after undoing the inferred scale.
            y, x = m["base"]
            s = m["scale"]
            room_canvas = np.full(m["mask"].shape, m["background"], dtype=int)
            for q, color in zip(m["rooms"], m["colors"]):
                t, l, b, r = q
                block = out[y + s * t:y + s * b, x + s * l:x + s * r]
                assert np.all(block == color), "Room interiors must be uniform, including empty rooms"
                room_canvas[t:b, l:r] = color
            big = out[y:y + s * m["mask"].shape[0], x:x + s * m["mask"].shape[1]]
            assert np.array_equal(big, np.repeat(np.repeat(room_canvas, s, 0), s, 1)), "Wall gutters must remain empty"
            t, l, b, r = m["miniature"]
            assert np.array_equal(out[t:b, l:r], g[t:b, l:r]), "Blueprint must be preserved"
            analyses[split].append((g, out, m))
    train = analyses["train"]
    assert len({m["scale"] for _, _, m in train}) >= 2, "Demonstrate scale inference"
    assert any(m["damage"] and _partial(m["mask"]) and _color_matches(m) >= 2
               and not np.array_equal(_lattice_shortcut(g, m), out) for g, out, m in train), "Missing topology/registration witness"
    assert any(((_directions(m)[0] and _directions(m)[1])
                or (_directions(m)[2] and _directions(m)[3]))
               and len({m["colors"][i] for i in m["added"]}) >= 2 for _, _, m in train), "Missing bidirectional literal transfer witness"
    assert any(not np.array_equal(_origin_shortcut(g, m), out) for g, out, m in train), "Registration ablation did not change an answer"
    training_topologies = {_topology(m["rooms"]) for _, _, m in train}
    for _, _, m in analyses["test"]:
        assert _topology(m["rooms"]) not in training_topologies, "Test must recombine partition topology"
        dirs = _directions(m)
        assert (dirs[0] and dirs[1]) or (dirs[2] and dirs[3]), "Test must extend on opposite sides of fragment"
        if m["background"] in m["colors"]:
            assert any(a["background"] in a["colors"] for _, _, a in train), "Empty rooms need a training demonstration"


def self_test():
    """Bounded 42-seed semantic audit, ablations, and input-only invariance."""
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in p:
            for pair in p[split]:
                _, m = _decode(pair["input"])
                records.append((split, pair, m))
    assert len({tuple(_key(pair[k]) for s in p for pair in p[s] for k in ("input", "output"))
                for p in puzzles}) == 42
    for seed in (0, 7):
        replay = generate_puzzle(seed)
        assert all(np.array_equal(a[k], b[k]) for s in replay
                   for a, b in zip(replay[s], puzzles[seed][s]) for k in a)
    assert len({pair["input"].shape for _, pair, _ in records}) >= 25
    assert len({_topology(m["rooms"]) for _, _, m in records}) >= 25
    assert len({m["base"] for _, _, m in records}) >= 20
    assert {m["scale"] for _, _, m in records} == {2, 3, 4}
    assert len({len(m["rooms"]) for _, _, m in records}) >= 4
    assert len({m["background"] for _, _, m in records}) >= 9
    assert len({m["wall"] for _, _, m in records}) >= 9
    assert any(m["background"] in m["colors"] for _, _, m in records)
    assert any(not m["damage"] for _, _, m in records)
    assert any(not _partial(m["mask"]) for _, _, m in records)
    # Reflection/transposition moves both representations; no hidden orientation.
    for _, pair, m in records[:12]:
        g, out = pair["input"], pair["output"]
        for transform in (lambda a: a.T, np.fliplr, np.flipud, lambda a: np.rot90(a)):
            assert np.array_equal(solve(transform(g)), transform(out))
        permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
        assert np.array_equal(solve(permutation[g]), permutation[out])
        # Moving each swatch within its own room cannot change the enlargement.
        altered = g.copy()
        t, l, b, r = m["miniature"]
        forbidden = _repair_sites(m["mask"])
        for q, color, dot in zip(m["rooms"], m["colors"], m["dots"]):
            if dot is None:
                continue
            altered[t + dot[0], l + dot[1]] = m["background"]
            y, x, d, e = q
            choices = [(yy, xx) for yy in range(y, d) for xx in range(x, e) if not forbidden[yy, xx]]
            yy, xx = choices[-1]
            altered[t + yy, l + xx] = color
        changed_out = solve(altered)
        changed_out[t:b, l:r] = out[t:b, l:r]
        assert np.array_equal(changed_out, out)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(4242 + 11 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # Ambiguity is checked before canvas bounds: a negative origin cannot be
    # used as a hidden tie-breaker to repair uninformative fragment evidence.
    rooms = [(1, 1, 3, 3), (1, 4, 3, 6), (4, 1, 6, 3), (4, 4, 6, 6)]
    rects = [(2, 2, 6, 6, 1), (2, 8, 6, 12, 1)]
    assert len(_registrations(rooms, [1] * 4, rects)) == 2
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
                 for s in ("train", "test")}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a corrupted answer")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, room/registration invariants, all legal counts)")
