"""Veiled communicating vessels: complete porous walls and settle their water.

The reader uses only pixels. Generation renders complete scenes, constructs
complete stable liquid levels, and then hides terrain with a marked shutter.
No parent generator, external file, or seed-dependent solving convention is used.
"""

import numpy as np


def _integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, c) for c in range(l, r + 1)}
            | {(b, c) for c in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)}
            | {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _closed_pools(grid, background, water):
    """Independent local closure check: every body fills its basin below its top.

    A free path to an unfilled cell must start with an unfilled neighbor. Thus
    checking such neighbors at/below each body's meniscus is a flood-closure
    test, including sideways leaks, unsupported cells, and partial levels.
    """
    h, w = grid.shape
    pools = _components(grid == water)
    if not pools:
        return None
    levels = []
    for pool in pools:
        surface = min(y for y, _ in pool)
        levels.append(surface)
        for y, x in pool:
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if surface <= yy < h and 0 <= xx < w and grid[yy, xx] == background:
                    return None
    return tuple(sorted(levels))


def _equalizing_cell(grid, wet_point, background, water, origin, reverse):
    """An added unit can displace a pool into unfilled space at its own level.

    This matters below overhanging inner floors: a column-only pour would stack
    beside the overhang instead of spreading beneath it.
    """
    h, w = grid.shape
    stack, body = [wet_point], {wet_point}
    while stack:
        y, x = stack.pop()
        for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            yy, xx = q
            if 0 <= yy < h and 0 <= xx < w and grid[q] == water and q not in body:
                body.add(q)
                stack.append(q)
    top = min(y for y, x in body)
    outlets = {q for y, x in body for q in
               ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
               if top <= q[0] < h and 0 <= q[1] < w and grid[q] == background}
    if not outlets:
        return None
    return min(outlets, key=lambda p: (-p[0], abs(p[1] - origin[1]),
                                      -p[1] if reverse else p[1]))


def _drop(grid, point, background, water, reverse=False):
    """Fall; on support, seek a lateral downward outlet; otherwise retain a cell."""
    h, w = grid.shape
    y, x = map(int, point)
    if grid[y, x] == water:
        cell = _equalizing_cell(grid, (y, x), background, water, (y, x), reverse)
        if cell is not None:
            y, x = cell
    while y > 0 and grid[y, x] == water:
        y -= 1
    if grid[y, x] != background:
        raise ValueError("liquid source is trapped below a filled ceiling")
    for _ in range(h + 1):
        if y + 1 < h and grid[y + 1, x] == background:
            y += 1
            continue
        if y + 1 < h and grid[y + 1, x] == water:
            cell = _equalizing_cell(grid, (y + 1, x), background, water, (y, x), reverse)
            if cell is not None and cell[0] > y:
                y, x = cell
                continue
        left = right = x
        while left > 0 and grid[y, left - 1] == background:
            left -= 1
        while right + 1 < w and grid[y, right + 1] == background:
            right += 1
        exits = [] if y + 1 == h else [c for c in range(left, right + 1)
                                                     if grid[y + 1, c] == background]
        if not exits:
            grid[y, x] = water
            return
        x = min(exits, key=lambda c: (abs(c - x), -c if reverse else c))
        y += 1
    raise AssertionError("downward flow failed to terminate")


def _settle(scene, background, water, reverse=False, top_first=False):
    result = scene.copy()
    sources = sorted(map(tuple, np.argwhere(scene == water)),
                     key=lambda p: (p[0] if top_first else -p[0],
                                    -p[1] if reverse else p[1]))
    result[result == water] = background
    for p in sources:
        _drop(result, p, background, water, reverse)
    return result


def _shutter(grid):
    """The frame has a water-color tab; actual droplets can overprint it."""
    bg = int(grid[0, 0])
    if any(int(grid[p]) != bg for p in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("background must occur at all four canvas corners")
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(grid == color)))
        t, b, l, r = box = _bbox(cells)
        if b - t < 3 or r - l < 3 or t < 1 or l < 1:
            continue
        if b >= grid.shape[0] - 1 or r >= grid.shape[1] - 1:
            continue
        tab = (t, l)
        perimeter = _perimeter(box)
        if not cells <= perimeter - {tab}:
            continue
        water = int(grid[tab])
        if water in (bg, color):
            continue
        if any(int(grid[p]) not in (color, water) for p in perimeter):
            continue
        fills = set(map(int, np.unique(grid[t + 1:b, l + 1:r]))) - {water}
        if len(fills) != 1:
            continue
        veil = next(iter(fills))
        if veil in (bg, color, water):
            continue
        if set(map(tuple, np.argwhere(grid == veil))) - _area(box):
            continue
        candidates.append((bg, water, color, veil, box))
    if len(candidates) != 1:
        raise ValueError("exactly one marked shutter is required")
    return candidates[0]


def _read_input(value):
    g = _grid(value)
    bg, water, frame, veil, main = _shutter(g)
    mask = _area(main)
    tab = (main[0], main[2])
    supply = (g == water)
    supply[tab] = False
    if not supply.any():
        raise ValueError("there must be actual water in addition to the color tab")
    # Sources can sit in openings, so water at a perimeter is evidence of free
    # space. They can also be overprinted inside the opaque shutter.
    evidence = g.copy()
    evidence[evidence == water] = bg
    reserved = {bg, water, frame, veil}
    colors = set(map(int, np.unique(g))) - reserved
    points = {c: set(map(tuple, np.argwhere(g == c))) - mask for c in colors}
    if any(not p for p in points.values()):
        raise ValueError("a terrain color has no exposed evidence")
    candidates = []
    for base in sorted(colors):
        box = _bbox(points[base])
        t, b, l, r = box
        if b - t < 3 or r - l < 2:
            continue
        top = {(t, x) for x in range(l, r + 1)} - mask
        bottom = {(b, x) for x in range(l, r + 1)} - mask
        if not top or not bottom or any(evidence[p] != base for p in top | bottom):
            continue
        side = {(y, x) for y in range(t + 1, b) for x in (l, r)}
        visible = side - mask
        predictions, fits = {}, []
        for period in range(1, 7):
            word, witnesses = {}, [set() for _ in range(period)]
            for y, x in sorted(visible):
                residue, color = y % period, int(evidence[y, x])
                if residue in word and word[residue] != color:
                    break
                word[residue] = color
                witnesses[residue].add(y)
            else:
                if any(len(rows) < 2 for rows in witnesses):
                    continue
                values = set(word.values())
                if bg not in values or base not in values or len(values) > 3:
                    continue
                if values - {bg} - colors:
                    continue
                full_word = tuple(word[i] for i in range(period))
                prediction = tuple(full_word[y % period] for y, x in sorted(side))
                predictions[prediction] = full_word
                fits.append((period, full_word))
        if len(predictions) != 1:
            continue
        period, word = min(fits)
        values = {base} | (set(word) - {bg})
        assignments = {p: base for p in _perimeter(box) - side}
        assignments.update({(y, x): word[y % period] for y, x in side})
        # Every exposed occurrence of an assigned solid color must belong to
        # this vessel; no merging by similar colors or generation descriptors.
        if any({p for p, c in assignments.items() if c == color and p not in mask}
               != points[color] for color in values):
            continue
        if any(int(evidence[p]) != c for p, c in assignments.items() if p not in mask):
            continue
        candidates.append({"box": box, "base": base, "word": word,
                           "period": period, "colors": frozenset(values),
                           "assignments": assignments})
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for c in candidates:
            cells = set(c["assignments"])
            if first in c["colors"] and c["colors"] <= remaining and not (cells & _halo(occupied)):
                cover(remaining - c["colors"], chosen + [c], occupied | cells)

    cover(colors, [], set())
    if len(covers) != 1 or not covers[0]:
        raise ValueError("vessel geometry or palette grouping is not unique")
    vessels = covers[0]
    terrain = np.full(g.shape, bg, dtype=np.int8)
    for vessel in vessels:
        for p, c in vessel["assignments"].items():
            terrain[p] = c
    if np.any(supply & (terrain != bg)):
        raise ValueError("visible water cannot occupy a reconstructed solid cell")
    scene = terrain.copy()
    scene[supply] = water
    return scene, {"background": bg, "water": water, "frame": frame,
                   "veil": veil, "shutter": main, "mask": mask,
                   "supply": supply, "terrain": terrain, "vessels": vessels}


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene, info = _read_input(input_grid)
    result = _settle(scene, info["background"], info["water"])
    if _closed_pools(result, info["background"], info["water"]) is None:
        raise ValueError("family members require complete stable liquid levels")
    return result


def _vessel_grid(shape, boxes, words, bases, background):
    terrain = np.full(shape, background, dtype=np.int8)
    for (t, b, l, r), word, base in zip(boxes, words, bases):
        terrain[t, l:r + 1] = base
        terrain[b, l:r + 1] = base
        for y in range(t + 1, b):
            terrain[y, l] = terrain[y, r] = word[y % len(word)]
    return terrain


def _geometry(rng, count, max_period, communicating_demo=False):
    """Nested but usually nonconcentric chambers; tight or loose lower insets."""
    inner_period = int(rng.integers(3, min(4 if count == 3 else 5, max_period) + 1))
    ih = int(rng.integers(2 * inner_period + 3, 2 * inner_period + 7))
    iw = int(rng.integers(6, 10 if count == 3 else 13))
    boxes, periods = [(0, ih - 1, 0, iw - 1)], [inner_period]
    for _ in range(count - 1):
        t, b, l, r = boxes[-1]
        top = int(rng.integers(2, 4))
        bottom = 2 if communicating_demo else int(rng.integers(2, 5 if count == 3 else 7))
        left, right = map(int, rng.integers(2, 5, size=2))
        boxes.append((t - top, b + bottom, l - left, r + right))
        periods.append(int(rng.integers(3, min(max_period, (b + bottom - t + top - 1) // 2) + 1)))
    top_pad, bottom_pad, left_pad, right_pad = map(int, rng.integers(1, 4, size=4))
    t, b, l, r = boxes[-1]
    h, w = b - t + 1 + top_pad + bottom_pad, r - l + 1 + left_pad + right_pad
    if h > 30 or w > 30:
        return None
    dy, dx = top_pad - t, left_pad - l
    boxes = [(a + dy, z + dy, c + dx, d + dx) for a, z, c, d in boxes]
    return (h, w), boxes, periods


def _words(rng, boxes, periods, palette, background, rich, communicating_demo=False):
    words, bases = [], []
    for index, (box, period) in enumerate(zip(boxes, periods)):
        base, accent = palette[2 * index:2 * index + 2]
        use_accent = rich and index == 0 or bool(rng.integers(2))
        entries = [background] + [base] * (period - 1)
        if use_accent:
            for k in rng.choice(np.arange(1, period), size=int(rng.integers(1, period - 1)), replace=False):
                entries[int(k)] = accent
        rng.shuffle(entries)
        # Both side walls use a row-indexed word. A lowest opening must leave
        # at least one retained layer, rather than empty every vessel outright.
        phases = [p for p in range(period) if entries[(box[1] - 1 + p) % period] != background]
        if communicating_demo:
            opening_row = box[1] - (2 if index == 0 else 5)
            phase = (entries.index(background) - opening_row) % period
        else:
            phase = int(rng.choice(phases))
        words.append(tuple(entries[(i + phase) % period] for i in range(period)))
        bases.append(base)
    return words, bases


def _target_candidates(terrain, inner, background, water, overflow):
    """Enumerate bounded stream volumes; retain only closed, complete levels."""
    t, b, l, r = inner
    capacity = (b - t - 1) * (r - l - 1)
    stream = terrain.copy()
    anchor = (t + 1, (l + r) // 2)
    inside = np.zeros(terrain.shape, dtype=bool)
    inside[t + 1:b, l + 1:r] = True
    candidates = []
    for volume in range(1, capacity + 1):
        try:
            _drop(stream, anchor, background, water)
        except ValueError:
            break
        # Check a cheap necessary condition before computing components.
        wet = stream == water
        if volume < 2 or bool(np.any(wet & ~inside)) != overflow:
            continue
        if _closed_pools(stream, background, water) is not None:
            candidates.append((volume, stream.copy()))
    return candidates


def _source(rng, shape, inner, target, volume, style):
    t, b, l, r = inner
    points = np.array([(y, x) for y in range(t + 1, b) for x in range(l + 1, r)], dtype=int)
    anchor = points[int(rng.integers(len(points)))]
    dy, dx = abs(points[:, 0] - anchor[0]), abs(points[:, 1] - anchor[1])
    jitter = rng.random(len(points))
    if style == 0:
        score = dy + dx + jitter
    elif style == 1:
        score = jitter
    elif style == 2:
        score = 4 * dy + 0.1 * dx + jitter
    elif style == 3:
        score = 4 * dx + 0.1 * dy + jitter
    else:
        score = (~target[points[:, 0], points[:, 1]]) * 3 + jitter
    chosen = points[np.argsort(score)[:volume]]
    moving = np.flatnonzero(~target[points[:, 0], points[:, 1]])
    if len(moving) and np.all(target[chosen[:, 0], chosen[:, 1]]):
        # A mostly-settled source must still teach movement. Without this,
        # one-vessel retained examples of style 4 are necessarily identities.
        chosen[-1] = points[int(rng.choice(moving))]
    result = np.zeros(shape, dtype=bool)
    result[chosen[:, 0], chosen[:, 1]] = True
    return result


def _remove_vessel(terrain, box, background):
    result = terrain.copy()
    for p in _perimeter(box):
        result[p] = background
    return result


def _shutter_input(rng, terrain, source, target, boxes, periods, roles):
    bg, water, frame, veil = roles
    h, w = terrain.shape
    full = set().union(*map(_perimeter, boxes))
    openings = {p for p in full if terrain[p] == bg}
    for _ in range(140):
        i = int(rng.integers(len(boxes)))
        t0, b0, l0, r0 = boxes[i]
        focus = (b0 - int(rng.integers(periods[i] + 1)), int(rng.choice([l0, r0])))
        hh, ww = int(rng.integers(4, min(12, h - 2) + 1)), int(rng.integers(4, min(12, w - 2) + 1))
        t = int(np.clip(focus[0] - int(rng.integers(hh)), 1, h - hh - 1))
        l = int(np.clip(focus[1] - int(rng.integers(ww)), 1, w - ww - 1))
        box = (t, t + hh - 1, l, l + ww - 1)
        mask, border = _area(box), _perimeter(box)
        # The entire shutter is behind the droplets, including its frame.
        # Only the role tab may not coincide with an actual source cell.
        if source[box[0], box[2]]:
            continue
        if len(openings & mask) < 1 or sum(terrain[p] != bg for p in mask) < 3:
            continue
        missing = terrain.copy()
        for p in mask:
            missing[p] = bg
        missing[source] = water
        if np.array_equal(_settle(missing, bg, water) == water, target == water):
            continue
        inp = terrain.copy()
        inp[t:box[1] + 1, l:box[3] + 1] = veil
        for p in border:
            inp[p] = frame
        inp[t, l] = water
        inp[source] = water
        try:
            reconstructed, info = _read_input(inp)
        except ValueError:
            continue
        if not np.array_equal(info["terrain"], terrain) or len(info["vessels"]) != len(boxes):
            continue
        if np.array_equal(_settle(reconstructed, bg, water), target):
            return inp
    return None


def _make_pair(rng, roles, count, overflow, rich, style, max_period, period_demo=False):
    bg, water, _, _ = roles
    palette = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    for _ in range(70):
        geometry = _geometry(rng, count, max_period, period_demo)
        if geometry is None:
            continue
        shape, boxes, periods = geometry
        if period_demo:
            periods[-1] = max_period
        words, bases = _words(rng, boxes, periods, palette, bg, rich, period_demo)
        terrain = _vessel_grid(shape, boxes, words, bases, bg)
        choices = _target_candidates(terrain, boxes[0], bg, water, overflow)
        if period_demo:
            info = {"water": water, "vessels": [{"box": b} for b in boxes]}
            choices = [(v, out) for v, out in choices if _communicating_scene(out, info)]
        rng.shuffle(choices)
        for volume, target in choices[:12]:
            for _ in range(5):
                source = _source(rng, shape, boxes[0], target == water, volume, style)
                if period_demo and len(_components(source)) < 2:
                    continue
                scene = terrain.copy()
                scene[source] = water
                if np.array_equal(scene, target):
                    continue
                if np.argwhere(source)[:, 0].sum() >= np.argwhere(target == water)[:, 0].sum():
                    continue
                try:
                    if not all(np.array_equal(_settle(scene, bg, water, rev, order), target)
                               for rev in (False, True) for order in (False, True)):
                        continue
                    # Every sampled vessel must affect water placement, including
                    # the extra third vessel in tests. Restored art alone is not
                    # accepted as a contribution to the new transformation.
                    useful = True
                    for box in boxes:
                        ablated = _remove_vessel(terrain, box, bg)
                        ablated[source] = water
                        if np.array_equal(_settle(ablated, bg, water) == water, target == water):
                            useful = False
                            break
                    if not useful:
                        continue
                    if overflow:
                        sealed = terrain.copy()
                        for box, base in zip(boxes, bases):
                            for p in _perimeter(box):
                                if sealed[p] == bg:
                                    sealed[p] = base
                        sealed[source] = water
                        if np.array_equal(_settle(sealed, bg, water) == water, target == water):
                            continue
                except ValueError:
                    continue
                inp = _shutter_input(rng, terrain, source, target, boxes, periods, roles)
                if inp is not None:
                    return {"input": inp, "output": target.copy()}
    raise RuntimeError("bounded construction could not find an unambiguous veiled scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    nt, ne = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
    max_period = 5
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for index in range(number):
            if split == "test":
                count, overflow, rich = 3, True, True
            elif index == 0:
                count, overflow, rich = 1, False, False
            elif index == 1:
                count, overflow, rich = 2, True, True
            else:
                count = int(rng.integers(1, 4))
                overflow, rich = count > 1 or bool(rng.integers(2)), bool(rng.integers(2))
            style = index % 5 if split == "train" else int(rng.integers(5))
            for _ in range(8):
                pair = _make_pair(rng, roles, count, overflow, rich, style,
                                  max_period, split == "train" and index == 1)
                key = _key(pair["input"])
                if split == "train" and index == 1:
                    old_volume = int((puzzle["train"][0]["output"] == roles[1]).sum())
                    if int((pair["output"] == roles[1]).sum()) == old_volume:
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _inspect_pair(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    scene, info = _read_input(inp)
    bg, water = info["background"], info["water"]
    terrain, supply = info["terrain"], info["supply"]
    assert inp.shape == out.shape
    assert np.array_equal(_settle(scene, bg, water), out), "incorrect water placement"
    levels = _closed_pools(out, bg, water)
    assert levels is not None, "partial level or leaking pool"
    assert np.array_equal(out != bg, (terrain != bg) | (out == water))
    assert np.array_equal(out[terrain != bg], terrain[terrain != bg]), "terrain was changed"
    assert int(supply.sum()) == int((out == water).sum()), "water volume was changed"
    assert np.argwhere(out == water)[:, 0].sum() > np.argwhere(supply)[:, 0].sum()
    assert not np.array_equal(scene, out), "gravity has no effect"
    visible = np.ones(inp.shape, dtype=bool)
    for p in info["mask"]:
        visible[p] = False
    rigid = visible & (terrain != bg)
    assert np.array_equal(inp[rigid], out[rigid])
    for rev in (False, True):
        for order in (False, True):
            assert np.array_equal(_settle(scene, bg, water, rev, order), out), "order-dependent allocation"
    omitted = terrain.copy()
    hidden_solids = hidden_openings = 0
    for p in info["mask"]:
        hidden_solids += int(terrain[p] != bg)
        omitted[p] = bg
    for v in info["vessels"]:
        hidden_openings += sum(c == bg and p in info["mask"] for p, c in v["assignments"].items())
    omitted[supply] = water
    assert hidden_solids >= 3 and hidden_openings >= 1
    assert not np.array_equal(_settle(omitted, bg, water) == water, out == water), "B has no hydraulic effect"
    for v in info["vessels"]:
        ablated = _remove_vessel(terrain, v["box"], bg)
        ablated[supply] = water
        assert not np.array_equal(_settle(ablated, bg, water) == water, out == water), "an unused vessel"
    # Return facts derived from pixels, not hidden generation descriptors.
    vessels = sorted(info["vessels"], key=lambda v: len(_area(v["box"])))
    inner = vessels[0]["box"]
    inside = np.zeros(inp.shape, dtype=bool)
    t, b, l, r = inner
    inside[t + 1:b, l + 1:r] = True
    # This certifies transfer out of the sole supplied chamber in sampled
    # scenes. Water in another separately supplied vessel is not overflow.
    overflow = bool(np.any((out == water) & ~inside)) and not bool(np.any(supply & ~inside))
    if overflow:
        sealed = terrain.copy()
        for v in vessels:
            for p, c in v["assignments"].items():
                if c == bg:
                    sealed[p] = v["base"]
        sealed[supply] = water
        assert not np.array_equal(_settle(sealed, bg, water) == water, out == water), "porosity is decorative"
    assert np.array_equal(solve(inp.copy()), out)
    return {"roles": (bg, water, info["frame"], info["veil"]), "shape": inp.shape,
            "vessels": len(vessels), "periods": tuple(v["period"] for v in vessels),
            "colors_per_vessel": tuple(len(v["colors"]) for v in vessels),
            "volume": int(supply.sum()), "levels": levels, "overflow": overflow,
            "source_components": len(_components(supply)),
            "retained_source_cells": int((supply & (out == water)).sum()),
            "lateral": bool(np.any(supply.sum(axis=0) != (out == water).sum(axis=0))),
            "shutter": info["shutter"], "boxes": tuple(v["box"] for v in vessels),
            "communicating": _communicating_scene(out, info)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen, "duplicate input"
            seen.add(key)
            stats[split].append(_inspect_pair(pair))
    all_stats = stats["train"] + stats["test"]
    assert len({s["roles"] for s in all_stats}) == 1, "inconsistent puzzle conventions"
    assert stats["train"][0]["vessels"] == 1 and not stats["train"][0]["overflow"]
    assert stats["train"][1]["vessels"] == 2 and stats["train"][1]["overflow"]
    assert stats["train"][0]["volume"] != stats["train"][1]["volume"]
    assert stats["train"][1]["source_components"] >= 2
    assert stats["train"][1]["communicating"], "communicating vessels were not demonstrated"
    assert max(stats["train"][1]["colors_per_vessel"]) >= 2
    assert all(s["vessels"] == 3 and s["overflow"] for s in stats["test"])
    demonstrated = max(stats["train"][1]["periods"])
    assert all(max(s["periods"]) <= demonstrated for s in stats["test"])
    return None


def self_test():
    """Bounded 42-seed diversity, ablation, schema, and deterministic checks."""
    signatures, shapes, layouts, roles, volumes, periods = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _inspect_pair(pair)
                records.append(s)
                roles.add(s["roles"])
                shapes.add(s["shape"])
                volumes.add(s["volume"])
                periods.update(s["periods"])
                layouts.add((s["shape"], s["boxes"], s["shutter"]))
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 40 and len(roles) >= 35 and len(volumes) >= 20
    assert periods >= {3, 4, 5}
    assert {s["vessels"] for s in records} == {1, 2, 3}
    assert sum(s["lateral"] for s in records) >= 100
    assert sum(len(s["levels"]) >= 2 for s in records) >= 50
    assert sum(s["source_components"] >= 2 for s in records) >= 42
    assert sum(s["retained_source_cells"] > 0 for s in records) >= 20
    # Nested pools can communicate through a submerged inner slot. Detect a
    # single water body with cells strictly inside and outside one inner loop.
    assert any(s["communicating"] for s in records)
    # Explicit examples outside the sampler's nested/period-3..5 core keep the
    # documented general rule honest. Their expected masks are direct complete
    # levels, not obtained by calling solve.
    for kind in ("period2", "period6", "separated", "mixed"):
        pair = _scope_fixture(kind)
        _inspect_pair(pair)
    for nt, ne in ((2, 1), (5, 4), (10, 4)):
        a, b = generate_puzzle(2026, nt, ne), generate_puzzle(2026, nt, ne)
        assert _signature(a) == _signature(b)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    _opening_source_fixture(bad["train"][1])
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")


def _communicating(pair):
    _, info = _read_input(pair["input"])
    out = np.asarray(pair["output"])
    return _communicating_scene(out, info)


def _communicating_scene(out, info):
    pools = _components(out == info["water"])
    for v in info["vessels"]:
        t, b, l, r = v["box"]
        inside = {(y, x) for y in range(t + 1, b) for x in range(l + 1, r)}
        area = _area(v["box"])
        if any(pool & inside and pool - area for pool in pools):
            return True
    return False


def _scope_fixture(kind):
    bg, water, frame, veil = roles = (0, 1, 8, 9)
    if kind == "period2":
        shape, boxes = (18, 18), [(2, 13, 3, 12)]
        words, bases = [(2, 0)], [2]
        source_rows = [4]
    elif kind == "period6":
        shape, boxes = (25, 20), [(2, 21, 3, 14)]
        words, bases = [(2, 3, 2, 2, 3, 0)], [2]
        source_rows = [4]
    elif kind == "separated":
        shape, boxes = (28, 30), [(2, 15, 2, 11), (4, 24, 16, 27)]
        words, bases = [(2, 0, 3), (4, 5, 4, 0, 5, 4)], [2, 4]
        source_rows = [4, 6]
    else:
        shape = (29, 30)
        boxes = [(5, 15, 5, 10), (2, 20, 2, 13), (3, 25, 18, 27)]
        words, bases = [(2, 0, 3), (4, 0, 5, 4), (6, 7, 6, 7, 0, 6)], [2, 4, 6]
        source_rows = [7, 4, 5]
    terrain = _vessel_grid(shape, boxes, words, bases, bg)
    source = np.zeros(shape, dtype=bool)
    target = terrain.copy()
    for (t, b, l, r), row in zip(boxes, source_rows):
        source[row, l + 1:r] = True
        target[b - 1, l + 1:r] = water
    inp = _shutter_input(np.random.default_rng(81), terrain, source, target,
                         boxes, list(map(len, words)), roles)
    assert inp is not None, "scope fixture's shutter was not reconstructable"
    return {"input": inp, "output": target}


def _opening_source_fixture(pair):
    """A source on an exposed opening must not corrupt word reconstruction."""
    inp, out = pair["input"].copy(), pair["output"]
    _, info = _read_input(inp)
    bg, water = info["background"], info["water"]
    remove = min(map(tuple, np.argwhere(info["supply"])))
    for v in info["vessels"]:
        for p, color in sorted(v["assignments"].items()):
            if color != bg or p in info["mask"] or inp[p] == water or out[p] != water:
                continue
            candidate = inp.copy()
            candidate[remove] = info["veil"] if remove in info["mask"] else bg
            candidate[p] = water
            try:
                _inspect_pair({"input": candidate, "output": out})
            except (AssertionError, ValueError):
                continue
            return
    raise AssertionError("could not exercise a water-overprinted visible slot")
