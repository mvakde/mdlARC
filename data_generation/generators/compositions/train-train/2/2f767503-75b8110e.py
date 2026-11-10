"""One-pass peeling of visible whole objects in registered transparent sheets.

The strip order is the depth order.  A sweep samples each forward coordinate
once, deletes its current frontmost four-connected component, and moves on.
Only standard-library Python and NumPy are used.
"""

from itertools import permutations

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_VARIANTS = ("no_erase", "all_layers", "snapshot", "recursive", "pixels",
             "eight", "wide", "first", "reverse_scan", "backward", "flat")


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _neighbors(p, shape, eight=False):
    r, c = p
    steps = ((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if dr or dc) if eight else _DIR4
    for dr, dc in steps:
        q = r + dr, c + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _groups(mask, eight=False):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop(), mask.shape, eight):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _bar_options(panel, background):
    """Find geometric controls, without assigning literal color meanings."""
    options = []
    for bar in map(int, np.unique(panel)):
        if bar == background:
            continue
        points = [tuple(map(int, p)) for p in np.argwhere(panel == bar)]
        if len(points) != 3:
            continue
        rows = sorted(p[0] for p in points)
        cols = sorted(p[1] for p in points)
        if len(set(rows)) == 1 and cols == list(range(cols[0], cols[0] + 3)):
            mid, directions = (rows[0], cols[1]), ((-1, 0), (1, 0))
        elif len(set(cols)) == 1 and rows == list(range(rows[0], rows[0] + 3)):
            mid, directions = (rows[1], cols[0]), ((0, -1), (0, 1))
        else:
            continue
        for dr, dc in directions:
            p = mid[0] - dr, mid[1] - dc
            if not (0 <= p[0] < panel.shape[0] and 0 <= p[1] < panel.shape[1]):
                continue
            mark = int(panel[p])
            if mark not in (background, bar) and np.count_nonzero(panel == mark) == 1:
                options.append((bar, mark, mid, (dr, dc), set(points) | {p}))
    return options


def _parse(input_grid):
    grid = _grid(input_grid)
    partitions = []
    # The separator color occurs ONLY on complete separator lines.  This
    # makes even a single separator between two sheets an explicit cue.
    for axis in (0, 1):
        for n in range(2, 5):
            span = grid.shape[axis] - (n - 1)
            if span <= 0 or span % n:
                continue
            size = span // n
            cuts = [i * (size + 1) - 1 for i in range(1, n)]
            lines = np.take(grid, cuts, axis=axis)
            sep = int(lines.flat[0])
            if not np.all(lines == sep):
                continue
            expected = np.zeros(grid.shape, dtype=bool)
            if axis == 0:
                expected[cuts, :] = True
            else:
                expected[:, cuts] = True
            if not np.array_equal(grid == sep, expected):
                continue
            panels = [np.take(grid, range(i * (size + 1), i * (size + 1) + size), axis=axis)
                      for i in range(n)]
            common = set(map(int, np.unique(panels[0])))
            for panel in panels[1:]:
                common &= set(map(int, np.unique(panel)))
            if len(common) != 1:
                continue
            bg = next(iter(common))
            options = _bar_options(panels[0], bg)
            if len(options) != 1:
                continue
            bar, mark, mid, direction, controls = options[0]
            colors = []
            valid = True
            for k, panel in enumerate(panels):
                values = set(map(int, np.unique(panel))) - {bg}
                if k == 0:
                    values -= {bar, mark}
                elif values & {bar, mark}:
                    valid = False
                if len(values) > 1:
                    valid = False
                colors.append(next(iter(values)) if values else None)
            present = [c for c in colors if c is not None]
            if (not valid or len(present) != len(set(present))
                    or set(present) & {sep, bg, bar, mark}):
                continue
            # Controls belong to the tool, never to an object's hidden mask.
            if any(any(int(p[pnt]) != bg for pnt in controls) for p in panels[1:]):
                continue
            dr, dc = direction
            ray, p = [], (mid[0] + dr, mid[1] + dc)
            while 0 <= p[0] < panels[0].shape[0] and 0 <= p[1] < panels[0].shape[1]:
                ray.append(p)
                p = p[0] + dr, p[1] + dc
            if not ray:
                continue
            masks = np.stack([p == color if color is not None else np.zeros(p.shape, bool)
                              for p, color in zip(panels, colors)])
            partitions.append(dict(grid=grid, axis=axis, n=n, sep=sep, bg=bg,
                                   bar=bar, mark=mark, mid=mid, direction=direction,
                                   controls=controls, panels=panels, colors=tuple(colors),
                                   masks=masks, shape=panels[0].shape, ray=ray))
    _require(len(partitions) == 1, "input must have one unambiguous sheet/control interpretation")
    return partitions[0]


def _labels(info, eight=False):
    maps = np.zeros(info["masks"].shape, dtype=np.int32)
    objects = {}
    for k, mask in enumerate(info["masks"]):
        for cells in _groups(mask, eight):
            ident = len(objects) + 1
            objects[ident] = (k, cells)
            for p in cells:
                maps[k][p] = ident
    return maps, objects


def _render_maps(info, maps, removed=(), order=None):
    order = tuple(range(info["n"])) if order is None else tuple(order)
    out = np.full(info["shape"], info["bg"], dtype=np.int8)
    for k in reversed(order):
        occupied = maps[k] != 0
        if removed:
            occupied &= ~np.isin(maps[k], tuple(removed))
        if info["colors"][k] is not None:
            out[occupied] = info["colors"][k]
    for p in info["controls"]:
        out[p] = info["panels"][0][p]
    return out


def _sweep(info, variant="normal", order=None):
    maps, objects = _labels(info, eight=variant == "eight")
    order = tuple(range(info["n"])) if order is None else tuple(order)
    removed, trace = set(), []
    path = list(info["ray"])
    if variant == "reverse_scan":
        path.reverse()
    elif variant == "backward":
        dr, dc = info["direction"]
        p = info["mid"][0] - 2 * dr, info["mid"][1] - 2 * dc
        while 0 <= p[0] < info["shape"][0] and 0 <= p[1] < info["shape"][1]:
            path.append(p)
            p = p[0] - dr, p[1] - dc
    elif variant == "wide":
        dr, dc = info["direction"]
        path = [(r + s * dc, c - s * dr) for r, c in path for s in (0, -1, 1)
                if 0 <= r + s * dc < info["shape"][0]
                and 0 <= c - s * dr < info["shape"][1]]

    if variant == "no_erase":
        return _render_maps(info, maps, order=order), removed, trace, objects
    if variant == "flat":
        flat = _render_maps(info, maps, order=order)
        for p in path:
            color = int(flat[p])
            if color in (info["bg"], info["bar"], info["mark"]):
                continue
            group = next(g for g in _groups(flat == color) if p in g)
            for q in group:
                flat[q] = info["bg"]
        return flat, removed, trace, objects

    for p in path:
        candidates = [int(maps[k][p]) for k in order
                      if maps[k][p] and (variant == "snapshot" or int(maps[k][p]) not in removed)]
        if not candidates:
            continue
        chosen = candidates if variant in ("all_layers", "recursive") else candidates[:1]
        for ident in chosen:
            trace.append((p, objects[ident][0], ident))
            if variant == "pixels":
                maps[objects[ident][0]][p] = 0
            else:
                removed.add(ident)
        if variant == "first":
            break
    return _render_maps(info, maps, removed, order), removed, trace, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all conventions from the input; no generation state is consulted."""
    return _sweep(_parse(input_grid))[0]


def _direct_reference(info):
    """Construction check: erase a freshly flood-filled current mask per step.

    Unlike solve, this routine has no precomputed component labels or ID set.
    It is an authored cross-check, not an independent semantic oracle.
    """
    masks = info["masks"].copy()
    for p in info["ray"]:
        for mask in masks:
            if not mask[p]:
                continue
            mask[p] = False
            stack = [p]
            while stack:
                for q in _neighbors(stack.pop(), info["shape"]):
                    if mask[q]:
                        mask[q] = False
                        stack.append(q)
            break
    return _render_maps(info, masks.astype(np.int32))


def _grow(rng, core, size, shape, forbidden, style):
    cells = set(core)
    anchor = min(cells)
    axis = int(rng.integers(2))
    while len(cells) < size:
        candidates = sorted({q for p in cells for q in _neighbors(p, shape)} - cells - forbidden)
        if not candidates:
            break
        weights = []
        for q in candidates:
            adj = sum(p in cells for p in _neighbors(q, shape))
            if style == 0:
                weight = 9 if q[axis] == anchor[axis] else 0.3
            elif style == 1:
                weight = 1 + 5 * adj
            elif style == 2:
                weight = 7 if adj == 1 else 0.5
            else:
                weight = 1
            weights.append(weight)
        weights = np.asarray(weights, float)
        cells.add(candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))])
    return cells


def _witness_masks(rng, n, shape, mid, kind):
    """Two compact causal constructions; all later pairs use free objects."""
    h, w = shape
    r, c = mid
    side = int(rng.choice((-1, 1)))
    def at(t, p=0):
        return r + side * p, c + t
    a = int(rng.integers(2, 4))
    b = a + int(rng.integers(3, 5))
    late = min(w - c - 1, b + int(rng.integers(3, 5)))
    masks = np.zeros((n, h, w), bool)
    if kind == 0:
        # Removing this front arch at a uncovers the later contact at b.
        front = {at(a), at(b)} | {at(t, 1) for t in range(a, b + 1)}
        groups = [(0, front), (0, {at(b - 1, -1)}),
                  (0, {at(-2)}), (0, {at(late), at(late, 1)}),
                  (1, {at(a), at(a, -1)}), (1, {at(b), at(b, -1)})]
        # A protected lower singleton is left behind at the second peel.
        for k in range(2, n):
            groups.append((k, {at(b), at(b, -1 - (k % 2))}))
    else:
        # A lower arch passes under a preserved off-ray front occluder.
        lower = {at(a), at(b)} | {at(t, -1) for t in range(a, b + 1)}
        groups = [(0, {at(a)}), (0, {at(a + 1, 1)}), (0, {at(-2)}),
                  (0, {at(a + 2, -1)}), (0, {at(late), at(late, 1)}),
                  (1, lower), (1, {at(late), at(late, -1)})]
        for k in range(2, n):
            # Each deeper sheet is exposed when the shallower arch vanishes.
            groups.append((k, {at(a), at(a, 1)}))
    for k, cells in groups:
        for p in cells:
            masks[k][p] = True

    # Grow selected witness objects sideways without changing their ray
    # intersections, diagonal separation, or the hidden bridge.
    band = {at(t, p) for t in range(-c, w - c) for p in (-1, 0, 1)}
    for k in range(n):
        initial = _groups(masks[k])
        for group in initial:
            if rng.random() < .55:
                others = {tuple(map(int, p)) for p in np.argwhere(masks[k])} - group
                blocked = band - group
                blocked |= {q for p in others for q in _neighbors(p, shape)} | others
                grown = _grow(rng, group, len(group) + int(rng.integers(1, 7)),
                              shape, blocked, int(rng.integers(4)))
                for p in grown:
                    masks[k][p] = True
        # Unrelated islands outside the teaching band add varied preserved
        # structure; they cannot accidentally connect to its objects.
        for _ in range(int(rng.integers(0, 4))):
            occupied = {tuple(map(int, p)) for p in np.argwhere(masks[k])}
            blocked = band | occupied | {q for p in occupied for q in _neighbors(p, shape)}
            available = sorted(set(np.ndindex(shape)) - blocked)
            if not available:
                break
            p = available[int(rng.integers(len(available)))]
            grown = _grow(rng, {p}, int(rng.integers(1, 9)), shape, blocked,
                          int(rng.integers(4)))
            for p in grown:
                masks[k][p] = True
    if kind == 1:
        # Small untouched overlaps reveal every adjacent depth comparison.
        # They sit behind the tool, off its row, with a one-cell moat.
        positions = ((0, 0), (h - 1, 0), (0, 2))[:n - 1]
        for p in positions:
            clear = {p} | set(_neighbors(p, shape))
            for q in clear:
                masks[:, q[0], q[1]] = False
        for k, p in enumerate(positions):
            masks[k, p[0], p[1]] = True
            masks[k + 1, p[0], p[1]] = True
    return masks


def _random_masks(rng, n, shape, mid, controls):
    h, w = shape
    masks = np.zeros((n, h, w), bool)
    density_mode = int(rng.integers(3))
    for k in range(n):
        count = int(rng.integers(2, (5, 8, 12)[density_mode]))
        for j in range(count):
            if j == 0 or rng.random() < .45:
                p = mid[0], int(rng.integers(mid[1] + 1, w))
            else:
                p = int(rng.integers(h)), int(rng.integers(w))
            if p in controls:
                continue
            size = int(rng.choice((1, 2, 3, 5, 8, 13, 21, 30)))
            cells = _grow(rng, {p}, size, shape, controls, int(rng.integers(4)))
            for q in cells:
                masks[k][q] = True
        # Independently aligned shapes create substantial and partial
        # occlusion without prescribing a table of joint pixel signatures.
        if k and rng.random() < .40:
            earlier = int(rng.integers(k))
            shared = rng.random(shape) < rng.uniform(.45, .9)
            masks[k] |= masks[earlier] & shared
    if n >= 3 and rng.random() < .17:
        masks[int(rng.integers(1, n))] = False
    return masks


def _pack(masks, mid, rotation, palette):
    n, h, w = masks.shape
    bg, sep, bar, mark = palette[:4]
    colors = palette[4:4 + n]
    panels = [np.where(mask, color, bg).astype(np.int8)
              for mask, color in zip(masks, colors)]
    r, c = mid
    panels[0][r - 1:r + 2, c] = bar
    panels[0][r, c - 1] = mark
    panels = [np.rot90(panel, rotation) for panel in panels]
    # Depth increases downward for horizontal strips, rightward for vertical
    # strips, regardless of the direction in which the local tool points.
    axis = 0 if rotation % 2 == 0 else 1
    separator_shape = list(panels[0].shape)
    separator_shape[axis] = 1
    sep_panel = np.full(separator_shape, sep, dtype=np.int8)
    parts = []
    for k, panel in enumerate(panels):
        if k:
            parts.append(sep_panel)
        parts.append(panel)
    return np.concatenate(parts, axis=axis)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _differences(info, output):
    return {variant: not np.array_equal(_sweep(info, variant)[0], output)
            for variant in _VARIANTS}


def _make_pair(rng, palette, rotation, n, kind, test):
    for _ in range(160):
        maximum_short = (31 - n) // n
        h = int(rng.integers(5, min(maximum_short, 12 if test else 8) + 1))
        w = int(rng.integers(19, 31) if test else rng.integers(13, 19))
        r = int(rng.integers(2, h - 2)) if h > 5 else 2
        c = 2 if kind is not None else int(rng.integers(1, max(2, w // 3)))
        mid = r, c
        controls = {(r - 1, c), (r, c), (r + 1, c), (r, c - 1)}
        masks = (_witness_masks(rng, n, (h, w), mid, kind) if kind is not None
                 else _random_masks(rng, n, (h, w), mid, controls))
        for p in controls:
            masks[:, p[0], p[1]] = False
        grid = _pack(masks, mid, rotation, palette)
        try:
            info = _parse(grid)
        except AssertionError:
            continue
        output = _direct_reference(info)
        # Both parent mechanisms affect EVERY generated answer.  Free scenes
        # are not forced to have every other teaching feature.
        if any(np.array_equal(_sweep(info, v)[0], output) for v in ("no_erase", "all_layers")):
            continue
        if kind == 0:
            differences = _differences(info, output)
            if not all(differences[v] for v in ("snapshot", "recursive", "pixels", "eight", "wide",
                                                "first", "reverse_scan", "backward", "flat")):
                continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.permutation(10)[:8]))
    changing_palette = bool(rng.integers(2))
    base_rotation = int(rng.integers(4))
    orientation_mode = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    train_rotations = []
    for index in range(nt + ns):
        test = index >= nt
        kind = index if index < 2 else None
        # Two front sheets already teach the full interaction; a second
        # demonstration with 3/4 sheets teaches repetition of the same depth
        # convention rather than a fixed number of layers.
        n = 2 if index == 0 else (int(rng.integers(3, 5)) if index == 1
                                  else int(rng.integers(2, 5)))
        if orientation_mode == 0:
            rotation = base_rotation
        elif not test:
            rotation = (base_rotation + (index % 2)) % 4
        elif orientation_mode == 1:
            rotation = train_rotations[int(rng.integers(len(train_rotations)))]
        else:
            rotation = (base_rotation + 2 + ((index - nt) % 2)) % 4
        if not test:
            train_rotations.append(rotation)
        palette = tuple(map(int, rng.permutation(10)[:8])) if changing_palette and index else base_palette
        for _ in range(10):
            pair = _make_pair(rng, palette, rotation, n, kind, test)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("duplicate construction exhausted")
        seen.add(key)
        puzzle["test" if test else "train"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, demonstrated, train_infos = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            expected, removed, trace, objects = _sweep(info)
            _require(output.shape == info["shape"] and np.array_equal(output, expected), "wrong output")
            _require(np.array_equal(output, _direct_reference(info)), "flood-fill/label disagreement")
            _require(len(removed) == len(trace) and removed, "each object is erased once")
            _require(all(len([step for step in trace if step[0] == p]) <= 1 for p in info["ray"]),
                     "one deletion at most per sampled coordinate")
            _require(all(output[p] == info["panels"][0][p] for p in info["controls"]), "tool changed")
            # Reconstruct surviving masks and check full object deletion and
            # coordinate registration.  Hidden cells are included in objects.
            masks = info["masks"].copy()
            for ident, (k, cells) in objects.items():
                if ident in removed:
                    for p in cells:
                        masks[k][p] = False
            allowed = np.zeros(output.shape, bool)
            allowed |= output == info["bg"]
            for k, color in enumerate(info["colors"]):
                if color is not None:
                    allowed |= (output == color) & masks[k]
            for p in info["controls"]:
                allowed[p] = True
            _require(np.all(allowed), "output has invented or shifted cells")
            differences = _differences(info, output)
            _require(differences["no_erase"] and differences["all_layers"], "a parent is inactive")
            if split == "train":
                demonstrated.update(v for v, differs in differences.items() if differs)
                train_infos.append(info)
    _require(set(_VARIANTS) <= demonstrated, "training leaves a tested shortcut unresolved")
    # Check every depth permutation for the larger teaching example.  The
    # displayed order is a fixed family convention, and demonstrations must
    # actually distinguish it from a different priority assignment.
    info = train_infos[1]
    output = np.asarray(puzzle["train"][1]["output"])
    orders = [order for order in permutations(range(info["n"]))
              if np.array_equal(_sweep(info, order=order)[0], output)]
    _require(orders == [tuple(range(info["n"]))], "training does not identify sheet depth order")


def _puzzle_key(puzzle):
    return tuple((s, _key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _hand_checks():
    """Two literal expected survivors test the temporal and hidden-path rules."""
    shape, mid = (5, 12), (2, 1)
    for kind in (0, 1):
        masks = np.zeros((3, *shape), bool)
        if kind == 0:
            front = {(2, 3), (2, 6)} | {(3, c) for c in range(3, 7)}
            front |= {(1, 5)}  # diagonal neighbor of the erased front arch
            middle = {(1, 3), (2, 3), (1, 6), (2, 6)}
            survivors = ((0, {(1, 5)}), (1, {(1, 3), (2, 3)}), (2, {(2, 6)}))
        else:
            front = {(2, 3), (1, 5)}
            middle = {(2, 3), (2, 7)} | {(1, c) for c in range(3, 8)}
            survivors = ((0, {(1, 5)}), (2, {(2, 7)}))
        bottom = {(2, 6 if kind == 0 else 7)}
        for k, cells in enumerate((front, middle, bottom)):
            for p in cells:
                masks[k][p] = True
        for palette in ((0, 9, 8, 7, 1, 2, 3, 4), (6, 1, 4, 0, 9, 7, 2, 5)):
            expected = np.full(shape, palette[0], dtype=np.int8)
            for k, cells in survivors:
                for p in cells:
                    expected[p] = palette[4 + k]
            expected[1:4, 1] = palette[2]
            expected[2, 0] = palette[3]
            for rotation in range(4):
                inp = _pack(masks, mid, rotation, palette)
                before = inp.copy()
                _require(np.array_equal(solve(inp), np.rot90(expected, rotation)),
                         "literal one-pass/hidden-bridge fixture")
                _require(np.array_equal(inp, before), "solve mutated its input")


def self_test():
    """Bounded 42-seed semantic/diversity audit plus count boundary checks."""
    _hand_checks()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    features = []
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                _, removed, trace, objects = _sweep(info)
                features.append((info, removed, trace, objects, split))
    _require({f[0]["n"] for f in features} == {2, 3, 4}, "sheet count diversity")
    _require({f[0]["direction"] for f in features} == set(_DIR4), "cardinal directions")
    _require({f[0]["axis"] for f in features} == {0, 1}, "strip orientations")
    _require(len({f[0]["shape"] for f in features}) >= 35, "panel geometry diversity")
    _require(len({len(f[3]) for f in features}) >= 12, "object count diversity")
    sizes = {len(cells) for _, _, _, objects, _ in features for _, cells in objects.values()}
    _require({1, 2} <= sizes and max(sizes) >= 20, "object size diversity")
    _require(any(len({k for _, k, _ in trace}) >= 3 for _, _, trace, _, _ in features),
             "sweep must reach deeper sheets")
    _require(any(any(color is None for color in info["colors"]) for info, *_ in features), "blank sheets")
    _require(len({info["bg"] for info, *_ in features}) >= 8, "background palette diversity")
    _require(any(30 in info["grid"].shape for info, *_ in features), "boundary dimension")
    palette_changes = []
    for puzzle in puzzles:
        cores = []
        for pair in puzzle["train"][:2]:
            info = _parse(pair["input"])
            cores.append((info["bg"], info["sep"], info["bar"], info["mark"], *info["colors"][:2]))
        palette_changes.append(cores[0] != cores[1])
    _require(any(palette_changes) and not all(palette_changes),
             "both changing and constant literal role palettes")
    for puzzle in puzzles:
        train = [_parse(p["input"]) for p in puzzle["train"]]
        test = [_parse(p["input"]) for p in puzzle["test"]]
        _require(max(max(i["shape"]) for i in train) < min(max(i["shape"]) for i in test),
                 "longer-sheet train/test shift")
    _require(_puzzle_key(generate_puzzle(912, 5, 2)) == _puzzle_key(generate_puzzle(912, 5, 2)),
             "determinism")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(500 + 10 * nt + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal count combinations)")
