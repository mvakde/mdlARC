"""A locally demonstrated stamp evolution indexed by rooted network distance.

Only standard library and NumPy. No parent imports, example store, or solver
state. Frames and arrows encode all conventions needed by an individual input.
"""
from itertools import permutations
import json

import numpy as np


_ARROW = np.array([[0, 0, 1, 0], [1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool)
_DIRS = ((0, 1), (-1, 0), (0, -1), (1, 0))
_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _STEPS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.append(p)
                    stack.append(p)
        groups.append(sorted(cells))
    return groups


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _interior(grid, box):
    r, c, h, w = box
    return grid[r + 1:r + h - 1, c + 1:c + w - 1].copy()


def _geometry(a, g):
    if g < 4:
        return np.rot90(a, g).copy()
    if g == 4:
        return np.fliplr(a).copy()
    if g == 5:
        return np.flipud(a).copy()
    if g == 6:
        return a.T.copy()
    if g == 7:
        return a[::-1, ::-1].T.copy()
    raise ValueError("invalid symmetry")


def _color_map(source, target, bg):
    """Infer a complete bijection from corresponding cells, fixing background."""
    if source.shape != target.shape:
        return None
    mapping = {bg: bg}
    for x, y in zip(source.flat, target.flat):
        x, y = int(x), int(y)
        if (x == bg) != (y == bg) or (x in mapping and mapping[x] != y):
            return None
        mapping[x] = y
    if len(set(mapping.values())) != len(mapping):
        return None
    return mapping


def _recolor(a, mapping):
    lookup = np.arange(10, dtype=np.int8)
    for x, y in mapping.items():
        lookup[x] = y
    return lookup[a]


def _apply(a, operation):
    g, mapping = operation
    return _recolor(_geometry(a, g), mapping)


def _operation_candidates(before, after, bg):
    palette = set(map(int, np.unique(before))) | {bg}
    if palette != set(map(int, np.unique(after))) | {bg}:
        return []
    answers = []
    for g in range(8):
        moved = _geometry(before, g)
        mapping = _color_map(moved, after, bg)
        if mapping is not None and set(mapping) == palette and set(mapping.values()) == palette:
            answers.append((g, mapping))
    return answers


def _period(stamp, operation):
    a = stamp.copy()
    for n in range(1, 25):
        a = _apply(a, operation)
        if np.array_equal(a, stamp):
            return n
    raise ValueError("nonperiodic calibration")


def _distances(mask, roots):
    """Multi-source BFS; -1 denotes an inactive or unreachable cell."""
    levels = np.full(mask.shape, -1, dtype=int)
    queue = [tuple(map(int, p)) for p in np.argwhere(roots)]
    for p in queue:
        levels[p] = 0
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in _STEPS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                    and mask[nr, nc] and levels[nr, nc] < 0):
                levels[nr, nc] = levels[r, c] + 1
                queue.append((nr, nc))
    return levels


def _read_input(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    rim = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    if not np.all(rim == bg):
        raise ValueError("inputs have a uniform background rim")
    readings = []
    for frame in map(int, np.unique(grid)):
        if frame == bg:
            continue
        groups = _components(grid == frame)
        if len(groups) != 4:
            continue
        boxes = [_box(cells) for cells in groups]
        if any(h < 5 or w < 5 or len(cells) != 2 * h + 2 * w - 4
               or any(r < x < r + h - 1 and c < y < c + w - 1 for x, y in cells)
               for cells, (r, c, h, w) in zip(groups, boxes)):
            continue
        outside = grid.copy()
        for r, c, h, w in boxes:
            outside[r:r + h, c:c + w] = bg
        cells = np.argwhere(outside != bg)
        if not len(cells):
            continue
        ar, ac, ah, aw = _box([tuple(p) for p in cells])
        arrow = outside[ar:ar + ah, ac:ac + aw]
        signals = set(map(int, np.unique(arrow))) - {bg}
        if len(signals) != 1:
            continue
        node = next(iter(signals))
        turns = [k for k in range(4) if np.array_equal(arrow != bg, np.rot90(_ARROW, k))]
        if len(turns) != 1:
            continue
        dr, dc = _DIRS[turns[0]]
        before_boxes, after_boxes = [], []
        for b in boxes:
            r, c, h, w = b
            if dc:
                aligned = r <= ar and ar + ah <= r + h
                low = c + w + 1 == ac
                high = c == ac + aw + 1
            else:
                aligned = c <= ac and ac + aw <= c + w
                low = r + h + 1 == ar
                high = r == ar + ah + 1
            if aligned and (low if dr + dc > 0 else high):
                before_boxes.append(b)
            if aligned and (high if dr + dc > 0 else low):
                after_boxes.append(b)
        if len(before_boxes) != 1 or len(after_boxes) != 1:
            continue
        b0, b1 = before_boxes[0], after_boxes[0]
        if b0 == b1 or b0[2:] != b1[2:]:
            continue
        remaining = [b for b in boxes if b not in (b0, b1)]
        graph_boxes = [b for b in remaining if len(set(map(int, np.unique(_interior(grid, b)))) - {bg, node}) == 1]
        if len(graph_boxes) != 1:
            continue
        gb = graph_boxes[0]
        sb = next(b for b in remaining if b != gb)
        network = _interior(grid, gb)
        extra = set(map(int, np.unique(network))) - {bg, node}
        if len(extra) != 1:
            continue
        root = next(iter(extra))
        before, after, stamp = (_interior(grid, b) for b in (b0, b1, sb))
        if before.shape != stamp.shape or not all(3 <= n <= 6 for n in stamp.shape):
            continue
        if not all(3 <= n <= 10 for n in network.shape):
            continue
        payload = set(map(int, np.unique(before))) - {bg}
        target_payload = set(map(int, np.unique(stamp))) - {bg}
        if (not 3 <= len(payload) <= 4 or len(payload) != len(target_payload)
                or (payload | target_payload) & {frame, node, root}):
            continue
        role_map = _color_map(before, stamp, bg)
        choices = _operation_candidates(before, after, bg)
        if role_map is None or len(choices) != 1:
            continue
        operation = choices[0]
        period = _period(before, operation)
        if period < 3:
            continue
        # Conjugate the demonstrated color action through the visible role map.
        target_op = (operation[0], {role_map[x]: role_map[y] for x, y in operation[1].items()})
        mask = network != bg
        roots = network == root
        levels = _distances(mask, roots)
        if (not np.any(roots) or np.any(levels[mask] < 0)
                or np.any(np.asarray(mask.shape) * np.asarray(stamp.shape) > 30)):
            continue
        readings.append(dict(background=bg, frame=frame, node=node, root=root,
                             before=before, after=after, stamp=stamp, mask=mask, roots=roots,
                             levels=levels, operation=operation, target_op=target_op,
                             role_map=role_map, period=period, boxes=boxes,
                             before_box=b0, after_box=b1, stamp_box=sb, graph_box=gb,
                             arrow_turn=turns[0]))
    if len(readings) != 1:
        raise ValueError(f"expected one readable calibration/network, found {len(readings)}")
    return readings[0]


def _states(stamp, operation, maximum):
    states = [stamp.copy()]
    for _ in range(maximum):
        states.append(_apply(states[-1], operation))
    return np.stack(states)


def _prediction(info, levels=None):
    if levels is None:
        levels = info["levels"]
    stamp, mask = info["stamp"], info["mask"]
    levels = np.where(mask, levels, 0)
    sh, sw = stamp.shape
    states = _states(stamp, info["target_op"], int(levels[mask].max()))
    rr, cc = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    mr, mc = rr // sh, cc // sw
    return np.where(mask[mr, mc], states[levels[mr, mc], rr % sh, cc % sw],
                    info["background"]).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from the input's frames, arrow and root graph."""
    return _prediction(_read_input(input_grid))


def _frame(interior, color):
    h, w = interior.shape
    a = np.full((h + 2, w + 2), color, dtype=np.int8)
    a[1:-1, 1:-1] = interior
    return a


def _separated(a, b):
    r, c, h, w = a
    y, x, bh, bw = b
    return r + h < y or y + bh < r or c + w < x or x + bw < c


def _layout(rng, before, after, stamp, network, bg, frame, node):
    tiles = [_frame(a, frame) for a in (before, after, stamp, network)]
    ph, pw = tiles[0].shape
    calibration = np.full((ph, 2 * pw + 6), bg, dtype=np.int8)
    calibration[:, :pw] = tiles[0]
    calibration[:, pw + 6:] = tiles[1]
    arrow_top = (ph - 3) // 2
    patch = calibration[arrow_top:arrow_top + 3, pw + 1:pw + 5]
    patch[_ARROW] = node
    objects = [calibration, tiles[2], tiles[3]]
    locations = None
    for _ in range(90):
        h = int(rng.integers(max(17, ph + 3), 31))
        w = int(rng.integers(calibration.shape[1] + 2, 31))
        placed = {}
        order = [0, 1, 2]
        rng.shuffle(order)
        for i in order:
            oh, ow = objects[i].shape
            for _ in range(40):
                box = (int(rng.integers(1, h - oh)), int(rng.integers(1, w - ow)), oh, ow)
                if all(_separated(box, other) for other in placed.values()):
                    placed[i] = box
                    break
            else:
                break
        if len(placed) == 3:
            locations = placed
            break
    if locations is None:
        # Same rule, readable compact packing; at most 24 by 24 before padding.
        mh, mw = tiles[3].shape
        h = ph + max(ph, mh) + 3
        w = max(calibration.shape[1], pw + mw + 1) + 2
        locations = {0: (1, 1, *calibration.shape), 1: (ph + 2, 1, ph, pw),
                     2: (ph + 2, pw + 2, mh, mw)}
    grid = np.full((h, w), bg, dtype=np.int8)
    for i, (r, c, oh, ow) in locations.items():
        grid[r:r + oh, c:c + ow] = objects[i]
    return grid


def _motif(rng, shape, colors, bg, force_holes):
    h, w = shape
    rr, cc = np.indices(shape)
    k = len(colors)
    kind = int(rng.integers(6))
    if kind == 0:
        labels = rng.integers(k, size=shape)
    elif kind == 1:
        labels = (rr + int(rng.integers(1, k)) * cc) % k
    elif kind == 2:
        labels = np.minimum.reduce((rr, cc, h - 1 - rr, w - 1 - cc)) % k
    elif kind == 3:
        labels = (rr // 2 + cc // 2) % k
    elif kind == 4:
        labels = ((rr == h // 2).astype(int) + 2 * (cc == w // 2)) % k
    else:
        tile = rng.integers(k, size=(2, 2))
        labels = tile[rr % 2, cc % 2]
    # Local details and asymmetric gaps make the combined action identifiable.
    for index in rng.choice(h * w, int(rng.integers(2, min(6, h * w) + 1)), replace=False):
        labels.flat[index] = int(rng.integers(k))
    motif = np.asarray(colors, dtype=np.int8)[labels]
    count = int(rng.integers(1 if force_holes else 0, max(2, h * w // 4) + 1))
    if count:
        motif.flat[rng.choice(h * w, count, replace=False)] = bg
    missing = [c for c in colors if c not in motif]
    positions = list(map(int, rng.permutation(h * w)))
    for color in missing:
        # Do not remove the last example of an already visible role.
        for index in positions:
            old = int(motif.flat[index])
            if old == bg or np.count_nonzero(motif == old) > 1:
                motif.flat[index] = color
                positions.remove(index)
                break
    if force_holes and not np.any(motif == bg):
        for index in positions:
            if np.count_nonzero(motif == motif.flat[index]) > 1:
                motif.flat[index] = bg
                break
    return motif


def _calibration(rng, shape, ref_colors, target_colors, bg, holes):
    for _ in range(200):
        before = _motif(rng, shape, ref_colors, bg, holes)
        g = int(rng.choice([i for i in range(8) if _geometry(before, i).shape == shape]))
        shuffled = list(map(int, rng.permutation(ref_colors)))
        operation = (g, {bg: bg, **dict(zip(ref_colors, shuffled))})
        after = _apply(before, operation)
        if len(_operation_candidates(before, after, bg)) != 1 or _period(before, operation) < 3:
            continue
        role_map = {bg: bg, **dict(zip(ref_colors, target_colors))}
        stamp = _recolor(before, role_map)
        target_op = (g, {role_map[x]: role_map[y] for x, y in operation[1].items()})
        return before, after, stamp, target_op
    raise RuntimeError("could not construct identifiable calibration")


def _mask(rng, shape, kind):
    h, w = shape
    rr, cc = np.indices(shape)
    if kind == "u":
        mask = (cc == 0) | (cc == w - 1) | (rr == h - 1)
    elif kind == "bars":
        mask = (rr < 2) | (rr >= h - 2)
    elif kind == "ring":
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
    elif kind == "comb":
        mask = (cc == 0) | (rr % 2 == 0) | (rr == h - 1)
    elif kind == "ladder":
        mask = (cc == 0) | (cc == w - 1) | (rr % 2 == 0) | (rr == h - 1)
    elif kind == "snake":
        mask = rr % 2 == 0
        for r in range(1, h, 2):
            mask[r, w - 1 if (r // 2) % 2 == 0 else 0] = True
        if h % 2 == 0:
            mask[-1, :] = True
    elif kind == "tree":
        mask = np.zeros(shape, dtype=bool)
        r = c = 0
        mask[0, 0] = True
        while r < h - 1 or c < w - 1:
            if c == w - 1 or (r < h - 1 and rng.random() < .5):
                r += 1
            else:
                c += 1
            mask[r, c] = True
        for _ in range(h + w):
            fringe = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                      and sum(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                              for dr, dc in _STEPS) == 1]
            if fringe:
                mask[fringe[int(rng.integers(len(fringe)))]] = True
    elif kind == "maze":
        mask = np.zeros(shape, dtype=bool)
        rows = list(range(0, h, 2))
        cols = list(range(0, w, 2))
        remaining = {(r, c) for r in rows for c in cols}
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        mask[start] = True
        while stack:
            r, c = stack[-1]
            options = [(r + 2 * dr, c + 2 * dc) for dr, dc in _STEPS
                       if (r + 2 * dr, c + 2 * dc) in remaining]
            if not options:
                stack.pop()
                continue
            nr, nc = options[int(rng.integers(len(options)))]
            mask[(r + nr) // 2, (c + nc) // 2] = True
            mask[nr, nc] = True
            remaining.remove((nr, nc))
            stack.append((nr, nc))
        # Frames make bounds explicit, including unused border rows/columns.
    elif kind == "dense":
        mask = np.ones(shape, dtype=bool)
        for index in rng.permutation(h * w)[:int(rng.integers(1, max(2, h * w // 3)))]:
            p = divmod(int(index), w)
            mask[p] = False
            if len(_components(mask)) != 1:
                mask[p] = True
    else:
        raise ValueError("unknown network construction")
    return np.asarray(mask, dtype=bool)


def _roots(rng, mask, witness):
    h, w = mask.shape
    roots = np.zeros(mask.shape, dtype=bool)
    if witness == 1:
        roots[0, 0] = True
    elif witness == 2:
        roots[1, 0] = roots[1, w - 1] = True
        roots[h - 1, int(rng.integers(w))] = True
    else:
        for component in _components(mask):
            count = 1 if len(component) < 4 else int(rng.integers(1, min(3, len(component) // 3) + 1))
            for i in rng.choice(len(component), count, replace=False):
                roots[component[int(i)]] = True
    return roots


def _build_output(mask, roots, stamp, operation, bg):
    """Construct labels as explicit block writes; solve uses pixel indexing."""
    levels = _distances(mask, roots)
    states = _states(stamp, operation, int(levels.max()))
    sh, sw = stamp.shape
    output = np.full((mask.shape[0] * sh, mask.shape[1] * sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = states[levels[r, c]]
    return output


def _make_pair(rng, style, split, index, previous_stamp=None):
    regime = style["regime"]
    witness = index + 1 if split == "train" and index < 2 else 0
    for _ in range(100):
        stamp_low, stamp_high = (5, 6) if split == "test" and regime == "stamp_growth" else (3, 4)
        if regime == "new_topology":
            stamp_low, stamp_high = 3, 5
        sh, sw = map(int, rng.integers(stamp_low, stamp_high + 1, size=2))
        if witness == 2 and (sh, sw) == previous_stamp:
            continue
        high_h, high_w = min(6, 30 // sh), min(6, 30 // sw)
        if split == "test" and regime == "graph_growth":
            mh = int(rng.integers(7, min(10, 30 // sh) + 1))
            mw = int(rng.integers(7, min(10, 30 // sw) + 1))
        else:
            mh = 6 if witness == 2 else int(rng.integers(3, high_h + 1))
            widths = [n for n in range(3, high_w + 1) if witness != 2 or n % 2]
            mw = int(rng.choice(widths))
        if witness:
            kind = "u" if witness == 1 else "bars"
        elif split == "train" and regime == "new_topology":
            kind = str(rng.choice(["u", "bars", "comb", "tree", "snake"]))
            if kind == "bars" and mh < 5:
                kind = "comb"
        elif split == "test" and regime == "new_topology":
            kind = str(rng.choice(["maze", "ring", "ladder", "dense"]))
        else:
            kind = str(rng.choice(["u", "ring", "comb", "tree", "snake", "maze", "ladder", "dense"]))
        mask = _mask(rng, (mh, mw), kind)
        if witness == 2:
            # One blank border row plus the interior separator. This explicitly
            # distinguishes frame extent from a tight active-cell crop.
            mask[0, :] = False
            mask[2, :] = True
        roots = _roots(rng, mask, witness)
        for axis in (0, 1):
            if rng.integers(2):
                mask, roots = np.flip(mask, axis).copy(), np.flip(roots, axis).copy()
        levels = _distances(mask, roots)
        if levels.max() < 2:
            continue
        palette = style["palette"] if style["stable"] else list(map(int, rng.permutation(10)))
        bg, frame, node, root = palette[:4]
        k = int(rng.integers(3, 5))
        ref_colors = palette[4:4 + k]
        target_colors = list(map(int, rng.choice(palette[4:], k, replace=False)))
        # The first witness always exposes color-role transfer too.
        if witness == 1 and target_colors == ref_colors:
            target_colors = target_colors[1:] + target_colors[:1]
        before, after, stamp, target_op = _calibration(
            rng, (sh, sw), ref_colors, target_colors, bg, holes=bool(witness or rng.integers(2)))
        network = np.where(roots, root, np.where(mask, node, bg)).astype(np.int8)
        grid = _layout(rng, before, after, stamp, network, bg, frame, node)
        output = _build_output(mask, roots, stamp, target_op, bg)
        turns = int(rng.integers(4))
        grid, output = np.rot90(grid, turns).copy(), np.rot90(output, turns).copy()
        # These checks concern construction readability, never an unknown label.
        info = _read_input(grid)
        if witness == 2 and info["stamp"].shape == previous_stamp:
            continue
        if witness == 1 and np.array_equal(_prediction(info), _prediction(info, _manhattan(info))):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct an informative pair")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    a, b = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    style = dict(regime=str(rng.choice(["graph_growth", "stamp_growth", "new_topology"])),
                 palette=list(map(int, rng.permutation(10))), stable=bool(rng.integers(2)))
    puzzle, seen = {"train": [], "test": []}, set()
    first_stamp = None
    for split, count in (("train", a), ("test", b)):
        for index in range(count):
            for _ in range(32):
                pair = _make_pair(rng, style, split, index, first_stamp)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        # Displayed shape, for independent fine-dimension variation.
                        info = _read_input(pair["input"])
                        first_stamp = info["stamp"].shape
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _manhattan(info):
    rr, cc = np.indices(info["mask"].shape)
    return np.minimum.reduce([abs(rr - r) + abs(cc - c) for r, c in np.argwhere(info["roots"])])


def _static_stamping(info):
    return _prediction(info, np.zeros(info["mask"].shape, dtype=int))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _read_input(inp)
            mask, roots, levels = info["mask"], info["roots"], info["levels"]
            assert np.array_equal(levels == 0, roots)
            assert np.all(levels[mask] >= 0) and np.all(levels[~mask] == -1)
            assert np.array_equal(output, _prediction(info)), "incorrect evolved expansion"
            sh, sw = info["stamp"].shape
            assert output.shape == (mask.shape[0] * sh, mask.shape[1] * sw)
            assert _period(info["stamp"], info["target_op"]) == info["period"]
            for r, c in np.argwhere(mask):
                neighbors = [(r + dr, c + dc) for dr, dc in _STEPS
                             if 0 <= r + dr < mask.shape[0] and 0 <= c + dc < mask.shape[1]
                             and mask[r + dr, c + dc]]
                assert all(abs(levels[r, c] - levels[p]) <= 1 for p in neighbors)
                assert roots[r, c] or any(levels[p] == levels[r, c] - 1 for p in neighbors)
                block = output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw]
                if roots[r, c]:
                    assert np.array_equal(block, info["stamp"]), "root must reset to initial state"
            for r, c in np.argwhere(~mask):
                assert np.all(output[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] == info["background"])
            assert not np.array_equal(output, _static_stamping(info)), "calibration must affect the answer"
            assert not ({info["frame"], info["node"], info["root"]} & set(map(int, np.unique(output))))
            if split == "train":
                training.append(info)
    assert any(np.any(~i["mask"]) and np.any(i["stamp"] == i["background"]) for i in training)
    assert any(i["roots"].sum() == 1 and i["levels"].max() >= 3
               and not np.array_equal(_prediction(i), _prediction(i, _manhattan(i))) for i in training), "detour witness required"
    assert any(len(_components(i["mask"])) >= 2 and i["roots"].sum() >= 3 for i in training), "multiple sources/components witness required"
    assert any(not all(np.any(side) for side in (i["mask"][0], i["mask"][-1],
                                                i["mask"][:, 0], i["mask"][:, -1]))
               for i in training), "frame extent must differ from a tight active crop"
    assert any(not np.array_equal(_prediction(i), _prediction(i, np.minimum(i["levels"], 1)))
               for i in training), "demonstrate repeated rather than one-time transformation"
    assert any(any(x != y for x, y in i["role_map"].items()) for i in training), "demonstrate palette transfer"
    assert len({i["stamp"].shape for i in training}) >= 2, "vary fine dimensions"


def _twins_test(pair):
    """Keep seed/network/layout fixed; only change the reference partner."""
    original = np.asarray(pair["input"]).copy()
    info = _read_input(original)
    before, bg = info["before"], info["background"]
    colors = sorted(set(map(int, np.unique(before))) - {bg})
    for g in range(8):
        if _geometry(before, g).shape != before.shape:
            continue
        for ordered in permutations(colors):
            operation = (g, {bg: bg, **dict(zip(colors, ordered))})
            after = _apply(before, operation)
            if (len(_operation_candidates(before, after, bg)) != 1 or _period(before, operation) < 3
                    or np.array_equal(after, info["after"])):
                continue
            twin = original.copy()
            r, c, h, w = info["after_box"]
            twin[r + 1:r + h - 1, c + 1:c + w - 1] = after
            result = solve(twin)
            assert not np.array_equal(result, pair["output"])
            twin_info = _read_input(twin)
            assert np.array_equal(twin_info["stamp"], info["stamp"])
            assert np.array_equal(twin_info["mask"], info["mask"])
            assert np.array_equal(twin_info["roots"], info["roots"])
            return
    raise AssertionError("failed to construct a reference-dependent twin")


def _route_test(pair):
    """Add routes without changing roots/calibration; inspect retained blocks."""
    original = np.asarray(pair["input"])
    info = _read_input(original)
    assert info["roots"].sum() == 1
    changed = original.copy()
    r, c, h, w = info["graph_box"]
    network = changed[r + 1:r + h - 1, c + 1:c + w - 1]
    network[network == info["background"]] = info["node"]
    result = solve(changed)
    sh, sw = info["stamp"].shape
    assert any(not np.array_equal(result[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw],
                                 pair["output"][y * sh:(y + 1) * sh, x * sw:(x + 1) * sw])
               for y, x in np.argwhere(info["mask"])), "route changes must affect existing placements"
    # The parser also handles the legal limiting case of a root at every node,
    # although generation avoids this case because it hides the calibration.
    network[:] = info["root"]
    assert np.array_equal(solve(changed), np.tile(info["stamp"], info["mask"].shape))


def self_test():
    shapes, masks, stamps, operations, palettes, periods, arrows, maxima = (set() for _ in range(8))
    puzzle_keys, pairs = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training = []
        encoded = tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s])
        puzzle_keys.add(encoded)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                if split == "train":
                    training.append((pair, info))
                shapes.add((pair["input"].shape, pair["output"].shape))
                masks.add(_key(info["mask"]))
                normalized = _recolor(info["stamp"], {info["background"]: 0, **{
                    color: i + 1 for i, color in enumerate(sorted(set(map(int, np.unique(info["stamp"]))) - {info["background"]}))}})
                stamps.add(_key(normalized))
                operations.add(info["operation"][0])
                palettes.add((info["background"], info["frame"], info["node"], info["root"]))
                periods.add(info["period"])
                arrows.add(info["arrow_turn"])
                maxima.add(int(info["levels"].max()))
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solver must not modify input"
                pairs += 1
        if seed < 3:
            single = next(p for p, i in training if i["roots"].sum() == 1
                          and not np.array_equal(_prediction(i), _prediction(i, _manhattan(i))))
            _twins_test(single)
            _route_test(single)
    assert len(puzzle_keys) == 42
    assert len(shapes) >= 80 and len(masks) >= 70 and len(stamps) >= 150
    assert len(operations) >= 6 and len(palettes) >= 30 and {3, 4, 6} <= periods
    assert arrows == {0, 1, 2, 3} and max(maxima) >= 12
    p = generate_puzzle(147412025, 2, 1)
    q = generate_puzzle(147412025, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    # All count combinations, with a bounded total of 270 pairs.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(385130886, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzle_keys),
            "shape_pairs": len(shapes), "network_patterns": len(masks), "normalized_stamps": len(stamps),
            "geometries": sorted(operations), "periods": sorted(periods), "arrow_directions": sorted(arrows),
            "largest_graph_distance": max(maxima), "independent_verification": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
