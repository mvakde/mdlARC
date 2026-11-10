"""Joint assembly with separately conserved carrier scales and motif states.

Only Python's standard library and NumPy are used. No files are read or written.
Searches and construction retries are bounded; there is no persistent cache.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json
import numpy as np


HOST_SEEDS = (0, 874509763, 563832252, 958629612, 247361653, 800219165,
              390393114, 259980586, 562268601, 510472826, 479844948,
              215050325, 500366761, 796030993, 107138954, 778277032,
              925173466, 992114439, 956735416, 789944438, 512465487,
              990639314, 822662924, 528046868, 191460399, 178513699,
              440834868, 772081920, 692517695, 709298059, 562572473,
              99253536, 396696060, 274468049, 689316807, 901151766,
              694868157, 640503367, 405401170, 264815955, 434196382,
              233358361)


class _SearchLimit(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bounds(points):
    return (min(y for y, x in points), min(x for y, x in points),
            max(y for y, x in points) + 1, max(x for y, x in points) + 1)


def _normalize(mapping):
    y0, x0, _, _ = _bounds(mapping)
    return {(y - y0, x - x0): int(c) for (y, x), c in mapping.items()}


def _signature(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _key(grid):
    return grid.shape, grid.tobytes()


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must be integer")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid color")
    return a.astype(np.int8, copy=True)


def _primitive(crop):
    runs = []
    for a in (crop, crop.T):
        for row in a:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            runs.extend(map(int, np.diff(cuts)))
    scale = int(np.gcd.reduce(runs))
    small = crop[::scale, ::scale]
    _require(np.array_equal(np.repeat(np.repeat(small, scale, 0), scale, 1), crop),
             "Nonuniform pixel scale")
    return scale, small


def _expanded(template, scale):
    return _normalize({(y * scale + dy, x * scale + dx): color
                       for (y, x), color in template.items()
                       for dy in range(scale) for dx in range(scale)})


def _read_input(input_grid):
    """Recover all conventions and resources from a single visible input."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] * 2 > inp.size, "Background must be a strict majority")
    frames = []
    for color in sorted(set(map(int, inp.flat)) - {bg}):
        coords = list(map(tuple, np.argwhere(inp == color)))
        y, x, bottom, right = _bounds(coords)
        board = inp[y:bottom, x:right]
        if min(board.shape) < 5:
            continue
        perimeter = np.r_[board[0], board[-1], board[:, 0], board[:, -1]]
        if np.all(perimeter == color) and np.any(board == bg):
            frames.append((y, x, bottom, right, color))
    _require(len(frames) == 1, "Need exactly one complete monochrome board frame")
    y, x, bottom, right, fixed_color = frames[0]
    board = inp[y:bottom, x:right].copy()
    outside = inp != bg
    outside[y:bottom, x:right] = False
    objects = []
    for comp in _components(map(tuple, np.argwhere(outside)), diagonal=True):
        y0, x0, y1, x1 = _bounds(comp)
        crop = inp[y0:y1, x0:x1]
        _require(np.count_nonzero(crop != bg) == len(comp), "Interleaved loose objects")
        scale, small = _primitive(crop)
        mapping = {(int(r), int(c)): int(small[r, c])
                   for r, c in np.argwhere(small != bg)}
        _require(fixed_color not in mapping.values(), "Board color occurs outside board")
        objects.append({"scale": scale, "signature": _signature(mapping),
                        "origin": (y0, x0)})
    sigs = sorted({o["signature"] for o in objects})
    _require(len(sigs) == 2, "Need exactly two normalized loose states")
    maps = [{(r, c): color for r, c, color in sig} for sig in sigs]
    _require({o["signature"] for o in objects if o["scale"] == 1} == set(sigs),
             "Both states must be explicitly visible at primitive scale one")
    # Shared whole-color masks identify the registration; state palettes are
    # disjoint. No bounding-box alignment is assumed for the two states.
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        ay, ax, _, _ = _bounds(a)
        by, bx, _, _ = _bounds(b)
        dy, dx = ay - by, ax - bx
        if a == {(r + dy, c + dx) for r, c in b}:
            shifts.add((dy, dx))
    _require(len(shifts) == 1, "Ambiguous or absent carrier registration")
    dy, dx = next(iter(shifts))
    maps[1] = {(r + dy, c + dx): v for (r, c), v in maps[1].items()}
    common = {p: v for p, v in maps[0].items() if maps[1].get(p) == v}
    _require(len(common) >= 3, "Carrier needs at least three invariant cells")
    stable_colors = set(common.values())
    _require(all({p: v for p, v in m.items() if v in stable_colors} == common
                 for m in maps), "Shared colors must belong entirely to the carrier")
    changing = [set(m.values()) - stable_colors for m in maps]
    _require(changing[0] and changing[1] and changing[0].isdisjoint(changing[1]),
             "Need disjoint state-specific accent palettes")
    _require(set(map(int, board.flat)) <= set(maps[0].values()) | set(maps[1].values()) | {bg, fixed_color},
             "Board imprints must use visible motif colors")
    cy, cx, _, _ = _bounds(common)
    templates = [{(r - cy, c - cx): v for (r, c), v in m.items()} for m in maps]
    for obj in objects:
        obj["state"] = sigs.index(obj["signature"])
    scales = Counter(o["scale"] for o in objects)
    states = Counter(o["state"] for o in objects)
    joint = Counter((o["scale"], o["state"]) for o in objects)
    _require(len(scales) >= 2, "Need a scale-transfer opportunity")
    return {"input": inp, "background": bg, "fixed_color": fixed_color,
            "bbox": (y, x, bottom, right), "board": board,
            "templates": templates, "common": common, "objects": objects,
            "scales": scales, "states": states, "joint": joint}


def _assemble(info, mode="separate"):
    """Up to two distinct colored exact covers, ignoring identical-piece swaps.

    Modes used for counterfactual checks change only resource constraints.
    Timeout is always an error, never evidence of uniqueness.
    """
    board, bg = info["board"], info["background"]
    # Printed fragments are fixed clues, not extra inventory. Each must be
    # covered by one matching motif, just like each still-empty cell.
    cells = sorted(map(tuple, np.argwhere(board != info["fixed_color"])))
    index = {p: i for i, p in enumerate(cells)}
    scales = sorted(info["scales"])
    types = [(s, state) for s in scales for state in (0, 1)]
    if mode == "separate":
        resources = tuple(info["scales"][s] for s in scales) + tuple(info["states"][t] for t in (0, 1))
        uses = [(scales.index(s), len(scales) + t) for s, t in types]
    elif mode in ("locked", "flipped"):
        resources = tuple(info["joint"][s, t if mode == "locked" else 1 - t] for s, t in types)
        uses = [(i,) for i in range(len(types))]
    elif mode == "free_states":
        resources = tuple(info["scales"][s] for s in scales)
        uses = [(scales.index(s),) for s, t in types]
    else:
        raise ValueError("Unknown inventory mode")
    by_cell = [[] for _ in cells]
    all_moves = []
    h, w = board.shape
    for tid, (scale, state) in enumerate(types):
        motif = _expanded(info["templates"][state], scale)
        _, _, mh, mw = _bounds(motif)
        for y in range(1, h - mh):
            for x in range(1, w - mw):
                paint_cells = [(y + r, x + c, color) for (r, c), color in motif.items()]
                if not all((r, c) in index for r, c, color in paint_cells):
                    continue
                if any(board[r, c] != bg and board[r, c] != color for r, c, color in paint_cells):
                    continue
                ids = [index[r, c] for r, c, color in paint_cells]
                mask = sum(1 << i for i in ids)
                paint = sum((color + 1) << (4 * index[r, c]) for r, c, color in paint_cells)
                move = (tid, mask, paint, y, x)
                all_moves.append(move)
                for i in ids:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=8000)
    def search(remaining, stock):
        nonlocal nodes
        nodes += 1
        if nodes > 8000:
            raise _SearchLimit("Exact-cover search exceeded 8000 nodes")
        if not remaining:
            return ((0, ()),) if not any(stock) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            choices = [m for m in by_cell[i]
                       if all(stock[j] > 0 for j in uses[m[0]]) and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = {}
        for tid, mask, paint, y, x in best:
            after = list(stock)
            for j in uses[tid]:
                after[j] -= 1
            for code, placements in search(remaining ^ mask, tuple(after)):
                combined = paint | code
                answers.setdefault(combined, ((types[tid][0], types[tid][1], y, x),) + placements)
                if len(answers) >= 2:
                    return tuple(answers.items())
        return tuple(answers.items())

    try:
        codes = search((1 << len(cells)) - 1, resources)
    finally:
        search.cache_clear()
    answers = []
    for code, placements in codes:
        result = board.copy()
        for (r, c), i in index.items():
            result[r, c] = ((code >> (4 * i)) & 15) - 1
        answers.append((result, placements))
    return answers, {"nodes": nodes, "candidate_placements": len(all_moves)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    answers, _ = _assemble(info)
    _require(len(answers) == 1, "Need exactly one final colored assembly")
    return answers[0][0].copy()


def _pick(rng, values):
    values = sorted(values)
    return values[int(rng.integers(len(values)))]


def _design(rng, change):
    """Grow a common scaffold and two attached state-specific patches."""
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    core = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(int(rng.integers(3, min(h * w, 7)))):
        frontier = {(y + dy, x + dx) for y, x in core
                    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                    if 0 <= y + dy < h and 0 <= x + dx < w} - core
        if frontier:
            core.add(_pick(rng, frontier))
    common = {p: 2 for p in core}
    if len(core) >= 5 and rng.random() < .45:
        common[_pick(rng, core)] = 3
    patches = []
    # Both patches live in a small logical region; their bounding boxes and
    # occupied areas may differ, while the entire carrier stays registered.
    for state in (0, 1):
        occupied, patch = set(core), set()
        for _ in range(int(rng.integers(1, 4))):
            frontier = {(y + dy, x + dx) for y, x in occupied
                        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                        if -1 <= y + dy <= h and -1 <= x + dx <= w} - occupied
            candidates = []
            for p in sorted(frontier):
                y0, x0, y1, x1 = _bounds(occupied | {p})
                if y1 - y0 <= 4 and x1 - x0 <= 4:
                    candidates.append(p)
            if candidates:
                p = _pick(rng, candidates)
                occupied.add(p)
                patch.add(p)
        patches.append(patch)
    if change == "recolor":
        patches[1] = patches[0].copy()
    elif patches[0] == patches[1]:
        return None
    templates = []
    for state, patch in enumerate(patches):
        m = dict(common)
        for j, p in enumerate(sorted(patch)):
            m[p] = 4 + state * 2 + (j % 2 if rng.random() < .5 else 0)
        templates.append(m)
    return templates


def _target_board(rng, templates, scales, states, require_contact, counted_choice):
    motifs = [_expanded(templates[t], s) for s, t in zip(scales, states)]
    sizes = [(_bounds(m)[2], _bounds(m)[3]) for m in motifs]
    max_h, max_w = max(h for h, w in sizes), max(w for h, w in sizes)
    area = sum(map(len, motifs))
    h = int(rng.integers(max(5, max_h + 2), 17))
    w = int(rng.integers(max(5, max_w + 2), 17))
    if area > (h - 2) * (w - 2) * .85:
        return None
    used, placed = set(), {}
    order = sorted(range(len(motifs)), key=lambda i: (-len(motifs[i]), i))
    for i in order:
        mh, mw = sizes[i]
        candidates, touching = [], []
        for y in range(1, h - mh):
            for x in range(1, w - mw):
                cells = {(y + r, x + c) for r, c in motifs[i]}
                if cells & used:
                    continue
                item = (y, x, cells)
                candidates.append(item)
                if any((r + dr, c + dc) in used for r, c in cells
                       for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))):
                    touching.append(item)
        if not candidates:
            return None
        choices = touching if touching and rng.random() < (.98 if require_contact else .45) else candidates
        y, x, cells = choices[int(rng.integers(len(choices)))]
        placed[i] = (y, x)
        used.update(cells)
    groups = _components(used)
    shared = any(sum(bool(comp & {(placed[i][0] + r, placed[i][1] + c)
                                  for r, c in motifs[i]}) for i in placed) > 1 for comp in groups)
    if require_contact and not shared:
        return None
    board = np.ones((h, w), dtype=np.int8)
    answer = board.copy()
    for i, (y, x) in placed.items():
        for (r, c), color in motifs[i].items():
            board[y + r, x + c] = 0
            answer[y + r, x + c] = color
    # Leave a few already-printed logical pixels in each motif's footprint.
    # For an in-place color change, all but the large carrier disclose their
    # state. Its state must be deduced from the remaining state inventory.
    common = {p: v for p, v in templates[0].items() if templates[1].get(p) == v}
    for i, (y, x) in placed.items():
        s, state = scales[i], states[i]
        raw = templates[state]
        ymin, xmin, _, _ = _bounds(raw)
        accent = sorted(set(raw) - set(common))
        printed = []
        if counted_choice and i != 2:
            printed.append(_pick(rng, accent))
        elif not counted_choice and rng.random() < .65:
            printed.append(_pick(rng, accent))
        if rng.random() < .7:
            printed.append(_pick(rng, common))
        for r, c in printed:
            yy, xx = y + (r - ymin) * s, x + (c - xmin) * s
            board[yy:yy + s, xx:xx + s] = raw[r, c]
    # A printed clue must not accidentally erase the shared-hole demonstration.
    holes = set(map(tuple, np.argwhere(board == 0)))
    if require_contact and not any(
            sum(bool(comp & {(placed[i][0] + r, placed[i][1] + c)
                              for r, c in motifs[i]}) for i in placed) > 1
            for comp in _components(holes)):
        return None
    return board, answer


def _separated(a, b):
    y, x, h, w = a
    r, c, hh, ww = b
    return y + h < r or r + hh < y or x + w < c or c + ww < x


def _scatter(rng, board, loose):
    items = [board] + loose
    sizes = [a.shape for a in items]
    options = []
    orders = [sorted(range(len(items)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True),
              list(map(int, rng.permutation(len(items))))]
    for width in range(max(w for h, w in sizes), 31):
        for order in orders:
            y = x = shelf = 0
            boxes = [None] * len(items)
            for i in order:
                h, w = sizes[i]
                if x and x + w > width:
                    y, x, shelf = y + shelf + 1, 0, 0
                boxes[i] = (y, x, h, w)
                x += w + 1
                shelf = max(shelf, h)
            height = max(y + h for y, x, h, w in boxes)
            if height <= 30:
                options.append(boxes)
    if not options:
        return None
    boxes = options[int(rng.integers(len(options)))]
    h0 = max(y + h for y, x, h, w in boxes)
    w0 = max(x + w for y, x, h, w in boxes)
    occupied = sum(np.count_nonzero(a) for a in items)
    if occupied * 2 >= 900:
        return None
    h = int(rng.integers(h0, min(30, h0 + 7) + 1))
    w = int(rng.integers(w0, min(30, w0 + 7) + 1))
    while h * w <= 2 * occupied:
        if h < 30 and (h <= w or w == 30):
            h += 1
        elif w < 30:
            w += 1
        else:
            return None
    dy, dx = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
    boxes = [(y + dy, x + dx, hh, ww) for y, x, hh, ww in boxes]
    for _ in range(40):
        i = int(rng.integers(len(boxes)))
        _, _, hh, ww = boxes[i]
        candidate = (int(rng.integers(h - hh + 1)), int(rng.integers(w - ww + 1)), hh, ww)
        if all(_separated(candidate, b) for j, b in enumerate(boxes) if j != i):
            boxes[i] = candidate
    canvas = np.zeros((h, w), dtype=np.int8)
    for a, (y, x, hh, ww) in zip(items, boxes):
        canvas[y:y + hh, x:x + ww] = a
    return canvas


def _pair(rng, split, index):
    for _ in range(96):
        change = "recolor" if split == "train" and index == 0 else (
            "reshape" if split == "train" and index == 1 else
            ("recolor" if rng.random() < .4 else "reshape"))
        templates = _design(rng, change)
        if templates is None:
            continue
        n = int(rng.integers(4, 7)) if split == "test" else (3 if index == 0 else int(rng.integers(3, 6)))
        scales = [1, 1, 2] + [int(rng.choice([1, 1, 2])) for _ in range(n - 3)]
        # A larger carrier exchanges its state with one of the two unit copies.
        # The two unit copies then explicitly supply both original templates.
        majority = int(rng.integers(2))
        target_states = [majority, majority, 1 - majority] + [int(rng.integers(2)) for _ in range(n - 3)]
        source_states = target_states.copy()
        source_states[1], source_states[2] = source_states[2], source_states[1]
        # Occasional already-demonstrated scale-two repetitions and scale-three
        # transfer, only for sufficiently compact motifs and small inventories.
        union = set(templates[0]) | set(templates[1])
        y0, x0, y1, x1 = _bounds(union)
        if max(y1 - y0, x1 - x0) <= 3 and n <= 4 and rng.random() < .25:
            scales[2] = 3
        if Counter(zip(scales, target_states)) == Counter((s, 1 - t) for s, t in zip(scales, source_states)):
            continue
        target = _target_board(rng, templates, scales, target_states,
                               require_contact=(split == "test" or index == 1),
                               counted_choice=(change == "recolor"))
        if target is None:
            continue
        board, answer = target
        loose = []
        for s, t in zip(scales, source_states):
            m = _expanded(templates[t], s)
            _, _, h, w = _bounds(m)
            a = np.zeros((h, w), dtype=np.int8)
            for p, color in m.items():
                a[p] = color
            loose.append(a)
        canvas = _scatter(rng, board, loose)
        if canvas is None:
            continue
        palette = rng.permutation(10).astype(np.int8)
        canvas, answer = palette[canvas], palette[answer]
        k, reflect = int(rng.integers(4)), bool(rng.integers(2))
        canvas, answer = np.rot90(canvas, k), np.rot90(answer, k)
        if reflect:
            canvas, answer = np.fliplr(canvas), np.fliplr(answer)
        try:
            info = _read_input(canvas)
            answers, _ = _assemble(info)
        except (ValueError, _SearchLimit):
            continue
        if len(answers) == 1 and np.array_equal(answers[0][0], answer):
            if change == "recolor":
                free, _ = _assemble(info, "free_states")
                if len(free) < 2:
                    continue
            return {"input": canvas.copy(), "output": answer.copy()}
    raise RuntimeError("Bounded construction exhausted 96 proposals")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            for _ in range(8):
                pair = _pair(rng, split, i)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted")
    return puzzle


def _analyze(pair, ablations=False):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _read_input(inp)
    answers, stats = _assemble(info)
    _require(len(answers) == 1, "Assembly is unsolved or color-ambiguous")
    expected, placements = answers[0]
    _require(np.array_equal(expected, out), "Output differs from unique input-only assembly")
    fixed = info["board"] != info["background"]
    _require(np.array_equal(out[fixed], info["board"][fixed]), "Changed fixed board cells")
    _require(not np.any(out == info["background"]), "Unfilled board cells")
    used_scales = Counter(s for s, t, y, x in placements)
    used_states = Counter(t for s, t, y, x in placements)
    _require(used_scales == info["scales"] and used_states == info["states"], "Inventory loss")
    joint = Counter((s, t) for s, t, y, x in placements)
    unchanged = joint == info["joint"]
    all_flipped = joint == Counter({(s, 1 - t): n for (s, t), n in info["joint"].items()})
    holes = set(map(tuple, np.argwhere(~fixed)))
    components = _components(holes)
    piece_supports = []
    for s, t, y, x in placements:
        piece_supports.append({(y + r, x + c) for r, c in _expanded(info["templates"][t], s)})
    shared = any(sum(bool(comp & p) for p in piece_supports) > 1 for comp in components)
    carrier_colors = set(info["common"].values())
    for c in carrier_colors:
        supplied = np.count_nonzero(inp == c) - np.count_nonzero(info["board"] == c)
        _require(supplied == np.count_nonzero(out == c), "Lost carrier pixels")
    accents = set.union(*(set(m.values()) for m in info["templates"])) - carrier_colors
    pixel_change = any(np.count_nonzero(inp == c) - np.count_nonzero(info["board"] == c)
                       != np.count_nonzero(out == c) for c in accents)
    features = {"board_shape": tuple(out.shape), "canvas_shape": tuple(inp.shape),
                "background": info["background"], "count": len(placements),
                "scale_stock": tuple(sorted(info["scales"].items())),
                "state_stock": tuple(info["states"][t] for t in (0, 1)),
                "transfer": not unchanged, "not_all_flip": not all_flipped,
                "shared_gap": shared, "gap_components": len(components),
                "pixel_counts_change": pixel_change,
                "template_areas": tuple(map(len, info["templates"])),
                "imprint_cells": int(np.count_nonzero(fixed & (info["board"] != info["fixed_color"]))),
                "footprints_equal": set(info["templates"][0]) == set(info["templates"][1]),
                "carrier": tuple(sorted((p[0], p[1]) for p in info["common"])),
                "vocabulary": _vocabulary(info),
                "edge": bool(np.any(inp[0] != info["background"]) or np.any(inp[-1] != info["background"])
                             or np.any(inp[:, 0] != info["background"]) or np.any(inp[:, -1] != info["background"])),
                "solver_nodes": stats["nodes"]}
    if ablations:
        locked, _ = _assemble(info, "locked")
        free, _ = _assemble(info, "free_states")
        features["locked_impossible"] = not locked
        features["state_stock_disambiguates"] = len(free) > 1
    return features


def _vocabulary(info):
    """Color-name and translation invariant, retaining scene orientation."""
    variants = []
    for order in ((0, 1), (1, 0)):
        roles, states = {}, []
        for t in order:
            states.append(tuple((y, x, roles.setdefault(c, len(roles)))
                                for y, x, c in _signature(info["templates"][t])))
        variants.append(tuple(states))
    return min(variants)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "Invalid pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp = _grid(pair["input"])
            key = _key(inp)
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            features[split].append(_analyze(pair, ablations=(split == "train")))
    _require(all(f["transfer"] and f["not_all_flip"] for fs in features.values() for f in fs),
             "Generated examples must distinguish intact and global-toggle rules")
    _require(any(f["shared_gap"] for f in features["train"]), "Training must show shared-gap assembly")
    _require(any(f["state_stock_disambiguates"] for f in features["train"]),
             "Training must require the state inventory to resolve a color choice")
    _require(any(not f["footprints_equal"] for f in features["train"]),
             "Training must demonstrate attachment-support changes")
    _require(all(f["shared_gap"] for f in features["test"]), "Test must recombine shared-gap assembly")
    return None


def self_test():
    """42 seeds, small negative fixtures, and bounded diversity checks.

    Grids are streamed one puzzle at a time. Only scalar/tuple measurements and
    SHA-256 digests survive each iteration; no disk output or stored traces.
    """
    metrics = {"puzzles": 42, "pairs": 0, "transfers": 0, "shared_gaps": 0,
               "locked_impossible": 0, "state_stock_disambiguates": 0,
               "pixel_count_changes": 0, "edges": 0, "max_search_nodes": 0}
    seen, vocabularies, shapes, carriers, backgrounds, scales, counts = (set() for _ in range(7))
    first = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        if first is None:
            first = puzzle
        local_vocab = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _analyze(pair, ablations=True)
                metrics["pairs"] += 1
                for source, dest in (("transfer", "transfers"), ("shared_gap", "shared_gaps"),
                                     ("locked_impossible", "locked_impossible"),
                                     ("state_stock_disambiguates", "state_stock_disambiguates"),
                                     ("pixel_counts_change", "pixel_count_changes"), ("edge", "edges")):
                    metrics[dest] += int(f[source])
                metrics["max_search_nodes"] = max(metrics["max_search_nodes"], f["solver_nodes"])
                for role in ("input", "output"):
                    digest.update(repr(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
                vocabularies.add(f["vocabulary"])
                local_vocab.add(f["vocabulary"])
                carriers.add(f["carrier"])
                shapes.add(f["board_shape"])
                backgrounds.add(f["background"])
                scales.add(f["scale_stock"])
                counts.add(f["count"])
        _require(len(local_vocab) >= 3, "Within-puzzle structural vocabulary collapsed")
        seen.add(digest.digest())
    _require(len(seen) == 42 and len(vocabularies) >= 100, "Insufficient structural diversity")
    _require(len(carriers) >= 40 and len(shapes) >= 40, "Carrier or board geometry collapsed")
    _require(backgrounds == set(range(10)) and len(scales) >= 6 and len(counts) >= 3,
             "Palette roles, scale stocks or counts collapsed")
    _require(metrics["locked_impossible"] == metrics["pairs"], "No-transfer shortcut survived")
    _require(metrics["state_stock_disambiguates"] >= 42, "State counts are redundant")
    _require(metrics["pixel_count_changes"] >= 100 and metrics["shared_gaps"] >= 100,
             "Important mechanisms not exposed")
    again = generate_puzzle(HOST_SEEDS[0])
    _require(all(np.array_equal(a[r], b[r]) for s in first for a, b in zip(first[s], again[s])
                 for r in ("input", "output")), "Seed not reproducible after intervening calls")
    # Output corruption must fail the family checker.
    bad = {s: [{r: p[r].copy() for r in p} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupt fixed frame accepted")
    first_info = _read_input(first["train"][0]["input"])
    # Removing a supplied object must not be repaired by copying a template.
    missing = first_info["input"].copy()
    obj = first_info["objects"][0]
    y, x = obj["origin"]
    raw = {(r, c): color for r, c, color in obj["signature"]}
    for r, c in _expanded(raw, obj["scale"]):
        missing[y + r, x + c] = first_info["background"]
    # Removing all state imprints from the first (equal-footprint) scene
    # leaves several colored answers with exactly the same state inventory.
    ambiguous = first_info["input"].copy()
    y, x, bottom, right = first_info["bbox"]
    region = ambiguous[y:bottom, x:right]
    accents = (set(first_info["templates"][0].values())
               | set(first_info["templates"][1].values())) - set(first_info["common"].values())
    region[np.isin(region, list(accents))] = first_info["background"]
    for malformed in (missing, ambiguous):
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted missing inventory or ambiguous colors")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    metrics.update({"vocabularies": len(vocabularies), "carrier_geometries": len(carriers),
                    "board_shapes": len(shapes), "scale_stocks": len(scales),
                    "motif_counts": sorted(counts), "backgrounds": sorted(backgrounds)})
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
