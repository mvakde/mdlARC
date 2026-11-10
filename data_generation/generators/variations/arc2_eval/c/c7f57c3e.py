"""Scaffold-registered exchange of two locally visible, scalable motif states."""

import hashlib
import numpy as np


def _check_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    nt = _check_count("num_train", num_train, 2, 10)
    ns = _check_count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    return nt, ns


def _pick(rng, values):
    values = sorted(values)
    return values[int(rng.integers(len(values)))]


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _signature(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _pair_digest(pair):
    content = tuple((g.shape[0], g.shape[1], tuple(int(v) for v in g.flat))
                    for g in (pair["input"], pair["output"]))
    return hashlib.sha256(repr(content).encode()).hexdigest()


# Content and dimensions of the three staged source pairs, not palette exclusions.
_SOURCE_DIGESTS = {
    "4aaa13080e5f1fb3f16a44ca3e9780f2418c1a73f448a957df93f1c4625dd414",
    "dcbbef495b1977fb9a43d036f150164e15949335534e54c7e271ad65329dcf2b",
    "1cad9e98c71dcc99a78c8ff279d94cb6cfbf6d332353d0cc995a1a2945de597e",
}


def _components(mask, diagonal=True):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    steps = [(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
             if (y or x) and (diagonal or abs(y) + abs(x) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                point = y + dy, x + dx
                if point in remaining:
                    remaining.remove(point)
                    group.add(point)
                    stack.append(point)
        groups.append(group)
    return groups


def _connected(points):
    y0, y1, x0, x1 = _bounds(points)
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    for y, x in points:
        mask[y - y0, x - x0] = True
    return len(_components(mask)) == 1


def _core(rng):
    """A procedural footprint; carrier color itself need not be connected."""
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    cy, cx = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    family = int(rng.integers(4))
    if family == 0:
        cells = {(cy, x) for x in range(w)} | {(y, cx) for y in range(h)}
    elif family == 1:
        cells = {(cy, cx)}
        for end in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            y, x = cy, cx
            while (y, x) != end:
                y += int(np.sign(end[0] - y))
                x += int(np.sign(end[1] - x))
                cells.add((y, x))
    elif family == 2:
        cells = ({(y, 0) for y in range(h)} | {(y, w - 1) for y in range(h)}
                 | {(h - 1, x) for x in range(w)} | {(cy, x) for x in range(cx + 1)})
    else:
        cells = {(cy, x) for x in range(w)} | {(y, cx) for y in range(h)}
        # Different proportions, branches, and sparsity without a layout bank.
        removable = [p for p in sorted(cells) if p != (cy, cx)]
        for p in rng.permutation(len(removable))[:int(rng.integers(1, 4))]:
            trial = cells - {removable[int(p)]}
            if len(trial) >= 5 and _connected(trial):
                cells = trial
    for _ in range(int(rng.integers(0, 5))):
        frontier = {(y + dy, x + dx) for y, x in cells
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= y + dy < h and 0 <= x + dx < w} - cells
        if frontier:
            cells.add(_pick(rng, frontier))
    cells.add((cy, cx))
    interior = {(cy, cx)}
    candidates = [p for p in sorted(cells) if 0 < p[0] < h - 1
                  and 0 < p[1] < w - 1 and p not in interior]
    rng.shuffle(candidates)
    interior.update(candidates[:int(rng.integers(0, min(3, len(candidates)) + 1))])
    if len(cells - interior) < 3:
        interior = {(cy, cx)}
    return cells, interior


def _decoration(rng, core, side):
    y0, y1, x0, x1 = _bounds(core)
    if side == 0:
        root = _pick(rng, {p for p in core if p[0] == y0})
        direction, lateral, depth_limit = (-1, 0), (0, 1), 2 if y1 - y0 == 2 else 1
    elif side == 1:
        root = _pick(rng, {p for p in core if p[0] == y1})
        direction, lateral, depth_limit = (1, 0), (0, 1), 2 if y1 - y0 == 2 else 1
    elif side == 2:
        root = _pick(rng, {p for p in core if p[1] == x0})
        direction, lateral, depth_limit = (0, -1), (1, 0), 2 if x1 - x0 == 2 else 1
    else:
        root = _pick(rng, {p for p in core if p[1] == x1})
        direction, lateral, depth_limit = (0, 1), (1, 0), 2 if x1 - x0 == 2 else 1
    patch = set()
    for d in range(1, int(rng.integers(1, depth_limit + 1)) + 1):
        width = int(rng.integers(1, 4))
        start = -int(rng.integers(width))
        for t in range(start, start + width):
            p = (root[0] + direction[0] * d + lateral[0] * t,
                 root[1] + direction[1] * d + lateral[1] * t)
            if y0 - 2 <= p[0] <= y1 + 2 and x0 - 2 <= p[1] <= x1 + 2:
                patch.add(p)
    return patch


def _sample_design(rng, change):
    core, interior = _core(rng)
    side = int(rng.integers(4))
    a = _decoration(rng, core, side)
    b = a if change == "recolor" else _decoration(rng, core, (side + int(rng.integers(1, 4))) % 4)
    moving_interior = interior if change == "combined" or (
        change == "recolor" and rng.random() < .7) else set()
    common = core - moving_interior
    changed = (a | moving_interior, b | moving_interior)
    # Common color masks never migrate; separate accents may remain unchanged.
    common_map = {p: 1 for p in common}
    stable_candidates = sorted(common)
    rng.shuffle(stable_candidates)
    stable_count = int(rng.integers(3))
    if change == "relocate" or not moving_interior:
        stable_count = max(1, stable_count)
    stable_points = sorted(interior & common)
    stable_points += [p for p in stable_candidates if p not in stable_points]
    for j, p in enumerate(stable_points[:min(stable_count, len(common) - 3)]):
        common_map[p] = 2 + j
    next_color = max(common_map.values()) + 1
    maps = []
    for state in range(2):
        points = sorted(changed[state])
        rng.shuffle(points)
        available = 10 - next_color - (1 if state == 0 else 0)
        count = int(rng.integers(1, min(3, len(points), available) + 1))
        # Often give an interior accent and endpoint separate colors.
        mapping = dict(common_map)
        for j, p in enumerate(points):
            mapping[p] = next_color + j % count
        next_color += count
        maps.append(mapping)
    # A primitive logical pixel must be observable even when reference copies
    # are already enlarged. Break an accidental coarser block pattern locally.
    for mapping in maps:
        y0, y1, x0, x1 = _bounds(mapping)
        crop = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=int)
        for (y, x), c in mapping.items():
            crop[y - y0, x - x0] = c
        if _primitive(crop)[0] > 1:
            point = min(p for p, c in common_map.items() if c == 1)
            for state in range(2):
                maps[state][point] = maps[state][min(changed[state])]
            break
    return tuple(maps)


def _colored_design(rng, design, background=None, palette=None):
    labels = sorted({0} | set(design[0].values()) | set(design[1].values()))
    if palette is None:
        palette = [int(c) for c in rng.permutation(10)]
        if background is not None:
            palette.remove(background)
            palette.insert(0, background)
    roles = dict(zip(labels, palette))
    return roles[0], tuple({p: roles[c] for p, c in t.items()} for t in design)


def _rectangles_separated(a, b, gap=1):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    return ay + ah + gap <= by or by + bh + gap <= ay or ax + aw + gap <= bx or bx + bw + gap <= ax


def _shelves(sizes, width, gap, order):
    placed = [None] * len(sizes)
    y, x, shelf = 0, 0, 0
    for i in order:
        h, w = sizes[i]
        if w > width:
            return None
        if x and x + w > width:
            y, x, shelf = y + shelf + gap, 0, 0
        if y + h > 30:
            return None
        placed[i] = (y, x, h, w)
        x += w + gap
        shelf = max(shelf, h)
    return placed


def _layout_options(rng, shape, scales, limit=30):
    sizes = [(shape[0] * s, shape[1] * s) for s in scales]
    orders = [sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True),
              list(rng.permutation(len(sizes)))]
    options = []
    for gap in (1, int(rng.integers(1, 4))):
        for width in range(max(w for h, w in sizes), limit + 1):
            for order in orders:
                p = _shelves(sizes, width, gap, order)
                if p is not None and max(y + h for y, x, h, w in p) <= limit:
                    options.append(p)
    return options


def _arrange(rng, templates, force_scaled, regime, simple):
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    shape = y1 - y0 + 1, x1 - x0 + 1
    largest_area = max(map(len, templates))
    margin = int(rng.random() < .4)
    options = None
    # Scale and count proposals are independent, subject only to actual packing.
    for attempt in range(32):
        reference = 2 if not simple and rng.random() < .18 else 1
        count = int(rng.integers(3 if force_scaled else 2, 6 if simple else 9))
        scales = [reference, reference]
        for _ in range(count - 2):
            choices = [1, 1, 1, 1, 2, 2, 3, 4, 5] if reference == 1 else [2, 2, 2, 4]
            scales.append(int(_pick(rng, choices)))
        if force_scaled:
            scales[2] = 2 * reference
        if regime == "target_test" and not simple and rng.random() < .25 and len(scales) > 2:
            scales[2] = 3 if reference == 1 else 4
        if attempt == 31:
            scales = [1, 1, 2] if force_scaled else [1, 1]
        area = largest_area * sum(s * s for s in scales)
        if area * 2 >= 900:
            continue
        options = _layout_options(rng, shape, scales, 30 - 2 * margin)
        if options:
            break
    assert options, "bounded construction must fit its reference copies"
    rects = options[int(rng.integers(len(options)))]
    h0 = max(y + h for y, x, h, w in rects)
    w0 = max(x + w for y, x, h, w in rects)
    h = int(rng.integers(h0 + 2 * margin, min(30, h0 + 10) + 1))
    w = int(rng.integers(w0 + 2 * margin, min(30, w0 + 10) + 1))
    if regime == "target_test" and rng.random() < .3:
        h, w = max(h, int(rng.integers(24, 31))), max(w, int(rng.integers(24, 31)))
    if rng.random() < .08:
        h = w = 30
    while h * w <= area * 2:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    # Shift shelves, then scatter individual rectangles. Reserved rectangles
    # include everything either state can write, including currently blank cells.
    dy = int(rng.integers(margin, h - h0 - margin + 1))
    dx = int(rng.integers(margin, w - w0 - margin + 1))
    rects = [(y + dy, x + dx, rh, rw) for y, x, rh, rw in rects]
    for _ in range(70):
        i = int(rng.integers(len(rects)))
        _, _, rh, rw = rects[i]
        candidate = (int(rng.integers(margin, h - rh - margin + 1)),
                     int(rng.integers(margin, w - rw - margin + 1)), rh, rw)
        if all(_rectangles_separated(candidate, other) for j, other in enumerate(rects) if j != i):
            rects[i] = candidate
    if not margin and rng.random() < .5:
        # Move the whole collection to a randomly chosen edge of its union.
        side = int(rng.integers(4))
        dy = -min(r[0] for r in rects) if side == 0 else h - max(r[0] + r[2] for r in rects) if side == 1 else 0
        dx = -min(r[1] for r in rects) if side == 2 else w - max(r[1] + r[3] for r in rects) if side == 3 else 0
        rects = [(y + dy, x + dx, rh, rw) for y, x, rh, rw in rects]
    states = [0, 1]
    probability = float(rng.choice([.15, .5, .85]))
    states += [int(rng.random() < probability) for _ in scales[2:]]
    return (h, w), [(y - y0 * s, x - x0 * s, s, state)
                    for (y, x, _, _), s, state in zip(rects, scales, states)]


def _render_template(grid, template, anchor, scale):
    for (y, x), color in template.items():
        yy, xx = anchor[0] + y * scale, anchor[1] + x * scale
        grid[yy:yy + scale, xx:xx + scale] = color


def _generate_pair(rng, design, palette, background, force_scaled, regime, simple):
    bg, templates = _colored_design(rng, design, background, palette)
    shape, placements = _arrange(rng, templates, force_scaled, regime, simple)
    inp, out = np.full(shape, bg, dtype=int), np.full(shape, bg, dtype=int)
    for y, x, scale, state in placements:
        _render_template(inp, templates[state], (y, x), scale)
        _render_template(out, templates[1 - state], (y, x), scale)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflection:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": inp.copy(), "output": out.copy()}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Only the binary exchange rule is semantic across a puzzle. Regimes control
    # demonstrations; local shapes, colors and orientations are visible data.
    regime_kind = int(rng.choice([0, 1, 2], p=[.2, .35, .45]))
    simple = regime_kind == 0
    simple_change = _pick(rng, ["recolor", "relocate", "combined"])
    shared_design = _sample_design(rng, simple_change)
    shared_palette = [int(c) for c in rng.permutation(10)]
    reuse_accent = rng.random() < .4
    puzzle, seen = {"train": [], "test": []}, set()
    first_accents = None
    for split, count, regime in (("train", num_train, "source_train"),
                                 ("test", num_test, "target_test")):
        for index in range(count):
            if simple:
                change = simple_change
            elif regime_kind == 1:
                change = ("recolor" if index % 2 == 0 else "relocate") if split == "train" else "combined"
            else:
                change = "combined" if split == "train" and index == 0 else _pick(rng, ["recolor", "relocate", "combined"])
            for attempt in range(32):
                design = shared_design if simple else _sample_design(rng, change)
                bg = _pick(rng, first_accents) if (not simple and split == "test" and index == 0 and reuse_accent) else None
                pair = _generate_pair(rng, design, shared_palette if simple else None, bg,
                                      split == "train" and index == 0, regime, simple)
                key = _grid_key(pair["input"])
                if key not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if first_accents is None:
                        first_accents = set(int(v) for v in pair["input"][pair["input"] != pair["output"]])
                        first_accents.discard(int(np.argmax(np.bincount(pair["input"].ravel(), minlength=10))))
                    break
            else:
                raise AssertionError("unique pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _primitive(crop):
    """Infer block size from color runs, without knowing a template or scale."""
    lengths = []
    for array in (crop, crop.T):
        for row in array:
            starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(int(n) for n in np.diff(starts))
    scale = int(np.gcd.reduce(lengths))
    reduced = crop[::scale, ::scale]
    assert np.array_equal(np.repeat(np.repeat(reduced, scale, 0), scale, 1), crop)
    return scale, reduced


def _read_input(grid):
    """Recover segmentation, scale, two states and registration from pixels only."""
    bg = int(np.argmax(np.bincount(grid.ravel().astype(int), minlength=10)))
    assert np.count_nonzero(grid == bg) * 2 > grid.size, "background is not unambiguous"
    objects = []
    for group in _components(grid != bg):
        y0, y1, x0, x1 = _bounds(group)
        crop = grid[y0:y1 + 1, x0:x1 + 1]
        assert np.count_nonzero(crop != bg) == len(group), "interleaved groups"
        scale, small = _primitive(crop)
        raw = {(int(y), int(x)): int(small[y, x]) for y, x in zip(*np.nonzero(small != bg))}
        objects.append({"box": (y0, x0), "scale": scale, "raw": _signature(raw)})
    assert 2 <= len(objects), "two visible states are required"
    signatures = sorted({o["raw"] for o in objects})
    assert len(signatures) == 2, "there must be exactly two normalized states"
    minimum = min(o["scale"] for o in objects)
    assert {o["raw"] for o in objects if o["scale"] == minimum} == set(signatures), "both states need a common reference scale"
    maps = [{(y, x): c for y, x, c in sig} for sig in signatures]
    colors = set(maps[0].values()) & set(maps[1].values())
    shifts = set()
    for color in colors:
        pa = {p for p, c in maps[0].items() if c == color}
        pb = {p for p, c in maps[1].items() if c == color}
        ay, _, ax, _ = _bounds(pa)
        by, _, bx, _ = _bounds(pb)
        shift = ay - by, ax - bx
        if pa == {(y + shift[0], x + shift[1]) for y, x in pb}:
            shifts.add(shift)
    candidates = []
    for dy, dx in sorted(shifts):
        shifted = {(y + dy, x + dx): c for (y, x), c in maps[1].items()}
        common = {p for p, c in maps[0].items() if shifted.get(p) == c}
        # A shared whole-color pattern is a visible registration cue. Score all
        # unchanged structure, not just the largest connected carrier fragment.
        if len(common) >= 3:
            candidates.append((len(common), (dy, dx), shifted, common))
    assert candidates, "no shared scaffold registration"
    best_score = max(c[0] for c in candidates)
    candidates = [c for c in candidates if c[0] == best_score]
    assert len(candidates) == 1, "ambiguous scaffold registration"
    _, shift, aligned, common = candidates[0]
    oy, _, ox, _ = _bounds(common)
    templates = [{(y - oy, x - ox): c for (y, x), c in m.items()} for m in (maps[0], aligned)]
    union = set(templates[0]) | set(templates[1])
    ymin, ymax, xmin, xmax = _bounds(union)
    rectangles, occurrences = [], []
    expected = grid.copy()
    for obj in objects:
        state = signatures.index(obj["raw"])
        s = obj["scale"]
        dy, dx = shift if state else (0, 0)
        anchor = obj["box"][0] + (oy - dy) * s, obj["box"][1] + (ox - dx) * s
        rect = (anchor[0] + ymin * s, anchor[1] + xmin * s,
                (ymax - ymin + 1) * s, (xmax - xmin + 1) * s)
        y, x, h, w = rect
        assert 0 <= y and y + h <= grid.shape[0] and 0 <= x and x + w <= grid.shape[1], "replacement clipped"
        assert all(_rectangles_separated(rect, r) for r in rectangles), "replacement regions collide"
        rectangles.append(rect)
        # Clear all state support, including disappearing endpoints. Full-grid
        # equality below also checks scaffold, anchors and unaffected background.
        for py, px in union:
            yy, xx = anchor[0] + py * s, anchor[1] + px * s
            expected[yy:yy + s, xx:xx + s] = bg
        _render_template(expected, templates[1 - state], anchor, s)
        occurrences.append({"anchor": anchor, "scale": s, "state": state})
    a, b = templates
    same = {p: c for p, c in a.items() if b.get(p) == c}
    recolor = any(a[p] != b[p] for p in set(a) & set(b))
    relocate = set(a) != set(b)
    assert recolor or relocate
    stable_colors = set(same.values())
    carrier_color = max(sorted(stable_colors), key=lambda c: sum(v == c for v in same.values()))
    carrier = {p for p, c in same.items() if c == carrier_color}
    return {"expected": expected, "background": bg, "templates": templates,
            "scaffold": tuple(sorted(same)), "carrier": tuple(sorted(carrier)),
            "stable_colors": stable_colors, "occurrences": occurrences,
            "palette": frozenset(int(v) for v in np.unique(grid)),
            "change": "combined" if recolor and relocate else "recolor" if recolor else "relocate",
            "recolor": recolor, "relocate": relocate,
            "bbox_shift": _bounds(a) != _bounds(b), "reference_scale": minimum,
            "edge": any(np.any(g[0] != bg) or np.any(g[-1] != bg) or np.any(g[:, 0] != bg) or np.any(g[:, -1] != bg) for g in (grid, expected))}


def _analyze_pair(input_grid, output_grid, regime=None):
    assert input_grid.shape == output_grid.shape
    info = _read_input(input_grid)
    assert np.array_equal(info["expected"], output_grid), "incorrect scaffold-aligned exchange"
    assert not np.array_equal(input_grid, output_grid)
    return info


def _inspect_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    infos, seen_inputs, seen_pairs = {"train": [], "test": []}, set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            ikey = _grid_key(pair["input"])
            pkey = ikey, _grid_key(pair["output"])
            assert ikey not in seen_inputs and pkey not in seen_pairs
            assert _pair_digest(pair) not in _SOURCE_DIGESTS, "exact source copy"
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            infos[split].append(_analyze_pair(pair["input"], pair["output"]))
    for feature in ("recolor", "relocate"):
        assert not any(i[feature] for i in infos["test"]) or any(i[feature] for i in infos["train"]), "training omits a required kind of change"
    scaled_train = any(len({o["scale"] for o in i["occurrences"]}) > 1 for i in infos["train"])
    assert scaled_train, "training must demonstrate enlarged substitution"
    return {"pair_infos": infos}


def validate_generated_puzzle(puzzle: dict) -> None:
    _inspect_puzzle(puzzle)


def solve(train_pairs, test_input):
    """Prediction interface receiving no test labels, RNG state or metadata."""
    training = [_analyze_pair(p["input"], p["output"]) for p in train_pairs]
    info = _read_input(test_input)
    for feature in ("recolor", "relocate"):
        assert not info[feature] or any(i[feature] for i in training)
    if len({o["scale"] for o in info["occurrences"]}) > 1:
        assert any(len({o["scale"] for o in i["occurrences"]}) > 1 for i in training)
    return info["expected"].copy()


def _canonical_geometry(points):
    variants = []
    for rotation in range(4):
        for reflected in (False, True):
            transformed = []
            for y, x in points:
                for _ in range(rotation):
                    y, x = x, -y
                transformed.append((y, -x if reflected else x))
            y0, _, x0, _ = _bounds(transformed)
            variants.append(tuple(sorted((y - y0, x - x0) for y, x in transformed)))
    return min(variants)


def _vocabulary(info):
    variants = []
    for order in ((0, 1), (1, 0)):
        colors, encoded = {}, []
        for state in order:
            template = []
            for y, x, c in _signature(info["templates"][state]):
                if c not in colors:
                    colors[c] = len(colors)
                template.append((y, x, colors[c]))
            encoded.append(tuple(template))
        variants.append(tuple(encoded))
    return min(variants)


def _design_key(info, colored=False):
    """Remove translation, pair orientation and state ordering for measurements."""
    variants = []
    for rotation in range(4):
        for reflected in (False, True):
            maps = []
            for template in info["templates"]:
                transformed = {}
                for (y, x), c in template.items():
                    for _ in range(rotation):
                        y, x = x, -y
                    transformed[y, -x if reflected else x] = c
                maps.append(transformed)
            y0, _, x0, _ = _bounds(set(maps[0]) | set(maps[1]))
            maps = [{(y - y0, x - x0): c for (y, x), c in m.items()} for m in maps]
            if colored:
                variants.append(tuple(sorted(_signature(m) for m in maps)))
            else:
                variants.append(_vocabulary({"templates": maps}))
    return min(variants)


def task_features(pair):
    """Review measurements reconstructed from returned pixels, not sample metadata."""
    info = _analyze_pair(pair["input"], pair["output"])
    a, b = info["templates"]
    common = {p for p, c in a.items() if b.get(p) == c}
    carrier = set(info["carrier"])
    y0, y1, x0, x1 = _bounds(carrier)
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    for y, x in carrier:
        mask[y - y0, x - x0] = True
    scales = tuple(sorted(o["scale"] for o in info["occurrences"]))
    state_counts = tuple(sorted(sum(o["state"] == s for o in info["occurrences"]) for s in (0, 1)))
    return {
        "geometry": _canonical_geometry(carrier),
        "vocabulary": _design_key(info),
        "colored_vocabulary": (info["background"], _design_key(info, colored=True)),
        "palette": tuple(sorted(info["palette"])),
        "palette_size": len(info["palette"]),
        "stable_colors": len(info["stable_colors"]),
        "state_colors": tuple(sorted(len(set(c for p, c in t.items() if p not in common)) for t in (a, b))),
        "change": info["change"],
        "bbox_shift": info["bbox_shift"],
        "unequal_support": len(a) != len(b),
        "decoration_sizes": tuple(sorted((len(set(a) - set(b)), len(set(b) - set(a))))),
        "recolored_cells": sum(a[p] != b[p] for p in set(a) & set(b)),
        "carrier_components": len(_components(mask, diagonal=False)),
        "scales": scales,
        "count": len(scales),
        "state_counts": state_counts,
        "reference_scale": info["reference_scale"],
        "edge": info["edge"],
        "shape": pair["input"].shape,
        "background": info["background"],
        "accent_colors": tuple(sorted({c for t in (a, b) for p, c in t.items() if p not in common})),
    }


def self_test() -> None:
    explicit = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]
    for index, (nt, ns) in enumerate(explicit):
        p = generate_puzzle(21000 + index, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for nt, ns in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)]:
        try:
            generate_puzzle(22000, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    train_counts, test_counts, geometries, vocabularies, shapes = set(), set(), set(), set(), set()
    counts, scales, palettes, stable, changes, fingerprints = set(), set(), set(), set(), set(), set()
    within_geometry, within_palette, edges, elevated, repeated_large, compositional = 0, 0, 0, 0, 0, 0
    state_frequencies, combinations, backgrounds = set(), set(), set()
    bbox_modes, state_colors, decoration_sizes, recolored_counts = set(), set(), set(), set()
    disconnected = unequal = fresh_tests = repeated_puzzles = accent_reuse = 0
    split_shapes = {"train": set(), "test": set()}
    scale_by_split = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        key = tuple(_grid_key(p[r]) for s in ("train", "test") for p in puzzle[s] for r in ("input", "output"))
        assert key == tuple(_grid_key(p[r]) for s in ("train", "test") for p in again[s] for r in ("input", "output"))
        fingerprints.add(key)
        infos = _inspect_puzzle(puzzle)["pair_infos"]
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_geometry, local_palette = set(), set()
        measured = {s: [task_features(p) for p in puzzle[s]] for s in puzzle}
        train_designs = {f["colored_vocabulary"] for f in measured["train"]}
        fresh_tests += sum(f["colored_vocabulary"] not in train_designs for f in measured["test"])
        repeated_puzzles += len({f["vocabulary"] for s in measured for f in measured[s]}) == 1
        accent_reuse += any(f["background"] in measured["train"][0]["accent_colors"] for f in measured["test"])
        for split in ("train", "test"):
            for pair, info, feature in zip(puzzle[split], infos[split], measured[split]):
                geometry = _canonical_geometry(info["carrier"])
                geometries.add(geometry)
                local_geometry.add(geometry)
                vocabularies.add(_vocabulary(info))
                local_palette.add(info["palette"])
                shapes.add(pair["input"].shape)
                split_shapes[split].add(pair["input"].shape)
                os = info["occurrences"]
                schedule = tuple(sorted(o["scale"] for o in os))
                scales.add(schedule)
                scale_by_split[split].update(schedule)
                counts.add(len(os))
                state_frequencies.add(tuple(sorted(sum(o["state"] == s for o in os) for s in (0, 1))))
                palettes.add(len(info["palette"]))
                backgrounds.add(info["background"])
                stable.add(len(info["stable_colors"]))
                bbox_modes.add(info["bbox_shift"])
                disconnected += feature["carrier_components"] > 1
                unequal += feature["unequal_support"]
                state_colors.add(feature["state_colors"])
                decoration_sizes.add(feature["decoration_sizes"])
                recolored_counts.add(feature["recolored_cells"])
                changes.add((split, info["change"]))
                combinations.add((info["change"], len(info["palette"]), len(os), max(schedule)))
                edges += info["edge"]
                elevated += info["reference_scale"] > 1
                repeated_large += any(schedule.count(s) > 1 for s in set(schedule) if s > info["reference_scale"])
                if split == "test":
                    train = [{r: p[r].copy() for r in ("input", "output")} for p in puzzle["train"]]
                    assert np.array_equal(solve(train, pair["input"].copy()), pair["output"])
        within_geometry += len(local_geometry) > 1
        within_palette += len(local_palette) > 1
        compositional += {i["change"] for i in infos["train"]} == {"recolor", "relocate"} and all(i["change"] == "combined" for i in infos["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(fingerprints) == 42
    assert len(geometries) >= 40 and len(vocabularies) >= 100 and len(shapes) >= 70
    assert within_geometry >= 25 and within_palette >= 25
    assert counts >= {2, 3, 4, 5, 6} and len(scales) >= 30 and len(state_frequencies) >= 8
    assert palettes >= {4, 5, 6, 7, 8} and stable >= {1, 2, 3} and len(backgrounds) == 10
    assert changes == {(s, c) for s in ("train", "test") for c in ("recolor", "relocate", "combined")}
    assert edges >= 40 and elevated >= 5 and repeated_large >= 10 and compositional >= 5
    assert len(combinations) >= 80
    assert bbox_modes == {False, True} and unequal >= 30 and disconnected >= 30
    assert len(state_colors) >= 5 and len(decoration_sizes) >= 8 and len(recolored_counts) >= 5
    assert fresh_tests >= 50 and repeated_puzzles >= 3 and accent_reuse >= 10
    assert len(split_shapes["train"] & split_shapes["test"]) >= 10
    assert scale_by_split["train"] >= {1, 2, 3, 4} and scale_by_split["test"] >= {1, 2, 3, 4}
    assert (2, 2, 4) in scales and any(5 in s for s in scales)
    # Mutation checks exercise erasure, scale blocks, and unaffected background.
    sample = generate_puzzle(190)
    for point in ((0, 0), tuple(np.argwhere(sample["test"][0]["input"] != sample["test"][0]["output"])[0])):
        bad = {s: [{r: p[r].copy() for r in ("input", "output")} for p in sample[s]] for s in sample}
        g = bad["test"][0]["output"]
        g[point] = (int(g[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("corrupted output accepted")
    ambiguous = np.zeros((8, 18), dtype=int)
    ambiguous[1:4, 1:6] = [[0, 1, 1, 1, 0], [0, 0, 3, 0, 0], [0, 2, 2, 2, 0]]
    ambiguous[1:4, 10:15] = [[0, 1, 1, 1, 0], [0, 0, 4, 0, 0], [2, 2, 2, 0, 0]]
    clipped = np.zeros((13, 13), dtype=int)
    carrier = {(0, 1): 1, (1, 0): 1, (1, 2): 1, (2, 1): 1, (1, 1): 2}
    _render_template(clipped, carrier, (0, 0), 1)
    clipped[3, 1] = 3
    _render_template(clipped, carrier, (7, 7), 1)
    clipped[6, 8] = 4
    missing_state = np.zeros((8, 18), dtype=int)
    missing_state[1:4, 1:6] = ambiguous[1:4, 1:6]
    missing_state[1:4, 10:15] = ambiguous[1:4, 1:6]
    for malformed in (ambiguous, clipped, missing_state):
        try:
            _read_input(malformed)
        except AssertionError:
            continue
        raise AssertionError("ambiguous, clipped or incomplete input accepted")
