"""Exchange positive stencil states inside their negative, framed replicas.

The public solver reads only pixels. Generation renders its labels from sampled
sets; decoding separately recovers specimens, frames, scale and registration.
"""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np


_STEPS8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)
_STEPS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pick(rng, values):
    values = list(values)
    return values[int(rng.integers(len(values)))]


def _bounds(points):
    return (int(min(y for y, x in points)), int(min(x for y, x in points)),
            int(max(y for y, x in points)) + 1, int(max(x for y, x in points)) + 1)


def _components(points, diagonal=True):
    remaining = set(points)
    result = []
    steps = _STEPS8 if diagonal else _STEPS4
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


def _connected(points):
    return bool(points) and len(_components(points)) == 1


def _transform_points(points, rotation, reflect):
    transformed = set(points)
    for _ in range(rotation):
        transformed = {(-x, y) for y, x in transformed}
    if reflect:
        transformed = {(y, -x) for y, x in transformed}
    return transformed


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(int, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _frame_rectangles(grid, bg):
    """A complete rim is one monochrome component, even with isolated fill islands."""
    candidates = []
    for color in sorted(int(c) for c in np.unique(grid) if c != bg):
        for component in _components(map(tuple, np.argwhere(grid == color)), False):
            y, x, b, r = _bounds(component)
            if min(b - y, r - x) < 4:
                continue
            crop = grid[y:b, x:r]
            if all(np.all(edge == color) for edge in (crop[0], crop[-1], crop[:, 0], crop[:, -1])):
                candidates.append((y, x, b, r, color))
    # A large plaque can contain smaller solid islands. Its outer rim wins.
    frames = [a for a in candidates if not any(
        a != b and b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]
        for b in candidates)]
    return sorted(frames)


def _separated(a, b):
    ay, ax, ab, ar = a[:4]
    by, bx, bb, br = b[:4]
    return ab < by or bb < ay or ar < bx or br < ax


def _read_input(input_grid):
    """Recover all local conventions and state templates without any labels."""
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    assert frequencies[bg] * 2 > grid.size, "background must be a strict majority"
    frames = _frame_rectangles(grid, bg)
    assert frames, "no plaques"
    excluded = np.zeros(grid.shape, dtype=bool)
    for i, frame in enumerate(frames):
        assert all(_separated(frame, other) for other in frames[:i]), "plaques touch or overlap"
        y, x, b, r, _ = frame
        excluded[y:b, x:r] = True
    free = [set(group) for group in _components(map(tuple, np.argwhere((grid != bg) & ~excluded)))]
    assert len(free) >= 2, "two specimens per class are required"
    color_sets = [{int(grid[p]) for p in group} for group in free]
    assert all(len(colors) == 2 for colors in color_sets), "specimens must be bicolored"
    rim_colors = {f[4] for f in frames}
    common_colors = set.intersection(*color_sets) - rim_colors
    assert len(common_colors) == 1, "carrier color is ambiguous"
    carrier_color = common_colors.pop()
    assert carrier_color != bg and carrier_color not in rim_colors
    catalog = {}
    specimen_boxes = []
    for group, colors in zip(free, color_sets):
        assert carrier_color in colors
        color = next(iter(colors - {carrier_color}))
        carrier = {p for p in group if grid[p] == carrier_color}
        oy, ox, _, _ = _bounds(carrier)
        core = frozenset((y - oy, x - ox) for y, x in carrier)
        state = frozenset((y - oy, x - ox) for y, x in group if grid[y, x] == color)
        assert len(core) >= 3 and state
        catalog.setdefault(color, []).append((core, state))
        specimen_boxes.append(_bounds(group))
    for i, box in enumerate(specimen_boxes):
        assert all(_separated(box, other) for other in specimen_boxes[:i]), "specimen boxes touch"
        assert all(_separated(box, f) for f in frames), "specimen too close to plaque"
    designs = {}
    for color, specimens in catalog.items():
        assert len(specimens) == 2, "each class must show exactly two specimens"
        (core, a), (other_core, b) = specimens
        assert core == other_core, "scaffolds disagree after registration"
        assert a != b and not core & (a | b), "states must differ and avoid scaffold"
        window = _bounds(core | a | b)
        designs[color] = {"core": core, "states": (a, b), "window": window}
    assert rim_colors <= set(designs), "a plaque has no matching specimens"
    occurrences = []
    expected = grid.copy()
    for y, x, bottom, right, color in frames:
        design = designs[color]
        ymin, xmin, ymax, xmax = design["window"]
        lh, lw = ymax - ymin + 2, xmax - xmin + 2
        h, w = bottom - y, right - x
        assert h % lh == 0 and w % lw == 0 and h // lh == w // lw, "nonuniform scale"
        scale = h // lh
        assert scale >= 1
        crop = grid[y:bottom, x:right]
        small = crop[::scale, ::scale]
        assert np.array_equal(np.repeat(np.repeat(small, scale, 0), scale, 1), crop), "broken pixel blocks"
        border = np.ones(small.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        assert np.all(small[border] == color), "damaged rim"
        inside = small[1:-1, 1:-1]
        assert set(np.unique(inside)) <= {bg, carrier_color, color}
        core = frozenset((int(yy) + ymin, int(xx) + xmin)
                         for yy, xx in np.argwhere(inside == carrier_color))
        holes = frozenset((int(yy) + ymin, int(xx) + xmin)
                          for yy, xx in np.argwhere(inside == bg))
        assert core == design["core"], "plaque scaffold is not registered"
        assert holes in design["states"], "cavities do not depict a specimen"
        state = design["states"].index(holes)
        target = design["states"][1 - state]
        replacement = np.full((lh, lw), color, dtype=int)
        for yy, xx in core:
            replacement[yy - ymin + 1, xx - xmin + 1] = carrier_color
        for yy, xx in target:
            replacement[yy - ymin + 1, xx - xmin + 1] = bg
        expected[y:bottom, x:right] = np.repeat(np.repeat(replacement, scale, 0), scale, 1)
        occurrences.append({"box": (y, x, bottom, right), "color": color,
                            "scale": scale, "state": state, "holes": holes,
                            "target": target, "rim": border})
    return {"expected": expected, "background": bg, "carrier_color": carrier_color,
            "designs": designs, "occurrences": occurrences, "frames_mask": excluded,
            "specimen_boxes": specimen_boxes}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only negative-space exchange, with no RNG or stored examples."""
    return _read_input(input_grid)["expected"].copy()


def _frontier(points, bounds):
    y0, x0, y1, x1 = bounds
    return sorted({(y + dy, x + dx) for y, x in points for dy, dx in _STEPS8
                   if y0 <= y + dy < y1 and x0 <= x + dx < x1} - set(points))


def _scaffold(rng, h, w):
    cells = {(y, x) for y in range(h) for x in range(w)}
    target = int(rng.integers(max(3, h, w), min(9, h * w - 1) + 1))
    # Deletion preserves a visible bounding box and allows diagonally connected cores.
    for index in rng.permutation(h * w):
        if len(cells) <= target:
            break
        p = (int(index) // w, int(index) % w)
        trial = cells - {p}
        if _bounds(trial) == (0, 0, h, w) and _connected(trial):
            cells = trial
    return cells


def _grow_state(rng, core, bounds, count, initial=()):
    state = set(initial)
    while len(state) < count:
        candidates = [p for p in _frontier(core | state, bounds) if p not in core]
        if not candidates:
            break
        state.add(_pick(rng, candidates))
    return state


def _design(rng, *, small=False, witness=False):
    for _ in range(256):
        h, w = (2, 2) if small else (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        if witness:
            h, w = _pick(rng, ((2, 3), (3, 2), (3, 3)))
        core = _scaffold(rng, h, w)
        bounds = (-1, -1, h + 1, w + 1)
        mode = 0 if witness else int(rng.integers(5))
        start = {_pick(rng, _frontier(core, bounds))}
        count = int(rng.integers(2, 7 if mode != 4 else 11))
        a = _grow_state(rng, core, bounds, count, start)
        if mode == 1:
            b = set(a)
            p = _pick(rng, _frontier(core | a, bounds))
            b.add(p)
        elif mode == 2:
            b = _grow_state(rng, core, bounds, int(rng.integers(1, 7)))
        elif mode == 3:
            # A bar on one side versus a freshly grown attachment.
            side = int(rng.integers(4))
            a = ({(-1, x) for x in range(w)} if side == 0 else
                 {(h, x) for x in range(w)} if side == 1 else
                 {(y, -1) for y in range(h)} if side == 2 else
                 {(y, w) for y in range(h)})
            b = _grow_state(rng, core, bounds, int(rng.integers(1, 7)))
        else:
            b = _grow_state(rng, core, bounds, int(rng.integers(2, 8)), start)
        if a == b or not _connected(core | a) or not _connected(core | b):
            continue
        window = _bounds(core | a | b)
        y0, x0, y1, x1 = window
        neutral = (y1 - y0) * (x1 - x0) - len(core | a | b)
        if witness and not (a & b and a - b and b - a and len(a) != len(b)
                            and _bounds(core | a)[:2] != _bounds(core | b)[:2] and neutral > 0):
            continue
        return {"core": core, "states": (a, b), "window": window}
    raise RuntimeError("bounded stencil sampling exhausted")


def _positive(design, state, color, carrier, bg):
    core, payload = design["core"], design["states"][state]
    y0, x0, y1, x1 = _bounds(core | payload)
    grid = np.full((y1 - y0, x1 - x0), bg, dtype=int)
    for y, x in core:
        grid[y - y0, x - x0] = carrier
    for y, x in payload:
        grid[y - y0, x - x0] = color
    return grid


def _fork_design(rng, design):
    """Same current holes/scaffold/window, but a color-selected different target."""
    core = design["core"]
    a, b = design["states"]
    candidates = [p for p in _frontier(core | b, design["window"]) if p not in core]
    if not candidates:
        return None
    other = b | {_pick(rng, candidates)}
    if other == a:
        return None
    return {"core": set(core), "states": (set(a), other), "window": design["window"]}


def _negative(design, state, color, carrier, bg, scale):
    y0, x0, y1, x1 = design["window"]
    grid = np.full((y1 - y0 + 2, x1 - x0 + 2), color, dtype=int)
    for y, x in design["core"]:
        grid[y - y0 + 1, x - x0 + 1] = carrier
    for y, x in design["states"][state]:
        grid[y - y0 + 1, x - x0 + 1] = bg
    return np.repeat(np.repeat(grid, scale, 0), scale, 1)


def _shelf(sizes, order, width, gap):
    rectangles = [None] * len(sizes)
    y = x = height = 0
    for i in order:
        h, w = sizes[i]
        if w > width:
            return None
        if x and x + w > width:
            y, x, height = y + height + gap, 0, 0
        if y + h > 30:
            return None
        rectangles[i] = (y, x, y + h, x + w)
        x += w + gap
        height = max(height, h)
    return rectangles


def _layout(rng, arrays, bg, max_foreground):
    sizes = [a.shape for a in arrays]
    if max_foreground >= 450:
        return None
    options = []
    orders = [sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True),
              list(map(int, rng.permutation(len(sizes))))]
    for gap in (1, 2, 3):
        for width in range(max(w for h, w in sizes), 31):
            for order in orders:
                placed = _shelf(sizes, order, width, gap)
                if placed is not None:
                    options.append(placed)
    if not options:
        return None
    rects = list(_pick(rng, options))
    h0, w0 = max(b for y, x, b, r in rects), max(r for y, x, b, r in rects)
    h = int(rng.integers(h0, min(30, h0 + 7) + 1))
    w = int(rng.integers(w0, min(30, w0 + 7) + 1))
    while h * w <= 2 * max_foreground:
        if h < 30 and (h <= w or w == 30):
            h += 1
        else:
            w += 1
    dy, dx = int(rng.integers(h - h0 + 1)), int(rng.integers(w - w0 + 1))
    rects = [(y + dy, x + dx, b + dy, r + dx) for y, x, b, r in rects]
    for _ in range(50):
        i = int(rng.integers(len(rects)))
        rh, rw = sizes[i]
        y, x = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
        new = (y, x, y + rh, x + rw)
        if all(_separated(new, other) for j, other in enumerate(rects) if j != i):
            rects[i] = new
    if rng.random() < .45:
        side = int(rng.integers(4))
        dy = -min(q[0] for q in rects) if side == 0 else h - max(q[2] for q in rects) if side == 1 else 0
        dx = -min(q[1] for q in rects) if side == 2 else w - max(q[3] for q in rects) if side == 3 else 0
        rects = [(y + dy, x + dx, b + dy, r + dx) for y, x, b, r in rects]
    return (h, w), rects


def _make_pair(rng, profile, background=None):
    for _ in range(160):
        if profile == "witness":
            active, dormant = 1, 0
        elif profile == "multi":
            active, dormant = 2, int(rng.random() < .25)
        elif profile == "upscale":
            active, dormant = 1, int(rng.random() < .2)
        elif profile == "combine":
            active, dormant = int(_pick(rng, (2, 2, 3))), 0
        else:
            active, dormant = int(_pick(rng, (1, 1, 2, 3))), int(rng.random() < .2)
        palette = list(map(int, rng.permutation(10)))
        if background is not None:
            palette.remove(background)
            palette.insert(0, background)
        bg, carrier = palette[:2]
        colors = palette[2:2 + active + dormant]
        designs = [_design(rng, small=profile == "upscale" or active >= 3,
                           witness=profile == "witness") for _ in colors]
        if profile == "multi":
            fork = _fork_design(rng, designs[0])
            if fork is None:
                continue
            designs[1] = fork
        assignments = []
        if profile == "witness":
            assignments = [(0, 0, 1), (0, 1, 2)]
        elif profile == "upscale":
            assignments = [(0, int(rng.integers(2)), 3)]
            if rng.random() < .8:
                assignments.append((0, int(rng.integers(2)), 1))
        else:
            assignments = [(i, 0 if profile == "multi" else int(rng.integers(2)), 1)
                           for i in range(active)]
            for _ in range(int(rng.integers(0, 3))):
                assignments.append((int(rng.integers(active)), int(rng.integers(2)), 1))
            if profile == "combine" or (active == 1 and profile != "multi" and rng.random() < .8):
                i, state, _ = assignments[0]
                assignments[0] = (i, state, 2)
        inputs, outputs = [], []
        for color, design in zip(colors, designs):
            for state in (0, 1):
                specimen = _positive(design, state, color, carrier, bg)
                inputs.append(specimen)
                outputs.append(specimen.copy())
        for i, state, scale in assignments:
            inputs.append(_negative(designs[i], state, colors[i], carrier, bg, scale))
            outputs.append(_negative(designs[i], 1 - state, colors[i], carrier, bg, scale))
        foreground = max(sum(np.count_nonzero(a != bg) for a in arrays) for arrays in (inputs, outputs))
        arrangement = _layout(rng, inputs, bg, foreground)
        if arrangement is None:
            continue
        shape, rectangles = arrangement
        inp, out = np.full(shape, bg, dtype=int), np.full(shape, bg, dtype=int)
        for rect, a, b in zip(rectangles, inputs, outputs):
            y, x, bottom, right = rect
            inp[y:bottom, x:right] = a
            out[y:bottom, x:right] = b
        # A whole-scene orientation changes visible geometry, never lookup conventions.
        orientations = [(r, f) for r in range(4) for f in (False, True)]
        if profile == "witness":
            d = designs[0]
            a, b = (d["core"] | state for state in d["states"])
            orientations = [(r, f) for r, f in orientations if
                _bounds(_transform_points(a, r, f))[:2] !=
                _bounds(_transform_points(b, r, f))[:2]]
        rotation, reflect = _pick(rng, orientations)
        inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
        if reflect:
            inp, out = np.fliplr(inp), np.fliplr(out)
        return {"input": inp.copy(), "output": out.copy()}
    raise RuntimeError("bounded scene packing exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    curriculum = "upscale" if rng.random() < .5 else "combine"
    puzzle, seen = {"train": [], "test": []}, set()
    first_background = None
    for index in range(nt + ne):
        split = "train" if index < nt else "test"
        profile = ("witness" if index == 0 else "multi" if index == 1 else
                   "varied" if split == "train" else curriculum)
        bg = None
        if index == 1:
            bg = int(_pick(rng, [c for c in range(10) if c != first_background]))
        for _ in range(32):
            pair = _make_pair(rng, profile, bg)
            signature = _key(pair["input"])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                if index == 0:
                    first_background = int(np.argmax(np.bincount(pair["input"].ravel(), minlength=10)))
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    result = {"classes": len(info["designs"]), "active": len({o["color"] for o in info["occurrences"]}),
              "plaques": len(info["occurrences"]), "scales": tuple(sorted(o["scale"] for o in info["occurrences"]))}
    result["sizes"] = tuple(sorted((d["window"][2] - d["window"][0], d["window"][3] - d["window"][1])
                                   for d in info["designs"].values()))
    return result


def _has_color_fork(info):
    for i, a in enumerate(info["occurrences"]):
        da = info["designs"][a["color"]]
        for b in info["occurrences"][i + 1:]:
            db = info["designs"][b["color"]]
            if (a["color"] != b["color"] and da["core"] == db["core"]
                    and da["window"] == db["window"] and a["holes"] == b["holes"]
                    and a["target"] != b["target"]):
                return True
    return False


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            assert np.array_equal(info["expected"], out), "incorrect negative-space state exchange"
            assert np.any(inp != out), "no state was exchanged"
            assert np.array_equal(inp[~info["frames_mask"]], out[~info["frames_mask"]]), "specimens or scene changed"
            assert np.array_equal(inp == info["carrier_color"], out == info["carrier_color"]), "scaffold moved"
            # Read the output as an input: the entire transformation is an involution.
            reverse = _read_input(out)
            assert np.array_equal(reverse["expected"], inp), "exchange is not reversible"
            for o in info["occurrences"]:
                y, x, b, r = o["box"]
                # Restoration and carving are set differences, never recoloring the rim.
                s = o["scale"]
                assert all(np.array_equal(a, z) for a, z in zip(
                    (inp[y:y+s, x:r], inp[b-s:b, x:r], inp[y:b, x:x+s], inp[y:b, r-s:r]),
                    (out[y:y+s, x:r], out[b-s:b, x:r], out[y:b, x:x+s], out[y:b, r-s:r])))
            infos[split].append(info)
    # Both directions and a nontrivial registration/occupancy witness occur even at 2/1.
    assert any({o["state"] for o in i["occurrences"]} == {0, 1} for i in infos["train"])
    assert any(o["scale"] > 1 for i in infos["train"] for o in i["occurrences"])
    assert any(_features(i)["active"] >= 2 for i in infos["train"])
    assert any(_has_color_fork(i) for i in infos["train"]), "color matching is not consequential"
    witness = False
    for info in infos["train"]:
        for design in info["designs"].values():
            core, (a, b) = design["core"], design["states"]
            y, x, bottom, right = design["window"]
            witness |= bool(a & b and a - b and b - a and len(a) != len(b)
                            and _bounds(core | a)[:2] != _bounds(core | b)[:2]
                            and (bottom - y) * (right - x) > len(core | a | b))
    assert witness, "training omits overlap, changed extent or neutral material"
    training_features = [_features(i) for i in infos["train"]]
    for info in infos["test"]:
        f = _features(info)
        assert max(f["scales"]) >= 3 or (f["active"] >= 2 and max(f["scales"]) >= 2)
        assert all(max(t["scales"]) < 3 and (t["active"] == 1 or max(t["scales"]) == 1)
                   for t in training_features), "curriculum shift lost"


def _fingerprint(puzzle):
    return tuple(_key(p[r]) for split in ("train", "test") for p in puzzle[split] for r in ("input", "output"))


def self_test():
    """Bounded, family-specific audit over 42 seeds; returns compact measurements."""
    seen, shapes, core_shapes, payloads, palettes = set(), set(), set(), set(), set()
    classes, plaque_counts, scales, windows, backgrounds = set(), set(), set(), set(), set()
    active_counts, dormant, edge, shifted, overlap, neutral, restores, creates = set(), 0, 0, 0, 0, 0, 0, 0
    regimes, disconnected, color_forks = set(), 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _fingerprint(p) == _fingerprint(generate_puzzle(seed)), "nondeterminism"
        fingerprint = hashlib.sha256(repr(_fingerprint(p)).encode()).hexdigest()
        seen.add(fingerprint)
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
                info = _read_input(inp)
                color_forks += _has_color_fork(info)
                f = _features(info)
                shapes.add(inp.shape); classes.add(f["classes"]); active_counts.add(f["active"])
                plaque_counts.add(f["plaques"]); scales.update(f["scales"]); windows.update(f["sizes"])
                backgrounds.add(info["background"]); palettes.add(tuple(sorted(np.unique(inp))))
                dormant += f["classes"] > f["active"]
                if split == "test":
                    regimes.add("upscale" if max(f["scales"]) == 3 else "combine")
                bg = info["background"]
                edge += any(np.any(a != bg) for a in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
                for d in info["designs"].values():
                    core, (a, b) = d["core"], d["states"]
                    core_shapes.add(tuple(sorted(core)))
                    payloads.add((tuple(sorted(core)), tuple(sorted((tuple(sorted(a)), tuple(sorted(b)))))))
                    disconnected += len(_components(core, False)) > 1
                    shifted += _bounds(core | a) != _bounds(core | b)
                    overlap += bool(a & b)
                    y, x, bottom, right = d["window"]
                    neutral += (bottom - y) * (right - x) > len(core | a | b)
                for o in info["occurrences"]:
                    restores += bool(o["holes"] - o["target"])
                    creates += bool(o["target"] - o["holes"])
                # Parent ablations: no exchange leaves the input; positive copying
                # writes class color where a target hole is required.
                assert not np.array_equal(inp, out), "B ablation did not change answer"
                bad = out.copy()
                o = info["occurrences"][0]
                y0, x0, _, _ = info["designs"][o["color"]]["window"]
                yy, xx = min(o["target"])
                y = o["box"][0] + (yy - y0 + 1) * o["scale"]
                x = o["box"][1] + (xx - x0 + 1) * o["scale"]
                bad[y, x] = o["color"]
                assert not np.array_equal(bad, out), "A ablation did not change answer"
    assert len(seen) == 42 and len(shapes) >= 30
    assert len(core_shapes) >= 30 and len(payloads) >= 150 and len(palettes) >= 30
    assert {1, 2, 3} <= active_counts and {1, 2, 3} <= scales and len(windows) >= 12
    assert {1, 2, 3, 4} <= plaque_counts and backgrounds == set(range(10))
    assert min(dormant, edge, shifted, overlap, neutral, restores, creates, disconnected) > 0
    assert regimes == {"upscale", "combine"}
    assert color_forks >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p = generate_puzzle(19, 2, 1)
    for location in ("hole", "rim", "specimen"):
        bad = copy.deepcopy(p)
        pair = bad["test"][0]
        info = _read_input(pair["input"])
        if location == "hole":
            y, x = map(int, np.argwhere(pair["input"] != pair["output"])[0])
        elif location == "rim":
            y, x = info["occurrences"][0]["box"][:2]
        else:
            mask = (pair["input"] != info["background"]) & ~info["frames_mask"]
            y, x = map(int, np.argwhere(mask)[0])
        pair["output"][y, x] = (int(pair["output"][y, x]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted damaged {location}")
    # Missing references, a changed scaffold and a broken enlargement block
    # must fail from input pixels, before any output comparison is possible.
    source = p["train"][0]["input"]
    info = _read_input(source)
    for mode in ("missing_specimen", "scaffold", "pixel_block"):
        damaged = source.copy()
        if mode == "missing_specimen":
            y, x, b, r = info["specimen_boxes"][0]
            damaged[y:b, x:r] = info["background"]
        else:
            o = next(o for o in info["occurrences"] if o["scale"] == 2)
            d = info["designs"][o["color"]]
            y0, x0, _, _ = d["window"]
            yy, xx = min(d["core"] if mode == "scaffold" else o["holes"])
            y = o["box"][0] + (yy - y0 + 1) * 2
            x = o["box"][1] + (xx - x0 + 1) * 2
            extent = 2 if mode == "scaffold" else 1
            damaged[y:y+extent, x:x+extent] = o["color"]
        try:
            solve(damaged)
        except AssertionError:
            continue
        raise AssertionError(f"accepted malformed input: {mode}")
    report = {"seeds": 42, "puzzles": len(seen), "input_shapes": len(shapes),
              "scaffold_geometries": len(core_shapes), "state_vocabularies": len(payloads),
              "palette_sets": len(palettes), "class_counts": sorted(classes),
              "active_class_counts": sorted(active_counts), "plaque_counts": sorted(plaque_counts),
              "scales": sorted(scales), "window_shapes": len(windows),
              "backgrounds": sorted(backgrounds), "dormant_class_pairs": dormant,
              "edge_pairs": edge, "different_specimen_bounds": shifted,
              "overlapping_state_pairs": overlap, "neutral_windows": neutral,
              "restoring_occurrences": restores, "carving_occurrences": creates,
              "disconnected_4_scaffolds": disconnected, "test_regimes": sorted(regimes),
              "color_fork_pairs": color_forks,
              "independent_semantic_verification": False}
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
