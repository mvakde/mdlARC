"""Aperture loom: periodic selectors activate compatible erasure markers.

Only standard-library modules and NumPy are used. Construction composes sampled
objects directly; solve reconstructs geometry and signals from input pixels.
"""
from itertools import permutations
import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 412779847, 163562139, 834103491, 345859974, 887421032,
              965129916, 80278580, 940107908, 418802070, 82030615, 426434614,
              146968097, 62500687, 912976378, 899918037, 196062654, 617930705,
              312538764, 72567464, 706520553, 452822842, 385963265, 283608048,
              762090262, 568220468, 182779014, 542506205, 296555222, 242912700,
              272662307, 248595346, 540374406, 76016676, 993696397, 404288087,
              827986760, 266032154, 649147270, 312725986, 724640957, 875231227)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(min(cc)), int(max(rr) + 1), int(max(cc) + 1)


def _collar(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _hub_mask(h, w, kind, arm=None):
    m = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = False
    elif kind == "cross":
        ah, aw = arm
        m[:] = False
        m[(h - ah)//2:(h + ah)//2, :] = True
        m[:, (w - aw)//2:(w + aw)//2] = True
    return m


def _hub_kind(m):
    h, w = m.shape
    if not (3 <= h <= 7 and 3 <= w <= 7):
        return None
    if m.all():
        return "rectangle"
    if np.array_equal(m, _hub_mask(h, w, "clipped")):
        return "clipped"
    for ah in range(1, h - 1):
        for aw in range(1, w - 1):
            if (h - ah) % 2 == (w - aw) % 2 == 0:
                if np.array_equal(m, _hub_mask(h, w, "cross", (ah, aw))):
                    return "cross"
    return None


def _maximum(shape, box, direction):
    r0, c0, r1, c1 = box
    return (r0, shape[0] - r1, c0, shape[1] - c1)[direction]


def _slice(box, direction, width, distance):
    r0, c0, r1, c1 = box
    if direction < 2:
        r = r0 - distance if direction == 0 else r1 - 1 + distance
        return [(r, c0 + (c1 - c0 - width)//2 + j) for j in range(width)]
    c = c0 - distance if direction == 2 else c1 - 1 + distance
    return [(r0 + (r1 - r0 - width)//2 + j, c) for j in range(width)]


def _period(word):
    # Complete words only: ABA has cyclic period three, not prefix period two.
    return next(p for p in range(1, len(word) + 1) if len(word) % p == 0
                and all(np.array_equal(row, word[i % p]) for i, row in enumerate(word)))


def _read_sources(g, bg, hub, fills):
    specs, source = [], set()
    groups = _components(g == hub)
    assert 1 <= len(groups) <= 3
    for cells in groups:
        box = _box(cells)
        r0, c0, r1, c1 = box
        kind = _hub_kind(g[r0:r1, c0:c1] == hub)
        assert kind is not None
        choices = []
        for direction in range(4):
            maximum = _maximum(g.shape, box, direction)
            if maximum < 1:
                continue
            transverse = c1 - c0 if direction < 2 else r1 - r0
            face = _slice(box, direction, transverse, 1)
            occupied = [j for j, q in enumerate(face) if int(g[q]) in fills]
            width = len(occupied)
            offset = (transverse - width)//2
            if not 1 <= width <= 3 or transverse - width < 2 or (transverse - width) % 2:
                continue
            if occupied != list(range(offset, offset + width)):
                continue
            dr, dc = DIRS[direction]
            if all((r - dr, c - dc) in cells for r, c in _slice(box, direction, width, 1)):
                choices.append((direction, width, maximum))
        assert len(choices) == 1, "Each hub needs one unambiguous attached word"
        direction, width, maximum = choices[0]
        rows, seed_cells = [], set()
        for d in range(1, maximum + 1):
            points = _slice(box, direction, width, d)
            values = [int(g[q]) for q in points]
            if all(v in fills for v in values):
                rows.append(values)
                seed_cells.update(points)
            else:
                assert all(v == bg for v in values), "The seed must end at a blank slice"
                break
        assert rows and len(rows) <= 8 and len(rows)*width <= 16
        assert maximum >= len(rows) + 2
        assert not source & (cells | seed_cells)
        source.update(cells | seed_cells)
        specs.append(dict(box=box, direction=direction, width=width, maximum=maximum,
                          word=np.asarray(rows, dtype=np.int8), hub_cells=cells,
                          seed_cells=seed_cells, kind=kind))
    return specs, source


def _paint_scene(shape, bg, rectangles):
    scene = np.full(shape, bg, dtype=np.int8)
    for r0, c0, r1, c1, color in rectangles:
        scene[r0:r1, c0:c1] = color
    return scene


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _geometry(rectangles, shape):
    if not 2 <= len(rectangles) <= 3:
        return False
    degrees = [0]*len(rectangles)
    for i, a in enumerate(rectangles):
        r0, c0, r1, c1, _ = a
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            return False
        if min(r1 - r0, c1 - c0) < 5 or (r1 - r0)*(c1 - c0) < 40:
            return False
        for j, b in enumerate(rectangles[:i]):
            if _overlap(a, b):
                if a[4] == b[4] or _overlap(a, b) < 4:
                    return False
                degrees[i] += 1
                degrees[j] += 1
    return sorted(degrees) == ([1, 1] if len(rectangles) == 2 else [1, 1, 2])


def _supported(scene, bg, rectangles):
    if not _geometry(rectangles, scene.shape) or np.count_nonzero(scene != bg) > scene.size*.44:
        return False
    for r0, c0, r1, c1, color in rectangles:
        visible = scene[r0:r1, c0:c1] == color
        if visible.sum() < 24 or visible.mean() < .4:
            return False
        if min(visible[0].sum(), visible[-1].sum(), visible[:, 0].sum(), visible[:, -1].sum()) < 4:
            return False
    return True


def _recover_scene(g, bg, fills, source):
    field = g.copy()
    for q in source:
        field[q] = bg
    boxes = []
    for color in fills:
        for cells in _components(field == color):
            if len(cells) < 18:
                continue
            # Singleton seam bumps/speckles cannot define an extreme side.
            row_counts = np.bincount([r for r, _ in cells], minlength=g.shape[0])
            col_counts = np.bincount([c for _, c in cells], minlength=g.shape[1])
            rr, cc = np.flatnonzero(row_counts >= 3), np.flatnonzero(col_counts >= 3)
            assert len(rr) and len(cc)
            boxes.append((int(rr[0]), int(cc[0]), int(rr[-1] + 1), int(cc[-1] + 1), color))
    assert _geometry(boxes, g.shape), "Recoverable overlapping rectangle chain required"
    scenes = {}
    for order in permutations(boxes):
        scene = _paint_scene(g.shape, bg, order)
        if not _supported(scene, bg, order):
            continue
        loss = int(np.count_nonzero(scene != field))
        scenes.setdefault(scene.tobytes(), (loss, scene, order))
    ranked = sorted(scenes.values(), key=lambda x: (x[0], tuple(x[2])))
    assert ranked and (len(ranked) == 1 or ranked[1][0] > ranked[0][0] + 1), "Ambiguous occlusion"
    loss, scene, rectangles = ranked[0]
    assert loss <= max(12, int(g.size*.08)), "Sparse damage required"
    # Each visible extreme remains supported after damage. One-cell boundary
    # alternatives are scored on raw input, not on the supplied answer.
    for i, a in enumerate(rectangles):
        r0, c0, r1, c1, color = a
        visible = scene[r0:r1, c0:c1] == color
        correct = field[r0:r1, c0:c1] == color
        assert (visible & correct).sum() >= .85*visible.sum()
        for side in range(4):
            for delta in (-1, 1):
                trial = list(rectangles)
                b = list(a)
                b[side] += delta
                trial[i] = tuple(b)
                if not _geometry(trial, g.shape):
                    continue
                alternative = _paint_scene(g.shape, bg, trial)
                if np.array_equal(alternative, scene) or not _supported(alternative, bg, trial):
                    continue
                assert np.count_nonzero(alternative != field) > loss + 1, "Uncertain rectangle edge"
    return scene, rectangles


def _signals(shape, specs, fills, prefix=False, reset_scene=None):
    signals = {color: np.zeros(shape, dtype=bool) for color in fills}
    coverage = np.zeros(shape, dtype=np.int8)
    for s in specs:
        word = s["word"]
        if prefix:
            p = next(p for p in range(1, len(word) + 1)
                     if all(np.array_equal(row, word[i % p]) for i, row in enumerate(word)))
            word = word[:p]
        entry = None
        for d in range(len(s["word"]) + 1, s["maximum"] + 1):
            points = _slice(s["box"], s["direction"], s["width"], d)
            if reset_scene is not None and entry is None and any(int(reset_scene[q]) in fills for q in points):
                entry = d
            phase = d - 1 if entry is None else d - entry
            for q, value in zip(points, word[phase % len(word)]):
                signals[int(value)][q] = True
                coverage[q] += 1
    return signals, coverage


def _ring(shape, p):
    r, c = p
    return [(rr, cc) for rr in range(max(0, r - 1), min(shape[0], r + 2))
            for cc in range(max(0, c - 1), min(shape[1], c + 2)) if (rr, cc) != p]


def _compose(base, centers, fills, hub):
    demands, center_set = {}, set(centers)
    for p in centers:
        ring_color = next(c for c in fills if c != int(base[p]))
        for q in _ring(base.shape, p):
            if q in center_set:
                continue
            assert q not in demands or demands[q] == ring_color, "Incompatible ring overlap"
            demands[q] = ring_color
    out = base.copy()
    for q, color in demands.items():
        out[q] = color
    for p in centers:
        out[p] = hub
    return out


def _analyze(g):
    g = _grid(g)
    colors, counts = np.unique(g, return_counts=True)
    assert len(colors) == 4
    bg = int(colors[np.argmax(counts)])
    assert np.count_nonzero(g == bg) > g.size//2, "Background must occupy a strict majority"
    candidates = []
    for hub in map(int, colors):
        if hub == bg:
            continue
        fills = tuple(int(c) for c in colors if c not in (bg, hub))
        try:
            specs, source = _read_sources(g, bg, hub, fills)
            scene, rectangles = _recover_scene(g, bg, fills, source)
            assert not any(scene[q] != bg for q in source)
            base = scene.copy()
            for q in source:
                base[q] = g[q]
            signals, coverage = _signals(g.shape, specs, fills)
            holes = [tuple(map(int, p)) for p in np.argwhere((scene != bg) & (g == bg))]
            centers = [p for p in holes if signals[int(scene[p])][p]]
            assert holes and centers
            assert not any(q in source for p in holes for q in _ring(g.shape, p))
            out = _compose(base, centers, fills, hub)
            candidates.append(dict(bg=bg, hub=hub, fills=fills, specs=specs, source=source,
                                   scene=scene, base=base, rectangles=rectangles, signals=signals,
                                   coverage=coverage, holes=holes, centers=centers, expected=out))
        except (AssertionError, StopIteration):
            continue
    assert len(candidates) == 1, "Unique geometric interpretation required"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the scene and evaluate virtual selectors using only this input."""
    return _analyze(input_grid)["expected"].copy()


def _sample_scene(rng, shape, count):
    h, w = shape
    rects = []
    for _ in range(180):
        rh = int(rng.integers(6, min(14, h - 3) + 1))
        rw = int(rng.integers(6, min(15, w - 3) + 1))
        r0 = int(rng.integers(h - rh + 1))
        c0 = int(rng.integers(w - rw + 1))
        box = (r0, c0, r0 + rh, c0 + rw, 1 + len(rects) % 2)
        if rects:
            overlap = _overlap(box, rects[-1])
            if not 4 <= overlap <= .45*min(rh*rw, (rects[-1][2] - rects[-1][0])*(rects[-1][3] - rects[-1][1])):
                continue
            if any(_overlap(box, a) for a in rects[:-1]):
                continue
        rects.append(box)
        if len(rects) == count:
            ordered = [rects[int(i)] for i in rng.permutation(count)]
            scene = _paint_scene(shape, 0, ordered)
            if _supported(scene, 0, ordered):
                components = [(*_box(cells), c) for c in (1, 2) for cells in _components(scene == c)]
                if sorted(components) == sorted(rects):
                    return scene
            return None
    return None


def _word(rng, width, period, style):
    word = rng.integers(1, 3, size=(period, width), dtype=np.int8)
    if style == "aba":
        a = int(rng.integers(1, 3))
        word[:] = a
        word[1] = 3 - a
        word[-1] = word[0]
    elif style == "transverse":
        word[:, 0], word[:, -1] = 1, 2
        if rng.integers(2):
            word = word[:, ::-1].copy()
    elif period > 1:
        # Permit repeated slices/colors, but make the whole requested word primitive.
        for _ in range(8):
            if _period(word) == period:
                break
            word = rng.integers(1, 3, size=(period, width), dtype=np.int8)
        if _period(word) != period:
            word[:] = 1
            word[-1, -1] = 2
    return word


def _new_spec(rng, scene, width, period, style, existing):
    direction = int(rng.integers(4))
    transverse = width + 2*int(rng.integers(1, 3))
    longitudinal = int(rng.integers(3, 6))
    h, w = (longitudinal, transverse) if direction < 2 else (transverse, longitudinal)
    r0 = int(rng.integers(scene.shape[0] - h + 1))
    c0 = int(rng.integers(scene.shape[1] - w + 1))
    box = (r0, c0, r0 + h, c0 + w)
    maximum = _maximum(scene.shape, box, direction)
    copies = 1 if style == "aba" else int(rng.integers(1, 3))
    visible_length = period*copies
    if visible_length*width > 16 or visible_length > 8:
        visible_length = period
    if maximum < visible_length + 4:
        return None
    kind = str(rng.choice(("rectangle", "clipped", "cross")))
    ah = [a for a in range(1, h - 1) if (h - a) % 2 == 0 and (direction < 2 or a >= width)]
    aw = [a for a in range(1, w - 1) if (w - a) % 2 == 0 and (direction >= 2 or a >= width)]
    mask = _hub_mask(h, w, kind, (int(rng.choice(ah)), int(rng.choice(aw))))
    cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
    tile = _word(rng, width, period, style)
    word = tile[np.arange(visible_length) % period].copy()
    seed = {q for d in range(1, visible_length + 1) for q in _slice(box, direction, width, d)}
    occupied = cells | seed
    collar = _collar(occupied)
    if any(0 <= r < scene.shape[0] and 0 <= c < scene.shape[1] and scene[r, c] != 0 for r, c in collar):
        return None
    if any(collar & (s["hub_cells"] | s["seed_cells"]) for s in existing):
        return None
    ray = {q for d in range(visible_length + 1, maximum + 1) for q in _slice(box, direction, width, d)}
    if sum(scene[q] != 0 for q in ray) < 4:
        return None
    if any(ray & (s["hub_cells"] | s["seed_cells"]) for s in existing):
        return None
    if any(occupied & {q for d in range(len(s["word"]) + 1, s["maximum"] + 1)
                       for q in _slice(s["box"], s["direction"], s["width"], d)} for s in existing):
        return None
    return dict(box=box, direction=direction, width=width, maximum=maximum, word=word,
                hub_cells=cells, seed_cells=seed, kind=kind)


def _fallback_sources(rng, widths, periods, styles):
    scene = _paint_scene((30, 30), 0, [(10, 10, 20, 24, 1), (16, 19, 27, 29, 2)])
    specs = []
    # A singleton constant selector must cross both fills; otherwise it cannot
    # exhibit both a matching and a nonmatching hole under the same ribbon.
    placements = [(16 if len(widths) == 1 else 11, 1, 3), (0, 19, 1), (23, 1, 3)]
    for width, period, style, (r0, c0, direction) in zip(widths, periods, styles, placements):
        h, w = (3, width + 2) if direction < 2 else (width + 2, 3)
        box = (r0, c0, r0 + h, c0 + w)
        mask = _hub_mask(h, w, "clipped")
        cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
        specs.append(dict(box=box, direction=direction, width=width,
                          maximum=_maximum(scene.shape, box, direction),
                          word=_word(rng, width, period, style), hub_cells=cells,
                          seed_cells={q for d in range(1, period + 1) for q in _slice(box, direction, width, d)},
                          kind="clipped"))
    return scene, specs


def _decorate(rng, scene, specs, teach, allow_multi):
    fills, hub = (1, 2), 3
    source = set().union(*(s["hub_cells"] | s["seed_cells"] for s in specs))
    base = scene.copy()
    for s in specs:
        for q in s["hub_cells"]:
            base[q] = hub
        for d, values in enumerate(s["word"], 1):
            for q, value in zip(_slice(s["box"], s["direction"], s["width"], d), values):
                base[q] = value
    signals, coverage = _signals(scene.shape, specs, fills)
    safe = [tuple(map(int, p)) for p in np.argwhere(scene != 0)
            if not any(q in source for q in _ring(scene.shape, tuple(p)))]
    matching = {p for p in safe if signals[int(scene[p])][p]}
    quiet_covered = [p for p in safe if coverage[p] and p not in matching]
    quiet_outside = [p for p in safe if not coverage[p]]
    rng.shuffle(quiet_covered)
    rng.shuffle(quiet_outside)
    if not matching or not quiet_covered or not quiet_outside:
        return None
    quiet = [quiet_covered[0], quiet_outside[0]]
    active, holes = [], quiet.copy()
    def add_active(p):
        if p not in matching or p in holes or (not allow_multi and coverage[p] > 1):
            return False
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2 for q in quiet):
            return False
        if any(scene[p] != scene[q] and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2 for q in active):
            return False
        active.append(p)
        holes.append(p)
        return True
    candidates = sorted(matching)
    rng.shuffle(candidates)
    if teach == "aba":
        shorter, _ = _signals(scene.shape, specs, fills, prefix=True)
        witness = [p for p in candidates if not shorter[int(scene[p])][p]]
        if not any(add_active(p) for p in witness):
            return None
    if teach:
        for color in fills:
            if not any(scene[p] == color for p in active):
                if not any(add_active(p) for p in candidates if scene[p] == color):
                    return None
    if teach == "transverse":
        # Adjacent same-fill activated centers expose simultaneous composition.
        found = False
        for p in candidates:
            for dr, dc in DIRS:
                q = (p[0] + dr, p[1] + dc)
                if q not in matching or scene[p] != scene[q]:
                    continue
                old = len(active)
                if (p in active or add_active(p)) and (q in active or add_active(q)):
                    found = True
                    break
                del holes[len(quiet) + old:]
                del active[old:]
            if found:
                break
        if not found:
            return None
    target = int(rng.integers(max(2, len(active)), max(4, len(active) + 1) + 5))
    for p in candidates:
        if len(active) >= target:
            break
        add_active(p)
    if not active:
        return None
    # Optional additional quiet holes, including cases under a nonmatching beam.
    quiet_pool = quiet_covered[1:] + quiet_outside[1:]
    rng.shuffle(quiet_pool)
    for p in quiet_pool[:int(rng.integers(4))]:
        if p not in holes:
            holes.append(p)
    inp = base.copy()
    for p in holes:
        inp[p] = 0
    # Foreground errors are not erasures, even where a selector agrees.
    wrong_pool = [p for p in safe if p not in holes and
                  all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in holes)]
    rng.shuffle(wrong_pool)
    wrong_pool.sort(key=lambda p: (not signals[int(scene[p])][p],
                                  not any(scene[q] in fills and scene[q] != scene[p] for q in _ring(scene.shape, p))))
    if not wrong_pool:
        return None
    for p in wrong_pool[:int(rng.integers(1, 5))]:
        inp[p] = 3 - int(scene[p])
    source_collar = _collar(source)
    exterior = [tuple(map(int, p)) for p in np.argwhere(base == 0)
                if tuple(p) not in source_collar and
                all(scene[q] == 0 for q in _ring(scene.shape, tuple(p)))]
    rng.shuffle(exterior)
    chosen = []
    speckle_target = int(rng.integers(1, 7))
    for p in exterior:
        if len(chosen) >= speckle_target:
            break
        if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 2 for q in chosen):
            inp[p] = int(rng.integers(1, 3))
            chosen.append(p)
    out = _compose(base, active, fills, hub)
    return dict(input=inp, output=out)


def _make_pair(rng, widths, periods, styles, count, teach, allow_multi, size_low):
    for attempt in range(100):
        if attempt < 80:
            shape = (int(rng.integers(size_low, 31)), int(rng.integers(size_low, 31)))
            scene = _sample_scene(rng, shape, count)
            if scene is None:
                continue
            specs = []
            for width, period, style in zip(widths, periods, styles):
                for _ in range(260):
                    spec = _new_spec(rng, scene, width, period, style, specs)
                    if spec is not None:
                        specs.append(spec)
                        break
                else:
                    break
            if len(specs) != len(widths):
                continue
        else:
            scene, specs = _fallback_sources(rng, widths, periods, styles)
        pair = _decorate(rng, scene, specs, teach, allow_multi)
        if pair is None:
            continue
        # Arbitrary injective pair-local palettes: all roles are visible.
        mapping = rng.choice(10, size=4, replace=False).astype(np.int8)
        turns = int(rng.integers(4))
        mirror = bool(rng.integers(2))
        pair = {k: np.ascontiguousarray(np.rot90(mapping[v], turns)) for k, v in pair.items()}
        intended_scene = np.ascontiguousarray(np.rot90(mapping[scene], turns))
        if mirror:
            pair = {k: np.ascontiguousarray(np.fliplr(v)) for k, v in pair.items()}
            intended_scene = np.ascontiguousarray(np.fliplr(intended_scene))
        try:
            analysis = _analyze(pair["input"])
            if analysis["bg"] != int(mapping[0]) or analysis["hub"] != int(mapping[3]):
                continue
            if not np.array_equal(analysis["scene"], intended_scene):
                continue
            if not np.array_equal(analysis["expected"], pair["output"]):
                continue
            if teach:
                witnesses = _witnesses(analysis, pair["input"])
                if not witnesses["wrong_on_match"] or (teach == "aba" and not witnesses["phase"]):
                    continue
            return pair
        except AssertionError:
            continue
    raise RuntimeError("Bounded construction exhausted")


def _witnesses(a, inp):
    centers = set(a["centers"])
    quiet = [p for p in a["holes"] if p not in centers and a["expected"][p] == a["scene"][p]]
    shorter, _ = _signals(inp.shape, a["specs"], a["fills"], prefix=True)
    reset, _ = _signals(inp.shape, a["specs"], a["fills"], reset_scene=a["scene"])
    wrong = [tuple(map(int, p)) for p in np.argwhere((inp != a["base"]) &
              (a["scene"] != a["bg"]) & (inp != a["bg"]))]
    return dict(both_fills={int(a["scene"][p]) for p in centers} == set(a["fills"]),
                quiet_covered=any(a["coverage"][p] > 0 for p in quiet),
                quiet_outside=any(a["coverage"][p] == 0 for p in quiet),
                whole_word=any(not shorter[int(a["scene"][p])][p] for p in centers),
                phase=any(not reset[int(a["scene"][p])][p] for p in centers),
                wrong_on_match=any(a["signals"][int(a["scene"][p])][p] and
                                   a["expected"][p] == a["scene"][p] for p in wrong),
                adjacent=any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                             for p in centers for q in centers if p != q),
                multi=any(all(a["signals"][c][p] for c in a["fills"]) for p in centers))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    for split, n in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(n):
            teach = "aba" if split == "train" and i == 0 else "transverse" if split == "train" and i == 1 else None
            if teach == "aba":
                widths, periods, styles = [1, 1], [3, 2], ["aba", "random"]
            elif teach == "transverse":
                widths, periods, styles = [2, 2], [1, 1], ["transverse", "transverse"]
            elif split == "train" and regime == 0:
                longitudinal = bool(rng.integers(2))
                widths = [1, 1] if longitudinal else [2, 2]
                periods = [int(rng.integers(2, 4))]*2 if longitudinal else [1, 1]
                styles = ["random", "random"] if longitudinal else ["transverse", "transverse"]
            else:
                nh = int(rng.integers(1, 4)) if split == "train" else (2 if regime == 0 else 3)
                width_high = 3 if split == "train" and regime == 1 else 4
                period_high = 4 if split == "train" and regime == 2 else 5
                widths = [int(rng.integers(1, width_high)) for _ in range(nh)]
                periods = [int(rng.integers(1, period_high)) for _ in range(nh)]
                styles = ["random"]*nh
                if split == "test" and regime == 0:
                    widths, periods, styles = [2, 2], [2, 2], ["random", "random"]
                elif split == "test" and regime == 1:
                    widths[0] = 3  # A larger vector than any training seed.
                elif split == "test" and regime == 2:
                    periods[0] = 4  # A longer primitive word than any training seed.
            train_analyses = [_analyze(p["input"]) for p in puzzle["train"]]
            taught_multi = any(_witnesses(a, p["input"])["multi"] for a, p in zip(train_analyses, puzzle["train"]))
            allow_multi = split == "train" or taught_multi
            for _ in range(8):
                pair = _make_pair(rng, widths, periods, styles, int(rng.integers(2, 4)), teach,
                                  allow_multi, 24 if split == "test" and regime == 1 else 20)
                if _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            assert np.array_equal(a["expected"], out), "Incorrect label"
            assert not np.array_equal(inp, out)
            assert all(out[q] == inp[q] for q in a["source"])
            assert all(out[p] == a["hub"] for p in a["centers"])
            # No painted virtual ribbon in uncorrupted exterior background.
            ring_cells = {q for p in a["centers"] for q in _ring(inp.shape, p)}
            unchanged_bg = (a["base"] == a["bg"])
            for q in ring_cells:
                unchanged_bg[q] = False
            assert np.all(out[unchanged_bg] == a["bg"])
            records[split].append((a, _witnesses(a, inp)))
    witnesses = [w for _, w in records["train"]]
    for key in ("both_fills", "quiet_covered", "quiet_outside", "whole_word", "phase", "wrong_on_match", "adjacent"):
        assert any(w[key] for w in witnesses), f"Missing training witness: {key}"
    assert any(s["width"] > 1 for a, _ in records["train"] for s in a["specs"])
    if any(w["multi"] for _, w in records["test"]):
        assert any(w["multi"] for w in witnesses), "Multiple selectors need a demonstration"
    train_specs = [s for a, _ in records["train"] for s in a["specs"]]
    for a, _ in records["test"]:
        test_specs = a["specs"]
        novel_combination = (not any(s["width"] > 1 and _period(s["word"]) > 1 for s in train_specs)
                             and any(s["width"] > 1 and _period(s["word"]) > 1 for s in test_specs))
        wider = max(s["width"] for s in test_specs) > max(s["width"] for s in train_specs)
        longer = max(_period(s["word"]) for s in test_specs) > max(_period(s["word"]) for s in train_specs)
        assert novel_combination or wider or longer, "Each test needs an observable compositional shift"


def self_test():
    assert _period(np.array([[1], [2], [1]])) == 3
    assert _period(np.array([[1], [2], [1], [2]])) == 2
    shapes, directions, widths, periods, counts, palettes, structures = set(), set(), set(), set(), set(), set(), set()
    total_holes, total_centers, clipped, multi, phase, repeated = set(), set(), 0, 0, 0, 0
    puzzles = []
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                a = _analyze(inp)
                w = _witnesses(a, inp)
                shapes.add(inp.shape)
                counts.add((len(a["rectangles"]), len(a["specs"])))
                palettes.add((a["bg"], a["hub"], *a["fills"]))
                total_holes.add(len(a["holes"]))
                total_centers.add(len(a["centers"]))
                roles = {}
                structures.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.ravel())))
                for s in a["specs"]:
                    directions.add(s["direction"])
                    widths.add(s["width"])
                    periods.add(_period(s["word"]))
                    repeated += len(set(map(tuple, s["word"]))) < len(s["word"])
                clipped += any(len(_ring(inp.shape, c)) < 8 for c in a["centers"])
                multi += w["multi"]
                phase += w["phase"]
                # Reverse center order: union of rings cannot create a priority.
                assert np.array_equal(_compose(a["base"], list(reversed(a["centers"])), a["fills"], a["hub"]), pair["output"])
                # Omitting A (mark every erasure) changes at least the quiet witnesses.
                if w["quiet_covered"]:
                    assert any(a["expected"][c] != a["hub"] for c in a["holes"] if c not in a["centers"])
                # Omitting B's local markers leaves an active center as an ordinary fill.
                assert any(a["expected"][c] != a["base"][c] for c in a["centers"])
    assert len(shapes) >= 45 and len(structures) == 210
    assert directions == {0, 1, 2, 3} and widths == {1, 2, 3} and periods == {1, 2, 3, 4}
    assert {r for r, _ in counts} == {2, 3} and {h for _, h in counts} == {1, 2, 3}
    assert len(palettes) >= 100 and len(total_holes) >= 8 and len(total_centers) >= 6
    assert phase >= 40 and multi > 0 and repeated > 0 and clipped > 0
    assert [_key(pair["input"]) for pair in puzzles[0]["train"]] == [_key(pair["input"]) for pair in generate_puzzle(0)["train"]]
    # Relabeling all colors must relabel the answer, without a hidden dictionary.
    mapping = np.array([9, 7, 8, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
    pair = puzzles[0]["test"][0]
    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    # Explicit clipped adjacent symbols: both centers survive their shared ring.
    corner = _compose(np.ones((4, 4), dtype=np.int8), [(0, 0), (0, 1)], (1, 2), 3)
    assert np.array_equal(corner, np.array([[3, 3, 2, 1], [2, 2, 2, 1],
                                          [1, 1, 1, 1], [1, 1, 1, 1]]))
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    try:
        _compose(np.array([[1, 1, 2, 2]]*4), [(1, 1), (1, 2)], (1, 2), 3)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incompatible rings accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "structures": len(structures),
            "object_count_combinations": sorted(counts), "palettes": len(palettes),
            "hole_counts": sorted(total_holes), "center_counts": sorted(total_centers),
            "phase_witness_pairs": phase, "multi_selector_pairs": multi, "clipped_pairs": clipped}
