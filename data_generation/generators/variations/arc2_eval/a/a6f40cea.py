"""Complete rectangular outlines hidden by the largest opaque rectangle.

Colors carry no fixed meaning. Geometry, visible repetitions, and uniform fills
identify the three layers. The input-only reader below is deliberately separate
from the scene renderer: neither output sizes nor scene descriptors locate a mask.
"""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "a5fc6146ec688e91a55a17d0b8b9860227e852b73c3f6f822697dabced0708f8",
    "6e0b78b65632c53727bd43af158210b48cddcd8d61cb7a87155d70ee48cb9144",
    "306a85c8af456af4547886e5fa41ca9bfcff35f104b9eee99198fca14f496407",
    "36fb102a2d9b2492989d09ed1ab1774d6b6dd5b7e8ad5c98e9801dd9e63cbea3",
}


def _integer_argument(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed_key, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer_argument(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")
    if num_test is not None:
        num_test = _integer_argument(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")
    if num_train is None and num_test is None:
        num_test = 1 + seed_key % 4
        low = max(2, num_test + 1)
        num_train = low + (seed_key // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


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
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _size(box):
    t, b, l, r = box
    return (b - t + 1) * (r - l + 1)


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    group.add(q)
        result.append(group)
    return result


def _index(box, point, model):
    y, x = point
    if model == "diagonal":
        return y + x
    if model == "antidiagonal":
        return y - x
    t, b, l, r = box
    if y == t:
        return x - l
    if x == r:
        return r - l + y - t
    if y == b:
        return r - l + b - t + r - x
    return 2 * (r - l) + b - t + b - y


def _fits(grid, box, visible, hidden):
    """All supported short periodic explanations must predict the same cells.

    Each residue needs two visible witnesses; a word never comes from an erased
    cell. Trying both diagonals and a perimeter walk also checks turn ambiguity.
    """
    predictions, fits = set(), []
    ordered = sorted(hidden)
    for model in ("diagonal", "antidiagonal", "walk"):
        indices = [(_index(box, p, model), int(grid[p])) for p in sorted(visible)]
        for period in range(1, 7):
            word, counts = {}, [0] * period
            for index, color in indices:
                phase = index % period
                if phase in word and word[phase] != color:
                    break
                word[phase] = color
                counts[phase] += 1
            else:
                if min(counts) >= 2:
                    prediction = tuple(word[_index(box, p, model) % period] for p in ordered)
                    predictions.add(prediction)
                    fits.append((period, model))
    return predictions, sorted(fits)


def _frames(grid):
    """Find filled rectangles from rectangular monochrome components, no label."""
    bg = int(grid[0, 0])
    h, w = grid.shape
    frames = []
    for color in np.unique(grid):
        points = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        for group in _components(points):
            t, b, l, r = _bbox(group)
            if len(group) != (b - t + 1) * (r - l + 1):
                continue
            if t == 0 or l == 0 or b == h - 1 or r == w - 1:
                continue
            box = (t - 1, b + 1, l - 1, r + 1)
            colors = {int(grid[p]) for p in _perimeter(box)}
            if bg not in colors and int(color) not in colors:
                frames.append({"box": box, "fill": int(color),
                               "colors": tuple(sorted(colors)), "interior_area": len(group)})
    return sorted(frames, key=lambda f: (-f["interior_area"], f["box"]))


def _read_input(grid):
    """Recover a scene using only its visible pixels, including split outlines."""
    bg = int(grid[0, 0])
    assert int(grid[0, -1]) == bg and int(grid[-1, 0]) == bg and int(grid[-1, -1]) == bg
    frames = _frames(grid)
    assert frames, "no complete opaque frame"
    main = frames[0]
    assert len(frames) == 1 or main["interior_area"] > frames[1]["interior_area"], "largest frame tie"
    # Uniform background-filled complete middle loops remain transparent.
    opaque = [f for f in frames if f is main or f["fill"] != bg]
    masked = set()
    for f in opaque:
        cells = _area(f["box"])
        assert not (masked & cells), "opaque rectangles overlap"
        masked.update(cells)
    points = {tuple(map(int, p)) for p in np.argwhere(grid != bg)} - masked
    components = _components(points)
    candidates = []
    # A single rectangular mask cuts a loop into at most two connected pieces.
    for i, first in enumerate(components):
        for j in range(i, len(components)):
            visible = first if i == j else first | components[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b - t < 2 or r - l < 2:
                continue
            full = _perimeter(box)
            if full - masked != visible:
                continue
            hidden = full & _area(main["box"])
            predictions, fits = _fits(grid, box, visible, hidden)
            if hidden and len(predictions) != 1:
                continue
            assignments = dict(zip(sorted(hidden), next(iter(predictions)))) if hidden else {}
            candidates.append({"parts": frozenset((i, j)), "box": box,
                               "visible": visible, "hidden": hidden,
                               "assignments": assignments, "fits": fits,
                               "colors": tuple(sorted({int(grid[p]) for p in visible}))})
    covers = []

    def search(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for candidate in candidates:
            parts = candidate["parts"]
            full = _perimeter(candidate["box"])
            if first in parts and parts <= remaining and not (full & _halo(occupied)):
                search(remaining - parts, chosen + [candidate], occupied | full)

    search(set(range(len(components))), [], set())
    assert len(covers) == 1, "outline geometry or periodic completion is ambiguous"
    outlines = covers[0]
    t, b, l, r = main["box"]
    output = np.full((b - t - 1, r - l - 1), main["fill"], dtype=np.int8)
    middle_colors = set()
    for outline in outlines:
        middle_colors.update(outline["colors"])
        for (y, x), color in outline["assignments"].items():
            if t < y < b and l < x < r:
                output[y - t - 1, x - l - 1] = color
    assert bg not in middle_colors and main["fill"] not in middle_colors
    assert middle_colors.isdisjoint(main["colors"])
    return output, {"background": bg, "main": main, "opaque": opaque, "outlines": outlines}


def predict(train_pairs, test_input):
    """Prediction API: labels are never provided for the queried input."""
    assert len(train_pairs) >= 2
    return _read_input(test_input)[0]


def _word(rng, palette, kind, max_period):
    colors = [int(c) for c in rng.permutation(palette)]
    if kind == "solid":
        return (colors[0],)
    if kind == "alternating":
        return tuple(colors[:2])
    period = int(rng.integers(3, max_period + 1))
    count = int(rng.integers(2, min(len(colors), period) + 1))
    # Repeated arbitrary words include cycles, runs, palindromes, and ABAC-like
    # motifs. Every chosen color actually occurs in the word.
    word = colors[:count] + [colors[int(rng.integers(count))] for _ in range(period - count)]
    rng.shuffle(word)
    return tuple(word)


def _paint_boundary(grid, box, rng, palette, kind, model, max_period):
    cells = sorted(_perimeter(box))
    if kind == "random":
        values = rng.choice(palette, size=len(cells))
        for p, color in zip(cells, values):
            grid[p] = color
    else:
        word = _word(rng, palette, kind, max_period)
        phase = int(rng.integers(len(word)))
        for p in cells:
            grid[p] = word[(_index(box, p, model) + phase) % len(word)]


def _random_box(rng, height, width):
    t, b = sorted(int(v) for v in rng.choice(np.arange(1, height - 1), 2, replace=False))
    l, r = sorted(int(v) for v in rng.choice(np.arange(1, width - 1), 2, replace=False))
    return t, b, l, r


def _geometry(rng, plan, fallback=False):
    nested = plan["nested"] and plan["crossing"] >= 2
    if fallback:
        h = w = 30
        main = (8, 23, 8, 23)
    elif nested:
        h, w = (int(rng.integers(24, 31)) for _ in range(2))
        oh, ow = (int(rng.integers(8, d - 11)) for d in (h, w))
        t, l = (int(rng.integers(5, d - o - 6)) for d, o in ((h, oh), (w, ow)))
        main = (t, t + oh + 1, l, l + ow + 1)
    else:
        h, w = (int(rng.integers(17, 31)) for _ in range(2))
        oh, ow = (int(rng.integers(4, min(17, d - 6))) for d in (h, w))
        t, l = (int(rng.integers(2, d - o - 3)) for d, o in ((h, oh), (w, ow)))
        main = (t, t + oh + 1, l, l + ow + 1)
    t, b, l, r = main
    mask, inside = _area(main), _area((t + 1, b - 1, l + 1, r - 1))
    boxes, occupied = [], set()

    def accept(box, crossing):
        y0, y1, x0, x1 = box
        if y1 - y0 < 2 or x1 - x0 < 2:
            return False
        full = _perimeter(box)
        visible, hidden = full - mask, full & inside
        if not visible or _bbox(visible) != box or bool(hidden) != crossing:
            return False
        if full & _halo(occupied):
            return False
        if not crossing and full & mask:
            return False
        if not crossing and _size(box) >= _size(main):
            return False
        if crossing and len(hidden) < 2:
            return False
        boxes.append(box)
        occupied.update(full)
        return True

    if nested:
        if fallback:
            outer, inner = (1, 17, 1, 17), (3, 14, 3, 14)
        else:
            outer = (int(rng.integers(1, t - 3)), int(rng.integers(t + 5, b)),
                     int(rng.integers(1, l - 3)), int(rng.integers(l + 5, r)))
            y0, y1, x0, x1 = outer
            inner = (int(rng.integers(y0 + 2, t - 1)), int(rng.integers(t + 2, y1 - 1)),
                     int(rng.integers(x0 + 2, l - 1)), int(rng.integers(l + 2, x1 - 1)))
        assert accept(outer, True) and accept(inner, True)
    safe_cross = ([(1, 17, 1, 17), (3, 14, 3, 14), (20, 28, 20, 28), (2, 11, 20, 27)]
                  if nested else [(1, 13, 1, 13), (18, 28, 18, 28), (2, 11, 18, 27), (19, 27, 2, 11)])
    for index in range(len(boxes), plan["crossing"]):
        if fallback:
            assert accept(safe_cross[index], True)
            continue
        for _ in range(1200):
            if accept(_random_box(rng, h, w), True):
                break
        else:
            return None
    for _ in range(plan["exterior"]):
        if fallback:
            choices = [(yy, yy + 2, xx, xx + 2) for yy in range(1, h - 3) for xx in range(1, w - 3)]
            if not any(accept(box, False) for box in choices):
                return None
            continue
        for attempt in range(900):
            if attempt > 300:
                hh, ww = int(rng.integers(3, 6)), int(rng.integers(3, 7))
                yy, xx = int(rng.integers(1, h - hh)), int(rng.integers(1, w - ww))
                box = (yy, yy + hh - 1, xx, xx + ww - 1)
            else:
                box = _random_box(rng, h, w)
            if accept(box, False):
                break
        else:
            return None
    extras = []
    blocked = occupied | mask
    for _ in range(plan["extras"]):
        # Enumerate all small free rectangles after randomized larger attempts.
        choices = []
        for _ in range(0 if fallback else 90):
            hh, ww = int(rng.integers(3, 6)), int(rng.integers(3, 7))
            yy, xx = int(rng.integers(1, h - hh)), int(rng.integers(1, w - ww))
            choices.append((yy, yy + hh - 1, xx, xx + ww - 1))
        choices += [(yy, yy + 2, xx, xx + 2) for yy in range(1, h - 3) for xx in range(1, w - 3)]
        for box in choices:
            cells = _area(box)
            if _size(box) < _size(main) and not (cells & _halo(blocked)):
                extras.append(box)
                blocked.update(cells)
                break
        else:
            return None
    return h, w, main, boxes, extras


def _fallback_boundary(grid, box, palette, kind):
    """A visibly witnessed, deterministic completion for the bounded fallback."""
    colors = list(palette)
    if kind == "random":
        # This irregular eight-cell word is used only on fully visible borders.
        a, b, c = colors[0], colors[1], colors[-1]
        irregular = (a, a, b, a, c, b, b, c)
        for i, p in enumerate(sorted(_perimeter(box))):
            grid[p] = irregular[(i * i + i // 3) % len(irregular)]
    else:
        period = 1 if kind == "solid" else 2 if kind == "alternating" else min(3, len(colors))
        for y, x in _perimeter(box):
            grid[y, x] = colors[(y + x) % period]


def _pair_plan(rng, index, split, parameters, num_train, num_test):
    level = parameters["level"]
    minimal, maximal = parameters["mode"] == "minimal", parameters["mode"] == "maximal"
    first, second = split == "source_train" and index == 0, split == "source_train" and index == 1
    zero = (not minimal and ((num_train > 2 and split == "source_train" and index == num_train - 1)
                            or (num_train == 2 and split == "target_test" and index == num_test - 1))
            and (maximal or parameters["empty_case"]))
    crossing = 2 if split == "source_train" else 3
    if not minimal and not first:
        crossing = int(rng.integers(1 if split == "source_train" else 3, 4 if split == "source_train" else 5))
    if second and maximal:
        crossing = 2
    if zero:
        crossing = 0
    exterior = 0 if minimal or first else int(rng.integers(1, 3))
    if second and not minimal and crossing + exterior == 2:
        exterior = 2
    extras = 0 if first else (1 if second else int(rng.integers(1, 3)))
    nested = not minimal and not first and crossing >= 2 and (maximal or rng.random() < level * 0.65)
    distinct_fill = second or (not first and (split == "target_test" or rng.random() < 0.6))
    if zero:
        distinct_fill = bool(rng.integers(2))
    top_kind = "solid"
    if distinct_fill and not minimal:
        top_kind = "random" if maximal or rng.random() < level else "complex"
    return {"crossing": crossing, "exterior": exterior, "extras": extras,
            "nested": nested, "distinct_fill": distinct_fill, "top_kind": top_kind,
            "demo": 0 if first else 1 if second else 2, "minimal": minimal,
            "maximal": maximal, "level": level}


def _orient_geometry(geometry, turns, flip):
    h, w, main, boxes, extras = geometry

    def orient(box):
        result = []
        for y, x in ((box[0], box[2]), (box[1], box[3])):
            hh, ww = h, w
            for _ in range(turns):
                y, x = ww - 1 - x, y
                hh, ww = ww, hh
            if flip:
                x = ww - 1 - x
            result.append((y, x))
        return _bbox(result)

    main, boxes, extras = orient(main), [orient(b) for b in boxes], [orient(b) for b in extras]
    return (w, h, main, boxes, extras) if turns % 2 else (h, w, main, boxes, extras)


def _make_pair(rng, bg, previous_frame, plan, parameters):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != bg])]
    top_primary = next(c for c in palette if c != previous_frame)
    palette.remove(top_primary)
    fill_color = palette.pop()
    top_count = 1 if plan["top_kind"] == "solid" else int(rng.integers(2, 4))
    top_palette = [top_primary] + [palette.pop() for _ in range(top_count - 1)]
    middle_palette = palette
    fill = fill_color if plan["distinct_fill"] else bg
    model, max_period = parameters["model"], parameters["max_period"]
    for attempt in range(29):
        fallback = attempt == 28
        geometry = _geometry(rng, plan, fallback=fallback)
        if geometry is None:
            continue
        if not fallback:
            geometry = _orient_geometry(geometry, int(rng.integers(4)), bool(rng.integers(2)))
        h, w, main, boxes, extras = geometry
        under = np.full((h, w), bg, dtype=np.int8)
        for index, box in enumerate(boxes):
            crossing = index < plan["crossing"]
            if not crossing:
                kind = "random" if plan["maximal"] or rng.random() < plan["level"] else "solid"
            elif plan["demo"] == 0:
                kind = "solid"
            elif plan["minimal"]:
                kind = "alternating" if plan["demo"] == 1 or index == 2 else "solid"
            elif index == 0 and plan["demo"] == 1:
                kind = "complex"
            else:
                kind = str(rng.choice(["solid", "alternating", "complex"], p=[0.25, 0.2, 0.55]))
            if fallback:
                _fallback_boundary(under, box, middle_palette, kind)
            else:
                _paint_boundary(under, box, rng, middle_palette, kind, model, max_period)
        t, b, l, r = main
        # Build truth from the complete middle layer, then apply the opaque mask.
        output = under[t + 1:b, l + 1:r].copy()
        output[output == bg] = fill
        grid = under.copy()
        for box in [main] + extras:
            y0, y1, x0, x1 = box
            grid[y0 + 1:y1, x0 + 1:x1] = fill if box == main else fill_color
            if fallback:
                _fallback_boundary(grid, box, top_palette, plan["top_kind"])
            else:
                _paint_boundary(grid, box, rng, top_palette, plan["top_kind"], model, max_period)
        try:
            predicted, scene = _read_input(grid)
            assert np.array_equal(output, predicted)
            assert len(scene["outlines"]) == len(boxes)
            if plan["demo"] == 1 and not plan["minimal"]:
                assert any(o["hidden"] and o["fits"][0][0] >= 3 for o in scene["outlines"])
            if plan["maximal"] and plan["demo"] == 1:
                assert any(o["hidden"] and len(o["colors"]) >= 3 for o in scene["outlines"])
        except AssertionError:
            continue
        return {"input": grid.copy(), "output": output.copy()}, top_primary
    raise AssertionError("could not produce a uniquely readable scene")


def _array_key(array):
    return tuple(map(int, array.shape)), tuple(map(int, array.flat))


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def _pair_digest(pair):
    return hashlib.sha256(repr(_pair_key(pair)).encode()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer_argument(seed, "seed") % (1 << 64)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    residue = seed % 14
    mode = "minimal" if residue == 0 else "maximal" if residue in (1, 2) else "intermediate"
    parameters = {"mode": mode, "level": 0.0 if mode == "minimal" else 1.0 if mode == "maximal" else float(rng.uniform(0.25, 0.85)),
                  "model": str(rng.choice(["diagonal", "walk"])),
                  "max_period": 2 if mode == "minimal" else int(rng.integers(4, 7)),
                  "empty_case": bool(rng.random() < 0.55)}
    backgrounds = [int(c) for c in rng.permutation(10)]
    puzzle, seen, previous_frame = {"train": [], "test": []}, set(), None
    for split, count, regime in (("train", num_train, "source_train"), ("test", num_test, "target_test")):
        for index in range(count):
            pair_number = len(seen)
            plan = _pair_plan(rng, index, regime, parameters, num_train, num_test)
            for _ in range(12):
                pair, frame = _make_pair(rng, backgrounds[pair_number % 10], previous_frame, plan, parameters)
                key = _array_key(pair["input"])
                if key not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    previous_frame = frame
                    break
            else:
                raise AssertionError("duplicate scene after bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    _, scene = _read_input(grid)
    outlines, main = scene["outlines"], scene["main"]
    t, b, l, r = main["box"]
    nested, nonconcentric, spanning = 0, 0, 0
    for outer in outlines:
        a, z, c, d = outer["box"]
        if outer["hidden"] and ((a < t and z > b) or (c < l and d > r)):
            spanning += 1
        for inner in outlines:
            e, f, g, h = inner["box"]
            if a < e < f < z and c < g < h < d:
                nested += 1
                nonconcentric += int((a + z, c + d) != (e + f, g + h))
    periods = [o["fits"][0][0] for o in outlines if o["hidden"]]
    random_count = sum(not o["fits"] for o in outlines if not o["hidden"])
    top_pred, top_fits = _fits(grid, main["box"], _perimeter(main["box"]), set())
    return {"background": scene["background"], "fill_matches": main["fill"] == scene["background"],
            "middle_count": len(outlines), "crossing_count": sum(bool(o["hidden"]) for o in outlines),
            "nested": nested, "nonconcentric": nonconcentric, "spanning": spanning,
            "periods": tuple(sorted(periods)), "random_exterior": random_count,
            "middle_palette_sizes": tuple(sorted(len(o["colors"]) for o in outlines)),
            "frame_colors": main["colors"], "frame_pattern": "random" if not top_fits else "solid" if top_fits[0][0] == 1 else "regular",
            "opaque_count": len(scene["opaque"]), "main_box": main["box"],
            "layout": tuple(sorted(o["box"] for o in outlines))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = _array_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _pair_digest(pair) not in _SOURCE_DIGESTS
            prediction, scene = _read_input(pair["input"])
            assert np.array_equal(prediction, pair["output"]), "incorrect hidden-layer crop"
            assert pair["input"].shape != pair["output"].shape
            features[split].append(_features(pair["input"]))
    train = features["train"]
    assert train[0]["fill_matches"] and not train[1]["fill_matches"]
    assert train[0]["crossing_count"] == 2 and train[1]["crossing_count"] > 0
    assert any(p >= 2 for p in train[1]["periods"]), "no periodic training demonstration"
    assert train[1]["opaque_count"] >= 2, "largest-frame selection not demonstrated"
    assert len({f["background"] for f in train}) >= 2
    assert len({f["frame_colors"] for f in train}) >= 2
    assert len({f["middle_count"] for fs in features.values() for f in fs}) >= 2
    for f in features["test"]:
        assert f["crossing_count"] == 0 or 3 <= f["crossing_count"] <= 4
        assert f["opaque_count"] >= 2
        if any(p >= 3 for p in f["periods"]):
            assert any(p >= 3 for p in train[1]["periods"])
    # Only copied training pairs and the test input reach the prediction API.
    examples = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
    for pair in puzzle["test"]:
        assert np.array_equal(predict(examples, pair["input"].copy()), pair["output"])


def self_test() -> None:
    counts, puzzles, inputs, outputs, structures, measured = set(), set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        puzzles.add(_puzzle_key(puzzle))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair["input"])
                local.append(f)
                inputs.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                structures.add((f["main_box"], f["layout"]))
        measured.extend(local)
        if seed % 14 == 0:
            assert all(max(f["periods"], default=1) <= 2 and not f["random_exterior"] for f in local)
        elif seed % 14 in (1, 2):
            assert any(f["nested"] for f in local)
            assert any(f["random_exterior"] and f["frame_pattern"] == "random" for f in local)
            assert any(f["crossing_count"] == 0 for f in local)
            assert any(max(f["middle_palette_sizes"], default=0) >= 3 for f in local)
    assert len(puzzles) == 42 and len(inputs) >= 50 and len(outputs) >= 50 and len(structures) >= 200
    assert len({a for a, b in counts}) >= 4 and len({b for a, b in counts}) >= 3
    assert {f["background"] for f in measured} == set(range(10))
    assert len({f["middle_count"] for f in measured}) >= 5
    assert {f["frame_pattern"] for f in measured} == {"solid", "regular", "random"}
    assert {f["fill_matches"] for f in measured} == {False, True}
    assert any(f["nonconcentric"] for f in measured) and any(f["spanning"] for f in measured)
    assert any(f["crossing_count"] == 0 and f["fill_matches"] for f in measured)
    assert {p for f in measured for p in f["periods"]} >= {1, 2, 3, 4, 5, 6}
    cases = [(2, 1), (10, 1), (10, 4), (6, 3), (None, 1), (None, 4), (2, None), (10, None)]
    for index, (a, b) in enumerate(cases):
        one, two = generate_puzzle(1000 + index, a, b), generate_puzzle(1000 + index, a, b)
        assert _puzzle_key(one) == _puzzle_key(two)
        assert a is None or len(one["train"]) == a
        assert b is None or len(one["test"]) == b
    assert _puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123))
    for a, b in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.0, 1)]:
        try:
            generate_puzzle(999, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # A changed hidden pixel must be rejected, not blessed by cached metadata.
    bad = generate_puzzle(211, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
