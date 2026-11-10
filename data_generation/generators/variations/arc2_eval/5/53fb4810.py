"""Hub-attached whole tiles continue to the boundary in hub-relative order.

Completed material is authoritative, except where an unfinished strip extends
across it.  All geometry and color roles used by the validator come from input
pixels; construction records are never passed to the solver.
"""

import itertools
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check_count_value(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return int(value)


def _choose_counts(seed, num_train, num_test):
    a = _check_count_value("num_train", num_train, 2, 10)
    b = _check_count_value("num_test", num_test, 1, 4)
    if a is None and b is None:
        b = 1 + seed % 4
        a = b + 1 + (seed // 4) % (10 - b)
    elif a is None:
        a = b + 1 + seed % (10 - b)
    elif b is None:
        b = 1 + seed % min(4, a - 1)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _array_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def _unique_pair(pair, used):
    source_keys = {_pair_key(p) for p in _source_pairs()}
    if _array_key(pair["input"]) not in used and _pair_key(pair) not in source_keys:
        return pair
    # A rare duplicate or exact source copy is repaired without unbounded RNG
    # retries. At least three visible colors give 720 injective recolorings;
    # fewer than 18 input/source exclusions can exist in one legal puzzle.
    colors = [int(c) for c in np.unique(pair["input"])]
    for values in itertools.permutations(range(10), len(colors)):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[v] for k, v in pair.items()}
        if _array_key(candidate["input"]) not in used and _pair_key(candidate) not in source_keys:
            return candidate
    raise AssertionError("finite recoloring space exhausted")


def _minimal_period(slices):
    """Only whole copies: ABA is a three-slice cyclic word, not period two."""
    word = [tuple(int(x) for x in row) for row in slices]
    assert word
    return next(p for p in range(1, len(word) + 1)
                if len(word) % p == 0
                and all(row == word[i % p] for i, row in enumerate(word)))


def _hub_mask(height, width, kind, thickness=None):
    mask = np.ones((height, width), dtype=bool)
    if kind == "clipped":
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
    elif kind == "cross":
        th, tw = thickness
        mask[:] = False
        mask[(height-th)//2:(height+th)//2, :] = True
        mask[:, (width-tw)//2:(width+tw)//2] = True
    return mask


def _shape_kind(mask):
    h, w = mask.shape
    if not (3 <= h <= 8 and 3 <= w <= 8):
        return None
    if np.all(mask):
        return "rectangle"
    if np.array_equal(mask, _hub_mask(h, w, "clipped")):
        return "cross" if h == w == 3 else "clipped"
    # Independent centered arm thicknesses, with at least one-cell shoulders.
    for th in range(1, h-1):
        for tw in range(1, w-1):
            if (h-th) % 2 == (w-tw) % 2 == 0:
                if np.array_equal(mask, _hub_mask(h, w, "cross", (th, tw))):
                    return "cross"
    return None


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                point = (r+dr, c+dc)
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return tuple(int(v) for v in (min(rows), min(cols), max(rows)-min(rows)+1, max(cols)-min(cols)+1))


def _maximum(shape, hub, direction):
    t, l, h, w = hub
    return (t, shape[0]-t-h, l, shape[1]-l-w)[direction]


def _coordinates(hub, direction, width, distance):
    t, l, h, w = hub
    if direction < 2:
        r = t-distance if direction == 0 else t+h-1+distance
        return [(r, l+(w-width)//2+i) for i in range(width)]
    c = l-distance if direction == 2 else l+w-1+distance
    return [(t+(h-width)//2+i, c) for i in range(width)]


def _dilate(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _parse_roles(grid, background, hub_colors, color_components):
    hubs, claimed = [], set()
    for color in hub_colors:
        for cells in color_components[color]:
            box = _box(cells)
            t, l, h, w = box
            mask = grid[t:t+h, l:l+w] == color
            kind = _shape_kind(mask)
            assert kind is not None
            hubs.append((box, color, cells, kind))
            claimed.update(cells)
    assert 1 <= len(hubs) <= 6
    strips = []
    for box, color, cells, kind in hubs:
        choices = []
        t, l, h, w = box
        for direction in range(4):
            maximum = _maximum(grid.shape, box, direction)
            if maximum < 1:
                continue
            transverse = w if direction < 2 else h
            # The entire adjacent face determines width, with centered shoulders.
            face = _coordinates(box, direction, transverse, 1)
            payload = [i for i, p in enumerate(face)
                       if int(grid[p]) not in (background, *hub_colors)]
            if not payload:
                continue
            width = len(payload)
            offset = (transverse-width)//2
            if payload != list(range(offset, offset+width)) or (transverse-width) % 2:
                continue
            dr, dc = _DIRECTIONS[direction]
            if not all((r-dr, c-dc) in cells for r, c in _coordinates(box, direction, width, 1)):
                continue
            if width > transverse-2:
                continue
            choices.append((direction, width, maximum))
        assert len(choices) == 1, "hub must have exactly one centered attached strip"
        direction, width, maximum = choices[0]
        slices, coords = [], []
        for d in range(1, maximum+1):
            points = _coordinates(box, direction, width, d)
            values = tuple(int(grid[p]) for p in points)
            if all(v not in (background, *hub_colors) for v in values):
                slices.append(values)
                coords.extend(points)
            else:
                assert all(v == background for v in values), "partial slice or hub collision"
                break
        assert slices
        assert not claimed.intersection(coords), "original objects overlap"
        claimed.update(coords)
        strips.append(dict(hub=box, hub_color=color, kind=kind, hub_cells=cells,
                           direction=direction, width=width, maximum=maximum,
                           slices=slices, cells=set(coords), complete=len(slices) == maximum,
                           period=_minimal_period(slices)))
    assert claimed == set(zip(*np.nonzero(grid != background))), "unattached material"
    return strips


def _infer_input(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(np.argmax(counts))
    assert sum(counts == counts[background]) == 1, "background must be visible and unique"
    components = {int(c): _components(grid == c) for c in np.unique(grid) if c != background}
    candidates = []
    for color, groups in components.items():
        good = True
        for cells in groups:
            t, l, h, w = _box(cells)
            if _shape_kind(grid[t:t+h, l:l+w] == color) is None:
                good = False
                break
        if good:
            candidates.append(color)
    interpretations = []
    for n in range(1, min(3, len(candidates))+1):
        for colors in itertools.combinations(candidates, n):
            try:
                strips = _parse_roles(grid, background, colors, components)
                interpretations.append((background, colors, strips))
            except AssertionError:
                continue
    assert len(interpretations) == 1, "ambiguous or missing geometric color roles"
    return interpretations[0]


def _predict_input(grid):
    background, colors, strips = _infer_input(grid)
    protected = set().union(*(s["hub_cells"] | (s["cells"] if not s["complete"] else set())
                              for s in strips))
    writes, crossing_cells = {}, set()
    completed_cells = set().union(*(s["cells"] for s in strips if s["complete"]))
    for s in strips:
        if s["complete"]:
            continue
        for d in range(len(s["slices"])+1, s["maximum"]+1):
            values = s["slices"][(d-1) % len(s["slices"])]
            for p, value in zip(_coordinates(s["hub"], s["direction"], s["width"], d), values):
                assert p not in protected, "continuation hits a hub or unfinished tile"
                assert p not in writes, "new strips intersect without a taught priority"
                if int(grid[p]) != background:
                    assert p in completed_cells
                    crossing_cells.add(p)
                writes[p] = value
    expected = grid.copy()
    for p, value in writes.items():
        expected[p] = value
    return expected, dict(background=background, hub_colors=colors, strips=strips,
                          crossing_cells=crossing_cells,
                          overwrite_count=sum(int(grid[p]) != writes[p] for p in crossing_cells),
                          changed_count=int(np.count_nonzero(expected != grid)))


def _analyze_pair(pair):
    expected, analysis = _predict_input(pair["input"])
    assert np.array_equal(expected, pair["output"]), "wrong continuation or changed authoritative material"
    return analysis


def _tile(rng, width, period, palette, style=None):
    palette = list(palette)
    a = int(rng.choice(palette))
    b = int(rng.choice([c for c in palette if c != a])) if len(palette) > 1 else a
    if style == "whole_word":
        first = rng.choice(palette, size=width).astype(np.int8)
        second = first.copy()
        second[int(rng.integers(width))] = b if first[0] == a else a
        if np.array_equal(first, second):
            second[0] = next(c for c in palette if c != int(first[0]))
        return np.array([first, second, first], dtype=np.int8)
    mode = int(rng.integers(5))
    tile = rng.choice(palette, size=(period, width)).astype(np.int8)
    if mode == 0:
        tile[:] = a
    elif mode == 1:
        tile[:] = rng.choice(palette, size=(period, 1))
    elif mode == 2:
        tile[:] = rng.choice(palette, size=(1, width))
    elif mode == 3 and period >= 3:
        tile[-1] = tile[0]
    if style == "primitive" and period > 1:
        # One exceptional slice makes every requested length a complete primitive word.
        if _minimal_period(tile) != period:
            tile[:] = a
            tile[-1, -1] = b
    if style == "transverse" and width > 1:
        tile[0, 0], tile[0, -1] = a, b
    return tile


def _make_spec(rng, shape, palette, hub_colors, direction, complete,
               width_limit, period_limit, primary=None, previous=(), irregular=False):
    width = int(primary[0]) if primary else int(rng.integers(1, width_limit+1))
    period = int(primary[1]) if primary else int(rng.integers(1, period_limit+1))
    if irregular:
        period = 2
    transverse = width + 2*int(rng.integers(1, 3))
    longitudinal = int(rng.integers(3, 8))
    h, w = (longitudinal, transverse) if direction < 2 else (transverse, longitudinal)
    if h > shape[0] or w > shape[1]:
        return None
    t = int(rng.integers(0, shape[0]-h+1))
    l = int(rng.integers(0, shape[1]-w+1))
    box = (t, l, h, w)
    maximum = _maximum(shape, box, direction)
    minimum = period + (period if primary and primary[2] == "whole_word" else 1)
    if maximum < (6 if irregular else 1 if complete else minimum):
        return None
    kind = str(rng.choice(("rectangle", "clipped", "cross")))
    th_options = [v for v in range(1, h-1) if (h-v) % 2 == 0]
    tw_options = [v for v in range(1, w-1) if (w-v) % 2 == 0]
    if direction < 2:
        tw_options = [v for v in tw_options if v >= width]
    else:
        th_options = [v for v in th_options if v >= width]
    mask = _hub_mask(h, w, kind, (int(rng.choice(th_options)), int(rng.choice(tw_options))))
    hub_cells = {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}
    hub_box = {(r, c) for r in range(t, t+h) for c in range(l, l+w)}
    local_palette = list(rng.choice(palette, size=int(rng.integers(1, len(palette)+1)), replace=False))
    if primary or irregular:
        local_palette = list(palette)
    tile = _tile(rng, width, period, local_palette,
                 "primitive" if irregular else primary[2] if primary else None)
    if previous and not primary and rng.random() < .22:
        compatible = [s for s in previous if s["tile"].shape == tile.shape]
        if compatible:
            tile = compatible[int(rng.integers(len(compatible)))]["tile"].copy()
    if complete:
        length = maximum
        slices = tile[np.arange(length) % len(tile)].copy()
        appearance = "periodic" if len(np.unique(slices)) > 1 else "monochrome"
        if irregular or (len(palette) >= 2 and length >= 4 and rng.random() < .45):
            # Perturb arbitrary positions, sometimes a distal block, sometimes the interior.
            start = 4 if irregular else 0
            count = int(rng.integers(1, min(length-start, max(1, length//2))+1))
            positions = (np.arange(length-count, length) if rng.random() < .5
                         else rng.choice(np.arange(start, length), size=count, replace=False))
            for pos in positions:
                j = int(rng.integers(width))
                slices[pos, j] = rng.choice([c for c in palette if c != int(slices[pos, j])])
            appearance = "irregular"
    else:
        max_copies = (maximum-1)//period
        copies = int(rng.integers(1, max_copies+1))
        if primary and primary[2] == "whole_word":
            copies = 1
        elif rng.random() < .35:
            copies = 1
        length = period*copies
        slices = tile[np.arange(length) % len(tile)].copy()
        appearance = "tile"
    original = {p for d in range(1, length+1) for p in _coordinates(box, direction, width, d)}
    extension = {p for d in range(length+1, maximum+1) for p in _coordinates(box, direction, width, d)}
    return dict(hub=box, mask=mask, hub_cells=hub_cells, hub_box=hub_box,
                color=int(rng.choice(hub_colors)), direction=direction, width=width,
                complete=complete, maximum=maximum, slices=slices, tile=tile,
                original=original, extension=extension, appearance=appearance)


def _compatible(candidate, specs, allow_crossing):
    if candidate is None:
        return False
    a = candidate
    for b in specs:
        # Empty collars keep attached faces and visible tile endings unambiguous.
        if _dilate(a["hub_cells"] | a["original"]) & (b["hub_cells"] | b["original"]):
            return False
        if (_dilate(a["hub_box"]) & (b["original"] | b["extension"]) or
                _dilate(b["hub_box"]) & (a["original"] | a["extension"])):
            return False
        if a["extension"] & b["extension"]:
            return False
        for x, y in ((a, b), (b, a)):
            if x["extension"] & y["original"]:
                if not allow_crossing or not y["complete"] or x["direction"]//2 == y["direction"]//2:
                    return False
    return True


def _paint(shape, specs, background):
    grid = np.full(shape, background, dtype=np.int8)
    for s in specs:
        for p in s["hub_cells"]:
            grid[p] = s["color"]
        for d, values in enumerate(s["slices"], 1):
            for p, v in zip(_coordinates(s["hub"], s["direction"], s["width"], d), values):
                grid[p] = v
    output = grid.copy()
    # Construction uses the sampled full tile, while the checker repeats visible pixels.
    for s in specs:
        if not s["complete"]:
            for d in range(len(s["slices"])+1, s["maximum"]+1):
                values = s["tile"][(d-1) % len(s["tile"])]
                for p, v in zip(_coordinates(s["hub"], s["direction"], s["width"], d), values):
                    output[p] = v
    return {"input": grid, "output": output}


def _fallback(rng, palette, hub_colors, background, primary, crossing, witness, count):
    """Guaranteed spacious construction used only after bounded packing attempts."""
    shape = (30, 30)
    width, period, style = primary or (1, 2, "primitive")
    placements = []
    if count > 1 or witness or crossing:
        placements.append(((5, 1, 3, 3), 3, 1, True))
    placements.append(((22, 21, 3, width+2), 0, width, False))
    if count > 2:
        placements.append(((20, 10, 5, 3), 0, 1, False))
    specs = []
    for box, direction, sw, complete in placements:
        t, l, h, w = box
        mask = _hub_mask(h, w, "clipped")
        maximum = _maximum(shape, box, direction)
        tile = _tile(rng, sw, period if not complete else 2, palette,
                     style if not complete and sw == width else "primitive")
        slices = tile[np.arange(maximum if complete else len(tile)) % len(tile)].copy()
        if complete and witness:
            slices[-1, 0] = next(c for c in palette if c != int(slices[-1, 0]))
        specs.append(dict(hub=box, mask=mask, direction=direction, width=sw,
                          maximum=maximum, tile=tile, slices=slices, complete=complete,
                          color=int(hub_colors[0]),
                          hub_cells={(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}))
    # If no crossing is requested, rotate the completed strip away from other rays.
    if specs[0]["complete"] and not crossing:
        s = specs[0]
        s["direction"] = 0
        s["maximum"] = 5
        s["slices"] = s["slices"][-5:]
    elif crossing:
        # Coincident payload colors must not hide the fallback's overwrite
        # demonstration. Change a crossed original cell, beyond the early cycle.
        completed = specs[0]
        original = {p: (d-1, j) for d in range(1, completed["maximum"]+1)
                    for j, p in enumerate(_coordinates(completed["hub"], completed["direction"],
                                                        completed["width"], d))}
        taught = False
        for s in specs[1:]:
            for d in range(len(s["slices"])+1, s["maximum"]+1):
                for p, value in zip(_coordinates(s["hub"], s["direction"], s["width"], d),
                                    s["tile"][(d-1) % len(s["tile"])]):
                    if p in original and not taught:
                        i, j = original[p]
                        completed["slices"][i, j] = next(c for c in palette if c != int(value))
                        taught = True
        assert taught
    pair = _paint(shape, specs, background)
    turns = int(rng.integers(4))
    return {k: np.ascontiguousarray(np.rot90(v, turns)) for k, v in pair.items()}


def _generate_pair(rng, config, primary=None, witness=False, require_crossing=False,
                   count=None, background=None, allow_crossing=False, allow_irregular=True):
    colors = [int(c) for c in rng.permutation(10)]
    if background is not None:
        colors.remove(background)
        colors.insert(0, background)
    background = colors[0]
    nh = int(rng.choice([1, 1, 1, 2, 3]))
    hub_colors = colors[1:1+nh]
    palette_count = int(rng.integers(2 if witness or primary else 1, 10-nh))
    palette = colors[1+nh:1+nh+palette_count]
    level, width_limit, period_limit = config
    count = int(count if count is not None else rng.choice(
        [1, 1, 2, 2, 3] if level == 0 else [1, 2, 2, 3, 3, 4, 5, 6]))
    if witness or require_crossing:
        count = max(2, count)
    all_complete = not primary and not witness and rng.random() < .075
    completed = [all_complete or bool(rng.random() < .35) for _ in range(count)]
    completed[0] = bool(witness or require_crossing) if not all_complete else True
    if not all_complete:
        completed[-1] = False
    completed.sort(reverse=True)
    first_unfinished = next((i for i, c in enumerate(completed) if not c), None)
    direction_mode = int(rng.integers(3))  # same, parallel/opposing, freely mixed
    base_direction = int(rng.integers(4))
    for scene_attempt in range(14):
        if scene_attempt < 7:
            low = 8 if count == 1 else 12 if count <= 3 else 20
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
            if count == 1 and rng.random() < .35:
                shape = (int(rng.integers(7, 12)), int(rng.integers(16, 31)))
        else:
            shape = (30, 30)
        specs = []
        for i, complete in enumerate(completed):
            for attempt in range(150):
                direction = (base_direction if direction_mode == 0 else
                             (base_direction//2)*2+int(rng.integers(2)) if direction_mode == 1 else
                             int(rng.integers(4)))
                if require_crossing and i == first_unfinished:
                    direction = (1-specs[0]["direction"]//2)*2+int(rng.integers(2))
                candidate = _make_spec(rng, shape, palette, hub_colors, direction, complete,
                                       width_limit, period_limit,
                                       primary if i == first_unfinished else None, specs,
                                       irregular=witness and i == 0)
                if not _compatible(candidate, specs, allow_crossing or require_crossing):
                    continue
                if not allow_irregular and candidate["complete"]:
                    candidate["slices"] = candidate["tile"][np.arange(candidate["maximum"]) % len(candidate["tile"])]
                if require_crossing and i == first_unfinished:
                    if not any(candidate["extension"] & s["original"] for s in specs if s["complete"]):
                        continue
                specs.append(candidate)
                break
            else:
                break
        if len(specs) != count:
            continue
        pair = _paint(shape, specs, background)
        try:
            analysis = _analyze_pair(pair)
            if require_crossing and analysis["overwrite_count"] == 0:
                continue
            if witness and not _irregular_preserved(analysis, pair["input"], pair["output"]):
                continue
            return pair
        except AssertionError:
            continue
    return _fallback(rng, palette, hub_colors, background, primary,
                     require_crossing, witness, count)


def _has_whole_word_witness(analysis):
    for s in analysis["strips"]:
        word = s["slices"]
        if s["complete"] or s["maximum"]-len(word) < len(word):
            continue
        # A short nondividing prefix period predicts something different.
        for p in range(1, len(word)):
            if len(word) % p and all(row == word[i % p] for i, row in enumerate(word)):
                return True
    return False


def _completed_kind(s):
    word = s["slices"]
    if len({v for row in word for v in row}) == 1:
        return "monochrome"
    # A completed strip can end in a clipped periodic tile; this is descriptive,
    # never a license to overwrite its visible exceptions.
    for p in range(1, min(6, len(word)//2+1)):
        if all(row == word[i % p] for i, row in enumerate(word)):
            return "periodic"
    return "irregular"


def _irregular_preserved(analysis, inp, out):
    for s in analysis["strips"]:
        if not s["complete"]:
            continue
        word = s["slices"]
        # Preserve a visible exception to an established early cycle. Merely
        # having a nonperiodic strip does not distinguish preservation from an
        # alternative that regularizes strips only after seeing two copies.
        for period in range(1, min(5, len(word)//2)+1):
            if not all(word[i] == word[i % period] for i in range(2*period)):
                continue
            for i in range(2*period, len(word)):
                for j, p in enumerate(_coordinates(s["hub"], s["direction"], s["width"], i+1)):
                    if (word[i][j] != word[i % period][j] and p not in analysis["crossing_cells"]
                            and inp[p] == out[p]):
                        return True
    return False


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    num_train, num_test = _choose_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # These are curriculum/distribution choices. The geometric transformation,
    # whole-tile convention, and crossing semantics are fixed across all pairs.
    regime = str(rng.choice(("compositional", "familiar", "wider", "longer", "layout")))
    level = int(rng.integers(3))
    crosses = bool(rng.random() < .40)
    width_limit = 2 if level == 0 else 3 if level == 1 else 4
    period_limit = 2 if regime == "compositional" else 3 if regime == "longer" else 5
    if regime == "compositional":
        width_limit = 2
    if regime == "wider":
        width_limit = min(width_limit, 2)
    config = (level, width_limit, period_limit)
    background_order = [int(c) for c in rng.permutation(10)]
    puzzle, used = {"train": [], "test": []}, set()
    anchor_count = int(rng.integers(2, 5))
    for i in range(num_train):
        if regime == "compositional":
            primary = (1, 2, "primitive") if i == 0 else (2, 1, "transverse")
        elif i == 0:
            primary = (int(rng.integers(1, width_limit+1)), 3, "whole_word")
        elif i == 1:
            primary = (min(2, width_limit), 2, "primitive")
        else:
            primary = None
        local_config = config
        if regime == "compositional":
            local_config = (level, primary[0], primary[1])
        count = anchor_count if i == 0 else 1 if i == 1 else None
        pair = _generate_pair(rng, local_config, primary, witness=i == 0,
                              require_crossing=crosses and i == 0, count=count,
                              background=background_order[i % 10], allow_crossing=crosses)
        pair = _unique_pair(pair, used)
        used.add(_array_key(pair["input"]))
        puzzle["train"].append(pair)
    # Presentation order has no association with width, period, or crossing.
    rng.shuffle(puzzle["train"])
    train_analyses = [_analyze_pair(p) for p in puzzle["train"]]
    taught_crossing = any(a["overwrite_count"] for a in train_analyses)
    taught_irregular = any(_irregular_preserved(a, p["input"], p["output"])
                           for p, a in zip(puzzle["train"], train_analyses))
    for i in range(num_test):
        primary = None
        test_config = config
        if regime == "compositional":
            primary = (2, 2, "primitive")
        elif regime == "wider":
            primary = (int(rng.integers(3, 5)), int(rng.integers(1, 6)), "primitive")
            test_config = (level, 4, 5)
        elif regime == "longer":
            primary = (int(rng.integers(1, width_limit+1)), int(rng.integers(4, 6)), "primitive")
            test_config = (level, width_limit, 5)
        pair = _generate_pair(rng, test_config, primary,
                              count=int(rng.integers(3, 7)) if regime == "layout" else None,
                              background=background_order[(num_train+i) % 10],
                              allow_crossing=taught_crossing, allow_irregular=taught_irregular)
        # A long apparent irregular completed word may occur by chance. Its
        # preservation still needs a training witness; otherwise use periodic material.
        if not taught_irregular:
            a = _analyze_pair(pair)
            if any(s["complete"] and _completed_kind(s) == "irregular" for s in a["strips"]):
                pair = _generate_pair(rng, test_config, primary or (1, 2, "primitive"),
                                      count=1, background=background_order[(num_train+i) % 10])
        pair = _unique_pair(pair, used)
        used.add(_array_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    used_inputs, used_pairs, analyses = set(), set(), {"train": [], "test": []}
    source_keys = {_pair_key(p) for p in _source_pairs()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _pair_key(pair)
            assert key[0] not in used_inputs and key not in used_pairs and key not in source_keys
            used_inputs.add(key[0])
            used_pairs.add(key)
            analyses[split].append(_analyze_pair(pair))
    training, testing = analyses["train"], analyses["test"]
    unfinished = [s for x in training for s in x["strips"] if not s["complete"]]
    assert any(x["changed_count"] > 0 for x in training)
    assert any(s["period"] >= 2 for s in unfinished), "show cyclic phase in training"
    assert any(s["width"] >= 2 for s in unfinished), "show vector slices in training"
    if any(x["crossing_cells"] for x in testing):
        assert any(x["overwrite_count"] for x in training), "crossing needs a changed-color witness"
    if any(s["period"] > 2 for x in testing for s in x["strips"] if not s["complete"]):
        assert any(_has_whole_word_witness(x) for x in training), "teach complete cyclic words"
    if any(s["complete"] and _completed_kind(s) == "irregular" for x in testing for s in x["strips"]):
        assert any(_irregular_preserved(x, p["input"], p["output"])
                   for x, p in zip(training, puzzle["train"])), "teach preservation of distal exceptions"
    # Pair-level roles and object counts must not accidentally become hidden constants.
    all_analyses = training + testing
    assert len({x["background"] for x in all_analyses}) > 1
    assert len({len(x["strips"]) for x in all_analyses}) > 1


def review_features(pair):
    """Measurements reconstructed from actual pixels, also used by self_test."""
    a = _analyze_pair(pair)
    strips = a["strips"]
    unfinished = [s for s in strips if not s["complete"]]
    dirs = [s["direction"] for s in strips]
    palettes = [{v for row in s["slices"] for v in row} for s in strips]
    palette_relations = set()
    identical_tiles = False
    for i, j in itertools.combinations(range(len(strips)), 2):
        x, y = palettes[i], palettes[j]
        palette_relations.add("same" if x == y else "overlap" if x & y else "disjoint")
        if (strips[i]["width"] == strips[j]["width"] and
                strips[i]["slices"][:strips[i]["period"]] == strips[j]["slices"][:strips[j]["period"]]):
            identical_tiles = True
    mixed_completion_overlap = any(palettes[i] & palettes[j]
                                   for i, j in itertools.combinations(range(len(strips)), 2)
                                   if strips[i]["complete"] != strips[j]["complete"])
    tiles = [s["slices"][:s["period"]] for s in unfinished]
    # Hub central row/column thickness and size are measured separately from
    # the attached cross-section, including differently sized hubs in one pair.
    geometry = []
    for s in strips:
        t, l, h, w = s["hub"]
        cells = s["hub_cells"]
        edge_h = sum((r, l) in cells for r in range(t, t+h))
        edge_w = sum((t, c) in cells for c in range(l, l+w))
        geometry.append((h, w, edge_h, edge_w, s["width"]))
    distances = [abs(x["hub"][0]-y["hub"][0])+abs(x["hub"][1]-y["hub"][1])
                 for x, y in itertools.combinations(strips, 2)]
    return dict(shape=list(pair["input"].shape), hubs=len(strips),
                hub_shapes=sorted({s["kind"] for s in strips}),
                hub_sizes=sorted({tuple(s["hub"][2:]) for s in strips}),
                widths=sorted({s["width"] for s in strips}),
                periods=sorted({s["period"] for s in unfinished}),
                lengths=sorted({len(s["slices"]) for s in unfinished}),
                extensions=sorted({s["maximum"]-len(s["slices"]) for s in unfinished}),
                complete=sum(s["complete"] for s in strips), unfinished=len(unfinished),
                directions=sorted(set(dirs)), parallel=len({d//2 for d in dirs}) == 1,
                opposing=any(d^1 in dirs for d in dirs), same_direction=len(set(dirs)) < len(dirs),
                clipped=any(s["maximum"] % s["period"] != 0 for s in unfinished),
                full_final=any(s["maximum"] % s["period"] == 0 for s in unfinished),
                background=a["background"], hub_colors=list(a["hub_colors"]),
                palette_size=len(np.unique(pair["input"])), palette_relations=sorted(palette_relations),
                identical_tiles=identical_tiles, mixed_completion_overlap=mixed_completion_overlap,
                hub_geometry=sorted(geometry), separations=sorted(distances),
                boundary_distances=sorted(s["maximum"] for s in strips),
                repeated_tile_colors=any(len({v for row in tile for v in row}) < sum(map(len, tile)) for tile in tiles),
                repeated_tile_slices=any(len(set(tile)) < len(tile) for tile in tiles),
                uniform_tile_rows=any(len(row) > 1 and len(set(row)) == 1 for tile in tiles for row in tile),
                asymmetric_tile_rows=any(len(set(row)) > 1 for tile in tiles for row in tile),
                multiple_visible_copies=any(len(s["slices"]) > s["period"] for s in unfinished),
                crossing=len(a["crossing_cells"]),
                overwrite=a["overwrite_count"], unchanged=a["changed_count"] == 0,
                completed_appearances=sorted({_completed_kind(s) for s in strips if s["complete"]}),
                whole_word_witness=_has_whole_word_witness(a),
                combinations=sorted({(s["width"], s["period"], s["complete"], s["direction"],
                                      s["maximum"]-len(s["slices"])) for s in strips}))


def _observed_regimes(puzzle):
    """Curricula measured from returned examples, without sampled regime labels."""
    training = [_analyze_pair(p) for p in puzzle["train"]]
    testing = [_analyze_pair(p) for p in puzzle["test"]]
    ts = [s for a in training for s in a["strips"] if not s["complete"]]
    vs = [s for a in testing for s in a["strips"] if not s["complete"]]
    if not vs:
        return set()
    train_2d = any(s["width"] > 1 and s["period"] > 1 for s in ts)
    test_2d = any(s["width"] > 1 and s["period"] > 1 for s in vs)
    observed = set()
    if test_2d:
        observed.add("familiar_2d" if train_2d else "compositional")
    if max(s["width"] for s in vs) > max(s["width"] for s in ts):
        observed.add("wider")
    if max(s["period"] for s in vs) > max(s["period"] for s in ts):
        observed.add("longer")
    if max(len(a["strips"]) for a in testing) > max(len(a["strips"]) for a in training):
        observed.add("larger_count")
    return observed


def self_test() -> None:
    assert _minimal_period([(2,), (3,), (2,)]) == 3
    assert _minimal_period([(2,), (3,), (2,), (3,)]) == 2
    for source in _source_pairs():
        _analyze_pair(source)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    observed = [_observed_regimes(p) for p in puzzles]
    for regime in ("compositional", "familiar_2d", "wider", "longer", "larger_count"):
        assert sum(regime in r for r in observed) >= 3, regime
    records = [review_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    for name, minimum in (("shape", 60), ("hubs", 5), ("widths", 7), ("periods", 7),
                          ("palette_size", 5), ("background", 10), ("complete", 4),
                          ("unfinished", 4), ("extensions", 30)):
        assert len({str(f[name]) for f in records}) >= minimum, name
    assert {k for f in records for k in f["hub_shapes"]} == {"cross", "clipped", "rectangle"}
    assert {k for f in records for k in f["completed_appearances"]} == {"monochrome", "periodic", "irregular"}
    assert {k for f in records for k in f["palette_relations"]} == {"same", "overlap", "disjoint"}
    assert {v for f in records for v in f["widths"]} >= {1, 2, 3, 4}
    assert {v for f in records for v in f["periods"]} >= {1, 2, 3, 4, 5}
    assert any(1 in f["extensions"] for f in records)
    assert any(f["extensions"] and max(f["extensions"]) >= 20 for f in records)
    assert any(f["extensions"] and min(f["extensions"]) <= 3 and max(f["extensions"]) >= 10 for f in records)
    for flag in ("unchanged", "identical_tiles", "whole_word_witness", "clipped", "full_final", "opposing", "same_direction",
                 "mixed_completion_overlap", "repeated_tile_colors", "repeated_tile_slices",
                 "uniform_tile_rows", "asymmetric_tile_rows", "multiple_visible_copies"):
        assert any(f[flag] for f in records), flag
    assert any(f["parallel"] and f["hubs"] >= 3 for f in records)
    assert any(not f["parallel"] and f["hubs"] >= 3 for f in records)
    assert any(len(f["widths"]) > 1 and len(f["periods"]) > 1 for f in records)
    assert any(len(f["hub_sizes"]) > 1 and len(f["hub_shapes"]) > 1 for f in records)
    assert any(len(f["hub_colors"]) > 1 for f in records)
    assert any(min(f["boundary_distances"]) <= 2 for f in records)
    assert any(min(f["shape"]) < 12 and max(f["shape"]) >= 20 for f in records)
    for width in (1, 2, 3, 4):
        geometries = {g for f in records for g in f["hub_geometry"] if g[-1] == width}
        assert len(geometries) >= 6, "hub geometry collapsed onto strip width"
    assert len({str(f["combinations"]) for f in records}) >= 150
    for split in ("train", "test"):
        fs = [review_features(pair) for p in puzzles for pair in p[split]]
        assert {x for f in fs for x in f["completed_appearances"]} == {"monochrome", "periodic", "irregular"}
        assert any(f["crossing"] for f in fs)
    for p in puzzles:
        validate_generated_puzzle(p)
        fs = [review_features(pair) for s in ("train", "test") for pair in p[s]]
        assert len({f["hubs"] for f in fs}) > 1
        assert len({f["background"] for f in fs}) > 1
    for axis in ("complete", "unfinished", "directions", "palette_size", "widths", "periods", "extensions"):
        varied = 0
        for p in puzzles:
            fs = [review_features(pair) for s in ("train", "test") for pair in p[s]]
            varied += len({str(f[axis]) for f in fs}) > 1
        assert varied >= 25, (axis, varied)
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(92731, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(92731, a, b))
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # The bounded packer has a deterministic escape path; exercise its widest
    # words with and without crossings even when random sampling never needs it.
    for width in (1, 2, 3, 4):
        for crossing in (False, True):
            pair = _fallback(np.random.default_rng(width), [2, 3, 4, 5], [1], 8,
                             (width, 3, "whole_word"), crossing, True, 3)
            analysis = _analyze_pair(pair)
            assert _irregular_preserved(analysis, pair["input"], pair["output"])
            assert _has_whole_word_witness(analysis)
            if crossing:
                assert analysis["overwrite_count"] > 0
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
    # A mutation outside extension paths must not be silently accepted.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    g = damaged["test"][0]["output"]
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("changed output accepted")


def _source_pairs():
    """Exact staged source grids: regression evidence and content-based exclusion."""
    rows = (
        (
            (
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88881888888888",
                "88811123232323",
                "88881888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888884888",
                "88888888882888",
                "88888888881888",
                "88888888811188",
                "88888888881888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
            ),
            (
                "88888888882888",
                "88888888884888",
                "88888888882888",
                "88888888884888",
                "88881888882888",
                "88811123234323",
                "88881888882888",
                "88888888884888",
                "88888888882888",
                "88888888884888",
                "88888888882888",
                "88888888884888",
                "88888888882888",
                "88888888881888",
                "88888888811188",
                "88888888881888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
                "88888888888888",
            ),
        ),
        (
            (
                "8888888888888",
                "8811888888888",
                "8111188888888",
                "8111188888888",
                "8111188888888",
                "8811888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888842888",
                "8823888811888",
                "8823888111188",
                "8823888111188",
                "8823888111188",
                "8823888811888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
            ),
            (
                "8888888842888",
                "8811888842888",
                "8111188842888",
                "8111188842888",
                "8111188842888",
                "8811888842888",
                "8823888842888",
                "8823888842888",
                "8823888842888",
                "8823888842888",
                "8823888842888",
                "8823888842888",
                "8823888811888",
                "8823888111188",
                "8823888111188",
                "8823888111188",
                "8823888811888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
                "8823888888888",
            ),
        ),
        (
            (
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888888888888888888",
                "888118888888888888",
                "881111888888888888",
                "888118888888888888",
                "888438888888888888",
                "888628888888888888",
                "888438888888888888",
                "888628888888888888",
                "888438888888888888",
                "888628888888888888",
                "888438888258888888",
                "888628888738888888",
                "888438888118888888",
                "888628881111888888",
                "888438888118888888",
                "888628888888888888",
                "888438888888881188",
                "888628888888811118",
                "888438888888881188",
                "888628888888882488",
                "888638888888883588",
                "888428888888888888",
            ),
            (
                "888888888738888888",
                "888888888258888888",
                "888888888738888888",
                "888888888258888888",
                "888118888738888888",
                "881111888258888888",
                "888118888738888888",
                "888438888258888888",
                "888628888738888888",
                "888438888258888888",
                "888628888738888888",
                "888438888258888888",
                "888628888738888888",
                "888438888258888888",
                "888628888738888888",
                "888438888118888888",
                "888628881111888888",
                "888438888118888888",
                "888628888888888888",
                "888438888888881188",
                "888628888888811118",
                "888438888888881188",
                "888628888888882488",
                "888638888888883588",
                "888428888888882488",
            ),
        ),
    )
    return [{k: np.array([[int(c) for c in row] for row in grid], dtype=np.int8)
             for k, grid in zip(("input", "output"), pair)} for pair in rows]
