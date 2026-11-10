"""Transparent layers dock against a shape selected by an instruction key.

Only standard-library modules and NumPy are used.  The solver reads pixels;
the random sampler never provides it with parameters or hidden state.
"""

from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer))
             and low <= value <= high, f"illegal {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer)
             and np.all((0 <= a) & (a <= 9)), "grid colors/type")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _bbox(coords):
    a = np.asarray(list(coords), dtype=int)
    _require(len(a) > 0, "empty object")
    lo, hi = a.min(axis=0), a.max(axis=0)
    return tuple(map(int, (*lo, *(hi + 1))))


def _relative(mask):
    cr, cc = np.array(mask.shape) // 2
    return frozenset((int(r) - int(cr), int(c) - int(cc))
                     for r, c in np.argwhere(mask))


def _t_mask(size, side):
    a = np.zeros((size, size), dtype=bool)
    a[:, 0 if side == 1 else -1] = True
    a[size // 2, :] = True
    return a


def _distance(reference, layer, side):
    """Last collision encountered while approaching from the indicated side.

    s is the signed-side distance between the layer and reference centers.
    Collisions can occur only between pixels on the same registered row.
    Starting outside, the first collision has the greatest s; stop at s+1.
    This does not search for a disconnected 'legal pocket' behind a collision.
    """
    collisions = [(rc - pc) * side for rr, rc in reference
                  for pr, pc in layer if rr == pr]
    _require(collisions and max(collisions) >= 0, "layer has no docking contact")
    return max(collisions) + 1


def _docked(reference, layers, side):
    distances = [_distance(reference, p, side) for p in layers]
    return distances, [frozenset((r, c + side * s) for r, c in p)
                       for p, s in zip(layers, distances)]


def _transparent_witness(reference, targets):
    """An opaque bounding-box paste would erase another object's pixel."""
    for i, layer in enumerate(targets):
        t, l, b, r = _bbox(layer)
        others = set(reference).union(*(p for j, p in enumerate(targets) if j != i))
        if any(t <= y < b and l <= x < r for y, x in others):
            return True
    return False


def _paint(grid, coords, color, center=(0, 0)):
    for r, c in coords:
        grid[center[0] + r, center[1] + c] = color


def _canonical(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > grid.size // 2, "background is not a majority")
    divs = [r for r in range(grid.shape[0])
            if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    _require(len(divs) == 1 and 0 < divs[0] < grid.shape[0] - 1, "divider")
    cut = divs[0]
    div_color = int(grid[cut, 0])
    _require(np.count_nonzero(grid == div_color) == grid.shape[1], "divider role")
    upper, work = grid[:cut], grid[cut + 1:]
    objects = []
    for color in np.unique(upper):
        if color == bg:
            continue
        coords = np.argwhere(upper == color)
        t, l, b, r = _bbox(coords)
        objects.append((l, t, r, b, int(color), upper[t:b, l:r] == color))
    objects.sort(key=lambda x: x[0])
    _require(4 <= len(objects) <= 5, "reference, T, and two/three layers required")
    ref, marker, *payloads = objects
    _require(ref[2] < marker[0] and marker[2] < min(p[0] for p in payloads),
             "instruction order/gaps")
    ref_mask, marker_mask = ref[5], marker[5]
    rh, rw = ref_mask.shape
    _require(rh >= 3 and rw >= 3 and rh % 2 and rw % 2, "reference extent")
    _require(np.all(ref_mask[:, rw // 2]), "reference spine")
    _require(marker_mask.shape in ((3, 3), (5, 5)), "T extent")
    sides = [d for d in (-1, 1)
             if np.array_equal(marker_mask, _t_mask(marker_mask.shape[0], d))]
    _require(len(sides) == 1, "T shape")
    side = sides[0]
    layers, colors, centers, supports = [], [], [], []
    for l, t, r, b, color, mask in payloads:
        h, w = mask.shape
        _require(3 <= h <= 9 and 3 <= w <= 9 and h % 2 and w % 2, "layer extent")
        p = _relative(mask)
        _require(len(p) >= 4 and all((-y, x) in p and (y, -x) in p for y, x in p),
                 "layer reflection symmetries")
        rows = {y for y, _ in p}
        _require(all(rows.isdisjoint(old) for old in supports), "layers share a lane")
        layers.append(p)
        colors.append(color)
        centers.append(((t + b - 1) // 2, (l + r - 1) // 2))
        supports.append(rows)
    _require(len(set(centers)) > 1, "source layers already share a center")
    ref_color = ref[4]
    _require(set(map(int, np.unique(work))) == {bg, ref_color}, "work palette")
    template = np.full((rh, rw), bg, dtype=np.int8)
    template[ref_mask] = ref_color
    matches = [(r, c) for r in range(work.shape[0] - rh + 1)
               for c in range(work.shape[1] - rw + 1)
               if np.array_equal(work[r:r + rh, c:c + rw], template)]
    _require(len(matches) == 1, "reference match must be unique")
    at, al = matches[0]
    center = (at + rh // 2, al + rw // 2)
    reference = _relative(ref_mask)
    distances, destinations = _docked(reference, layers, side)
    output = work.copy()
    for color, cells in zip(colors, destinations):
        for r, c in cells:
            rr, cc = center[0] + r, center[1] + c
            _require(0 <= rr < work.shape[0] and 0 <= cc < work.shape[1], "clipped dock")
            _require(output[rr, cc] == bg, "dock collision")
            output[rr, cc] = color
    return {"background": bg, "divider": div_color, "reference_color": ref_color,
            "reference_mask": ref_mask, "reference": reference, "layers": layers,
            "layer_colors": colors, "source_centers": centers, "side": side,
            "center": center, "distances": distances, "destinations": destinations,
            "output": output, "work": work, "cut": cut, "marker_box": marker[:4],
            "marker_color": marker[4], "upper": upper}


def _analyze(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        try:
            a = _canonical(np.rot90(grid, rotation))
        except ValueError:
            continue
        a["rotation"] = rotation
        candidates.append(a)
    _require(len(candidates) == 1, "input has no unique instruction orientation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return np.rot90(a["output"], -a["rotation"]).copy()


def _sample_reference(rng, height, width):
    """A spine with uneven teeth; the exact pixel profile controls contact."""
    r = width // 2
    mask = np.zeros((height, width), dtype=bool)
    for row in range(height):
        left, right = map(int, rng.integers(0, r + 1, size=2))
        mask[row, r - left:r + right + 1] = True
    # Extrema remain visible, but need not be on a layer's contact rows.
    mask[int(rng.integers(height)), :r + 1] = True
    mask[int(rng.integers(height)), r:] = True
    if rng.random() < .35:
        holes = rng.random(mask.shape) < .25
        holes[:, r] = False
        mask[holes] = False
        # Removing teeth must not hide the nominal bounding box.
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
    return mask


def _sample_layers(rng, ref_height, count, test):
    # Distinct row supports make simultaneous docking independent of order.
    # The cross gets the center lane; corner orbits get other symmetric lanes.
    radii = list(range(1, max(3, ref_height // 2) + 1))
    rng.shuffle(radii)
    chosen = radii[:count]
    cross = int(rng.integers(count))
    # Any row radius beyond the reference must belong to the cross, whose
    # center-row arms still contact the spine.
    outside = [i for i, r in enumerate(chosen) if r > ref_height // 2]
    if len(outside) > 1:
        return None
    if outside:
        cross = outside[0]
    layers = []
    for i, row_radius in enumerate(chosen):
        col_radius = int(rng.integers(1, 5 if test else 4))
        if test and i == 0:
            col_radius = 4  # genuine new horizontal span in every test
        if i == cross:
            p = {(-row_radius, 0), (row_radius, 0), (0, -col_radius), (0, col_radius)}
            if rng.random() < .25:
                x = int(rng.integers(1, col_radius + 1))
                p.update((y, c) for y in (-row_radius, row_radius) for c in (-x, x))
        else:
            p = {(y, c) for y in (-row_radius, row_radius)
                 for c in (-col_radius, col_radius)}
            if rng.random() < .25:
                x = int(rng.integers(0, col_radius))
                p.update((y, c) for y in (-row_radius, row_radius) for c in (-x, x))
        layers.append(frozenset(p))
    return layers


def _sample_shapes(rng, count, test):
    for _ in range(300):
        rh = int(rng.choice((5, 7) if count == 3 else (3, 5, 7)))
        rw = int(rng.choice((3, 5, 7)))
        ref_mask = _sample_reference(rng, rh, rw)
        reference = _relative(ref_mask)
        layers = _sample_layers(rng, rh, count, test)
        if layers is None:
            continue
        good = True
        for side in (-1, 1):
            distances, targets = _docked(reference, layers, side)
            # Teach actual-pixel contact and independently moving layers.
            interlock = any(s < rw // 2 + max(abs(c) for _, c in p) + 1
                            for s, p in zip(distances, layers))
            if (not interlock or len(set(distances)) == 1
                    or not _transparent_witness(reference, targets)):
                good = False
        if good:
            return ref_mask, layers
    raise RuntimeError("could not sample a docking profile")


def _decoy(rng, mask):
    candidates = [(r, c) for r in range(mask.shape[0])
                  for c in range(mask.shape[1]) if c != mask.shape[1] // 2]
    rng.shuffle(candidates)
    for r, c in candidates:
        d = mask.copy()
        d[r, c] = not d[r, c]
        if d[:, 0].any() and d[:, -1].any():
            return d
    raise RuntimeError("could not change reference")


def _colors(rng, context, count, test):
    pool = context["pool"] if test else context["train_pool"]
    if context["stable"]:
        roles = context["roles"]
    else:
        roles = tuple(map(int, rng.choice(pool, 4, replace=False)))
    available = [c for c in pool if c not in roles]
    layers = list(map(int, rng.choice(available, count, replace=False)))
    held = context["held"]
    if test and held is not None and held not in roles and rng.random() < .65:
        if held not in layers:
            layers[int(rng.integers(count))] = held
    return (*roles, *layers)


def _make_pair(rng, context, count, side, rotation, test=False):
    ref_mask, layers = _sample_shapes(rng, count, test)
    rh, rw = ref_mask.shape
    reference = _relative(ref_mask)
    max_row = max(rh // 2, *(abs(r) for p in layers for r, _ in p))
    payload_height = 2 * max_row + 1
    marker_size = int(rng.choice((3, 5)))
    upper_h = max(payload_height, rh, marker_size) + int(rng.integers(0, 3))
    minimum_work_h = payload_height + rh + 1
    upper_h = min(upper_h, 29 - minimum_work_h)
    _require(upper_h >= max(payload_height, marker_size), "upper layout too tall")
    work_h = int(rng.integers(minimum_work_h, 30 - upper_h))
    max_pw = max(2 * max(abs(c) for _, c in p) + 1 for p in layers)
    both_cells = set(reference)
    for d in (-1, 1):
        _, targets = _docked(reference, layers, d)
        both_cells.update(cell for p in targets for cell in p)
    _, min_c, _, end_c = _bbox(both_cells)
    min_width = max(21, end_c - min_c, rw + marker_size + max_pw + 4)
    width = int(rng.integers(min_width, 31))
    colors = _colors(rng, context, count, test)
    bg, divider, rc, mc, *pcs = colors
    inp = np.full((upper_h + 1 + work_h, width), bg, dtype=np.int8)
    inp[upper_h] = divider
    # Two positional instruction roles; the remaining colors are free layers.
    extra = width - (rw + marker_size + max_pw + 2)
    gaps = rng.multinomial(extra, [.20, .20, .15, .45])
    ref_left = int(gaps[0])
    marker_left = ref_left + rw + 1 + int(gaps[1])
    source_left = marker_left + marker_size + 1 + int(gaps[2])
    rt = int(rng.integers(0, upper_h - rh + 1))
    mt = int(rng.integers(0, upper_h - marker_size + 1))
    inp[rt:rt + rh, ref_left:ref_left + rw][ref_mask] = rc
    inp[mt:mt + marker_size, marker_left:marker_left + marker_size][
        _t_mask(marker_size, side)] = mc
    source_centers = []
    occupied = set()
    for p, color in zip(layers, pcs):
        pr = max(abs(r) for r, _ in p)
        pc = max(abs(c) for _, c in p)
        options = [(r, c) for r in range(pr, upper_h - pr)
                   for c in range(source_left + pc, width - pc)]
        rng.shuffle(options)
        for center in options:
            cells = {(r + center[0], c + center[1]) for r, c in p}
            if cells.isdisjoint(occupied) and center not in source_centers:
                break
        else:
            # Bounded resampling of a whole example, never a different rule.
            return None
        occupied.update(cells)
        source_centers.append(center)
        _paint(inp, p, color, center)
    # Reserve a separate band for a same-color near-match. It is preserved.
    decoy_above = bool(rng.integers(2))
    if decoy_above:
        decoy_top = int(rng.integers(0, work_h - payload_height - rh))
        low = decoy_top + rh + 1 + max_row
        high = work_h - max_row - 1
    else:
        decoy_top = int(rng.integers(payload_height + 1, work_h - rh + 1))
        low = max_row
        high = decoy_top - max_row - 2
    center_row = int(rng.integers(low, high + 1))
    low_c, high_c = -min_c, width - end_c
    # Training uses varied legal positions. Tests emphasize an exact boundary.
    if test or rng.random() < .30:
        center_col = low_c if side < 0 else high_c
    else:
        center_col = int(rng.integers(low_c, high_c + 1))
    center = (center_row, center_col)
    work = inp[upper_h + 1:]
    _paint(work, reference, rc, center)
    dm = _decoy(rng, ref_mask)
    dl = int(rng.integers(0, width - rw + 1))
    work[decoy_top:decoy_top + rh, dl:dl + rw][dm] = rc
    output = work.copy()
    _, targets = _docked(reference, layers, side)
    for color, cells in zip(pcs, targets):
        _paint(output, cells, color, center)
    return {"input": np.rot90(inp, rotation).copy(),
            "output": np.rot90(output, rotation).copy()}


def _reverse_pair(pair):
    a = _analyze(pair["input"])
    canonical = np.rot90(pair["input"], a["rotation"]).copy()
    l, t, r, b = a["marker_box"]
    canonical[t:b, l:r] = a["background"]
    canonical[t:b, l:r][_t_mask(b - t, -a["side"])] = a["marker_color"]
    inp = np.rot90(canonical, -a["rotation"]).copy()
    return {"input": inp, "output": solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    pool = list(map(int, rng.permutation(10)[:int(rng.integers(8, 11))]))
    held = pool[-1] if rng.random() < .5 else None
    train_pool = [c for c in pool if c != held]
    context = {"pool": pool, "train_pool": train_pool, "held": held,
               "stable": bool(rng.integers(2)), "roles": tuple(train_pool[:4])}
    rotation = int(rng.integers(4))
    witness_side = int(rng.choice((-1, 1)))
    seen = set()

    def make(count, side, test, scene_rotation):
        for _ in range(50):
            pair = _make_pair(rng, context, count, side, scene_rotation, test)
            if pair is None or _key(pair["input"]) in seen:
                continue
            # Parsing also checks that the near-match didn't create a second key.
            _analyze(pair["input"])
            seen.add(_key(pair["input"]))
            return pair
        raise RuntimeError("could not construct a distinct source layout")

    first = make(2, witness_side, False, rotation)
    second = _reverse_pair(first)
    seen.add(_key(second["input"]))
    train = [first, second]
    scene_rotations = [rotation]
    for i in range(2, num_train):
        r = int(rng.integers(4)) if i == 2 and rng.random() < .45 else rotation
        scene_rotations.append(r)
        train.append(make(int(rng.integers(2, 4)), int(rng.choice((-1, 1))), False, r))
    rng.shuffle(train)
    test = [make(int(rng.integers(2, 4)), int(rng.choice((-1, 1))), True,
                 int(rng.choice(scene_rotations))) for _ in range(num_test)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _metrics(a):
    ref_radius = a["reference_mask"].shape[1] // 2
    interlocks = sum(s < ref_radius + max(abs(c) for _, c in p) + 1
                     for s, p in zip(a["distances"], a["layers"]))
    edge = any(a["center"][1] + c in (0, a["work"].shape[1] - 1)
               for p in a["destinations"] for _, c in p)
    return interlocks, edge


def _witness(a, b):
    # Identify the exact T-only intervention in input pixels, irrespective of
    # the order in which demonstrations were returned.
    if a["rotation"] != b["rotation"] or a["side"] != -b["side"]:
        return False
    if not np.array_equal(a["work"], b["work"]) or a["upper"].shape != b["upper"].shape:
        return False
    ua, ub = a["upper"].copy(), b["upper"].copy()
    for obj, upper in ((a, ua), (b, ub)):
        l, t, r, bot = obj["marker_box"]
        upper[t:bot, l:r] = obj["background"]
    return np.array_equal(ua, ub) and a["center"] == b["center"]


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _integer(len(puzzle["train"]), 2, 10, "training length")
    _integer(len(puzzle["test"]), 1, 4, "test length")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "split counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = np.rot90(a["output"], -a["rotation"])
            _require(np.array_equal(out, expected), "incorrect docking output")
            bg, work = a["background"], a["work"]
            _require(np.array_equal(a["output"][work != bg], work[work != bg]),
                     "reference/decoy was changed")
            ref = a["reference"]
            _require(np.count_nonzero(work != bg) > len(ref), "missing matching distractor")
            _require(_metrics(a)[0] > 0 and len(set(a["distances"])) > 1,
                     "missing contact / independent-depth witnesses")
            _require(_transparent_witness(ref, a["destinations"]),
                     "opaque pasting would accidentally suffice")
            for color, p, s, target in zip(a["layer_colors"], a["layers"],
                                           a["distances"], a["destinations"]):
                _require(target.isdisjoint(ref), "stopped on reference")
                next_step = {(r, c - a["side"]) for r, c in target}
                _require(bool(next_step.intersection(ref)), "no actual-pixel contact")
                # Every earlier inward position would have been reached without
                # collision: no jumping through the first obstruction.
                start = a["reference_mask"].shape[1] // 2 + max(abs(c) for _, c in p) + 1
                for offset in range(s, start + 1):
                    _require(ref.isdisjoint((r, c + a["side"] * offset) for r, c in p),
                             "layer jumped through an obstacle")
                _require(np.count_nonzero(a["output"] == color) == len(p), "layer not copied exactly")
            # Reverse direction is legal too: available space cannot reveal T.
            reverse = _reverse_pair(pair)
            _require(not np.array_equal(reverse["output"], out), "T has no effect")
            analyses[split].append(a)
    train = analyses["train"]
    _require(any(_witness(a, b) for a in train for b in train if a is not b),
             "training lacks a T-only opposite-direction intervention")
    _require(any(len({max(abs(r) for r, _ in p) for p in a["layers"]}) > 1 for a in train),
             "no unequal-height center-registration witness")
    _require(all(any(max(abs(c) for _, c in p) == 4 for p in a["layers"])
                 for a in analyses["test"]), "missing held-out nine-cell layer span")
    _require(all(max(abs(c) for _, c in p) <= 3 for a in train for p in a["layers"]),
             "training scale exceeded")


def self_test():
    """Bounded, deterministic structural and semantic exercise over 42 seeds."""
    measures = Counter()
    shapes, geometries, directions, rotations, backgrounds, sizes, fingerprints = (set() for _ in range(7))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "self solve")
                # List input/output representations are part of the interface.
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list solve")
                shapes.add(pair["input"].shape)
                backgrounds.add(a["background"])
                directions.add((a["rotation"], a["side"]))
                rotations.add(a["rotation"])
                sizes.add(a["reference_mask"].shape)
                geometries.add((tuple(sorted(a["reference"])),
                                tuple(sorted(tuple(sorted(p)) for p in a["layers"]))))
                interlocks, edge = _metrics(a)
                measures["pairs"] += 1
                measures["interlocking_pairs"] += interlocks > 0
                measures["edge_pairs"] += edge
                measures[f"layers_{len(a['layers'])}"] += 1
                measures["rich_layers"] += sum(len(p) > 4 for p in a["layers"])
                measures["four_cell_layers"] += sum(len(p) == 4 for p in a["layers"])
                measures["asymmetric_reference"] += not np.array_equal(
                    a["reference_mask"], np.fliplr(a["reference_mask"]))
                # A concrete shortcut: align the tops of the payload boxes to
                # the top of the reference instead of aligning their centers.
                top_aligned = [frozenset((r - a["reference_mask"].shape[0] // 2
                                          + max(y for y, _ in p), c)
                                         for r, c in target)
                               for p, target in zip(a["layers"], a["destinations"])]
                _require(top_aligned != a["destinations"], "top alignment shortcut")
                measures["top_alignment_failures"] += 1
                measures["opaque_paste_failures"] += _transparent_witness(
                    a["reference"], a["destinations"])
    _require(len(fingerprints) == 42 and len(shapes) >= 30 and len(geometries) >= 100,
             "insufficient structural variation")
    _require(len(rotations) == 4 and len(directions) == 8 and len(backgrounds) >= 8
             and len(sizes) >= 7, "missing palette/geometry directions")
    for metric in ("edge_pairs", "layers_2", "layers_3", "rich_layers", "four_cell_layers",
                   "asymmetric_reference"):
        _require(measures[metric] >= 15, f"missing coverage: {metric}")
    _require(measures["interlocking_pairs"] == measures["pairs"], "box-packing shortcut survives")
    p, q = generate_puzzle(19), generate_puzzle(19)
    _require(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                 for a, b in zip(p[s], q[s]) for k in ("input", "output")), "determinism")
    # Every legal count, including two-example minimal demonstrations.
    for nt in range(2, 11):
        for nv in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nv, nt, nv)
            _require((len(p["train"]), len(p["test"])) == (nt, nv), "explicit counts")
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return {"seeds": 42, **dict(measures), "input_shapes": len(shapes),
            "geometries": len(geometries), "reference_shapes": len(sizes)}


if __name__ == "__main__":
    print(self_test())
