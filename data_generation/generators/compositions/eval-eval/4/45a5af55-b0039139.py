"""Counted Contour Casting. Standard library + NumPy; no disk/network state."""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _arc_grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "color bounds")
    return grid.astype(np.int8, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _crop(mask):
    points = np.argwhere(mask)
    if not len(points):
        raise ValueError("empty silhouette")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _decode(input_grid):
    """Identify every role and the unique layout from this input alone."""
    grid = _arc_grid(input_grid)
    candidates = []
    for transposed in (False, True):
        g = grid.T if transposed else grid
        for divider in range(10):
            cuts = np.flatnonzero(np.all(g == divider, axis=0))
            if len(cuts) != 1 or not 0 < cuts[0] < g.shape[1] - 1:
                continue
            cut = int(cuts[0])
            recipe, panel = g[:, :cut], g[:, cut+1:]
            if np.any(panel == divider):
                continue
            bars = np.flatnonzero(np.all(recipe == divider, axis=1)).tolist()
            if not 1 <= len(bars) <= 4:
                continue
            edges = [-1] + bars + [g.shape[0]]
            lanes = [recipe[a+1:b] for a, b in zip(edges, edges[1:])]
            if any(not 1 <= lane.shape[0] <= 5 or np.any(lane == divider) for lane in lanes):
                continue
            panel_colors = set(map(int, np.unique(panel)))
            lane_colors = [set(map(int, np.unique(lane))) for lane in lanes]
            if len(panel_colors) != 2 or any(len(cs) != 2 for cs in lane_colors):
                continue
            common = panel_colors.intersection(*lane_colors)
            if len(common) != 1:
                continue
            bg = next(iter(common))
            core = next(iter(panel_colors - {bg}))
            paints = [next(iter(cs - {bg})) for cs in lane_colors]
            if core in paints or divider in paints or len({bg, core, divider}) != 3:
                continue
            masks = [lane == color for lane, color in zip(lanes, paints)]
            counts = [len(_components(mask)) for mask in masks]
            if any(not 1 <= n <= 5 or n != len(_components(m, True))
                   for n, m in zip(counts, masks)):
                continue
            radius = sum(counts)
            mask = _crop(panel == core)
            if not (2 <= radius <= 12 and max(mask.shape) <= 9
                    and max(mask.shape) + 2*radius <= 30
                    and len(_components(mask)) == 1):
                continue
            # This is a hard unique-input parse, never an output-label filter.
            candidates.append({"transposed": transposed, "background": bg,
                               "divider": divider, "core": core, "paints": paints,
                               "counts": counts, "tallies": masks, "mask": mask,
                               "panel_shape": panel.shape, "recipe_width": recipe.shape[1],
                               "lane_depths": [x.shape[0] for x in lanes]})
    if len(candidates) != 1:
        raise ValueError("input must have exactly one casting-recipe parse")
    return candidates[0]


def _depth_code(data):
    return [color for color, n in zip(data["paints"], data["counts"]) for _ in range(n)]


def _distance_render(mask, code, core, background, manhattan=False):
    """Direct distance from original ink; solver's construction."""
    radius = len(code)
    h, w = mask.shape
    rr, cc = np.indices((h + 2*radius, w + 2*radius))
    distance = np.full(rr.shape, 1000, dtype=np.int16)
    for r, c in np.argwhere(mask):
        dr, dc = abs(rr - int(r) - radius), abs(cc - int(c) - radius)
        np.minimum(distance, dr + dc if manhattan else np.maximum(dr, dc), out=distance)
    out = np.full(distance.shape, background, dtype=np.int8)
    out[distance == 0] = core
    for d in range(1, radius + 1):
        out[distance == d] = code[radius-d]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _decode(input_grid)
    out = _distance_render(data["mask"], _depth_code(data), data["core"], data["background"])
    return out.T.copy() if data["transposed"] else out


def _dilation_render(data, code=None, exterior_only=False):
    """Alternate construction for authored validation: grow occupied sets."""
    code = _depth_code(data) if code is None else code
    radius = len(code)
    mask = np.pad(data["mask"], radius)
    out = np.full(mask.shape, data["background"], dtype=np.int8)
    out[mask] = data["core"]
    occupied = mask.copy()
    # exterior_only is an explicit competing rule, used only for evidence checks.
    allowed = np.ones(mask.shape, dtype=bool)
    if exterior_only:
        blank = ~mask
        exterior = np.zeros(mask.shape, dtype=bool)
        for group in _components(blank):
            if any(r in (0, mask.shape[0]-1) or c in (0, mask.shape[1]-1) for r, c in group):
                for p in group:
                    exterior[p] = True
        allowed = exterior | mask
    for color in reversed(code):
        padded = np.pad(occupied, 1)
        grown = np.zeros_like(occupied)
        h, w = occupied.shape
        for dr in range(3):
            for dc in range(3):
                grown |= padded[dr:dr+h, dc:dc+w]
        grown &= allowed
        out[grown & ~occupied] = color
        occupied = grown
    return out


def _grow(rng, h, w, area):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _segments(rng, length, pieces):
    """Positive slots separated by at least one blank cell, with free margins."""
    values = np.ones(2*pieces + 1, dtype=int)
    values[0] = values[-1] = 0
    for _ in range(length - (2*pieces - 1)):
        values[int(rng.integers(len(values)))] += 1
    result, offset = [], int(values[0])
    for i in range(pieces):
        size = int(values[2*i+1])
        result.append((offset, offset+size))
        offset += size + int(values[2*i+2])
    return result


def _tally(rng, h, w, count, exact_area=None):
    if exact_area is not None:
        # Full-height separated slots ensure six cells always fit for n=2/3.
        widths = np.ones(count, dtype=int)
        for _ in range(w - (2*count-1)):
            widths[int(rng.integers(count))] += 1
        boxes, c = [], 0
        for width in widths:
            boxes.append((0, h, c, c+int(width)))
            c += int(width) + 1
    else:
        options = [(nr, nc) for nr in range(1, (h+1)//2+1)
                   for nc in range(1, (w+1)//2+1)
                   if count <= nr*nc <= count+2]
        nr, nc = options[int(rng.integers(len(options)))]
        rows, cols = _segments(rng, h, nr), _segments(rng, w, nc)
        boxes = [(a, b, c, d) for a, b in rows for c, d in cols]
        rng.shuffle(boxes)
        boxes = boxes[:count]
    capacities = [(b-a)*(d-c) for a, b, c, d in boxes]
    areas = np.ones(count, dtype=int)
    if exact_area is not None:
        for _ in range(exact_area-count):
            choices = [i for i in range(count) if areas[i] < capacities[i]]
            areas[choices[int(rng.integers(len(choices)))]] += 1
    else:
        areas = np.array([int(rng.integers(1, cap+1)) for cap in capacities])
    mask = np.zeros((h, w), dtype=bool)
    for (a, b, c, d), area in zip(boxes, areas):
        mask[a:b, c:d] = _grow(rng, b-a, d-c, int(area))
    # Preserve a visibly identifiable background in the single-object limit.
    if mask.all():
        mask[0, 0] = False
    return mask


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4))
               for k in range(1, 8))


def _stencil(rng, h, w, kind=None):
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    if kind is None:
        kind = str(rng.choice(["prune", "ring", "branch", "solid"], p=[.55, .20, .20, .05]))
    mask = np.ones((h, w), dtype=bool)
    if kind == "ring":
        mask[1:-1, 1:-1] = False
        if h > 3 and w > 3:
            for _ in range(int(rng.integers((h-2)*(w-2)))):
                r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
                if (r, c) != (h//2, w//2) and (r in (1, h-2) or c in (1, w-2)):
                    mask[r, c] = True
    elif kind == "branch":
        mask[:] = False
        row, col = int(rng.integers(h)), int(rng.integers(w))
        mask[row, :] = mask[:, col] = True
        for _ in range(int(rng.integers(h*w//2+1))):
            candidates = []
            for r, c in np.argwhere(~mask):
                if any(0 <= x < h and 0 <= y < w and mask[x, y]
                       for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
                    candidates.append((r, c))
            if candidates:
                mask[candidates[int(rng.integers(len(candidates)))]] = True
    elif kind in ("prune", "asymmetric"):
        target = int(rng.integers(h+w-1, h*w))
        for flat in rng.permutation(h*w):
            if mask.sum() <= target:
                break
            r, c = divmod(int(flat), w)
            mask[r, c] = False
            if _crop(mask).shape != (h, w) or len(_components(mask)) != 1:
                mask[r, c] = True
        if kind == "asymmetric":
            if all(mask[r, c] for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))):
                mask[0, 0] = False
                if len(_components(mask)) != 1 or _crop(mask).shape != (h, w):
                    mask[0, 0] = True
            if not _asymmetric(mask) or all(mask[r, c] for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))):
                mask[:] = False
                mask[:, 0] = mask[-1, :] = True
                mask[0, 1] = True
    return mask


def _counts(rng, radius, lanes, maximum):
    counts = np.ones(lanes, dtype=int)
    for _ in range(radius-lanes):
        choices = np.flatnonzero(counts < maximum)
        counts[int(rng.choice(choices))] += 1
    return counts.tolist()


def _make_pair(rng, palette, counts, mask, transposed, witness=None):
    bg, divider, core, *vocabulary = palette
    k = len(counts)
    depths = rng.integers(3, 6, size=k)
    while int(depths.sum()) + k-1 < mask.shape[0]:
        choices = np.flatnonzero(depths < 5)
        depths[int(rng.choice(choices))] += 1
    height = int(depths.sum()) + k-1
    width = int(rng.integers(max(7, 2*max(counts)-1), 15))
    if witness == 0:
        a, b, c = map(int, rng.choice(vocabulary, size=3, replace=False))
        paints = [a, b, a, c]
    elif witness is not None:
        paints = list(map(int, rng.choice(vocabulary, size=k, replace=False)))
    else:
        paints = list(map(int, rng.choice(vocabulary, size=k, replace=True)))
    recipe = np.full((height, width), bg, dtype=np.int8)
    offset = 0
    for i, (depth, n, color) in enumerate(zip(depths, counts, paints)):
        depth = int(depth)
        tally = _tally(rng, depth, width, n, exact_area=6 if witness is not None and i == 0 else None)
        recipe[offset:offset+depth][tally] = color
        offset += depth
        if i+1 < k:
            recipe[offset] = divider
            offset += 1
    max_pad = min(6, 30-width-1-mask.shape[1])
    padding = int(rng.integers(max_pad+1))
    panel = np.full((height, mask.shape[1]+padding), bg, dtype=np.int8)
    top = int(rng.integers(height-mask.shape[0]+1))
    left = int(rng.integers(padding+1))
    panel[top:top+mask.shape[0], left:left+mask.shape[1]][mask] = core
    # At least six spare columns are available; one suffices to identify the
    # background when a solid core happens to fill the chosen panel exactly.
    if np.all(panel == core):
        panel = np.pad(panel, ((0, 0), (0, 1)), constant_values=bg)
    inp = np.concatenate((recipe, np.full((height, 1), divider, dtype=np.int8), panel), axis=1)
    data = {"mask": mask, "counts": counts, "paints": paints, "core": core, "background": bg}
    out = _dilation_render(data)
    if transposed:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _valid_count(value, low, high):
    return (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and low <= int(value) <= high)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_count(num_train, 2, 10) or not _valid_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require integer counts: 2<=train<=10, 1<=test<=4, train>test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))[:int(rng.integers(6, 10))]
    extent_shift = bool(rng.integers(2))
    phase = bool(rng.integers(2))
    puzzle, seen, training_data = {"train": [], "test": []}, set(), []
    for split, amount in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(amount):
            witness = i if split == "train" and i < 2 else None
            if witness is not None:
                k = 4 if i == 0 else int(rng.integers(2, 4))
                counts = [2 if i == 0 else 3] + rng.integers(1, 3, size=k-1).tolist()
                h, w = map(int, rng.integers(3, 6, size=2))
                kind = "asymmetric" if i == 0 else "ring"
                transposed = phase if i == 0 else not phase
            else:
                if split == "train":
                    radius = int(rng.integers(2, 9))
                    maximum = 4
                    limit = 5 if extent_shift else min(9, 30-2*radius)
                    h, w = map(int, rng.integers(1, limit+1, size=2))
                elif extent_shift:
                    extent = int(rng.integers(6, 10))
                    radius = int(rng.integers(2, min(10, (30-extent)//2)+1))
                    h, w = extent, int(rng.integers(1, 10))
                    if rng.integers(2):
                        h, w = w, h
                    maximum = 5
                else:
                    radius = int(rng.integers(max(sum(d["counts"]) for d in training_data)+1, 13))
                    limit = min(9, 30-2*radius)
                    h, w = map(int, rng.integers(1, limit+1, size=2))
                    maximum = 5
                minimum_k = max(2, (radius+maximum-1)//maximum)
                k = int(rng.integers(minimum_k, min(5, radius)+1))
                counts = _counts(rng, radius, k, maximum)
                kind = None
                transposed = bool(rng.integers(2))
            mask = _stencil(rng, h, w, kind)
            pair = _make_pair(rng, palette, counts, mask, transposed, witness)
            if _key(pair["input"]) in seen:
                # Tight cropping makes a change of right-panel padding inert.
                # Padding 0..6 gives 28 distinct placements, more than 13 prior
                # inputs, while preserving the puzzle's literal palette roles.
                g = pair["input"].T if transposed else pair["input"]
                cut = _decode(pair["input"])["recipe_width"]
                prefix = g[:, :cut+1]
                bg, _, core = palette[:3]
                found = False
                for padding in range(7):
                    for left in range(padding+1):
                        panel = np.full((g.shape[0], mask.shape[1]+padding), bg, dtype=np.int8)
                        panel[:mask.shape[0], left:left+mask.shape[1]][mask] = core
                        if np.all(panel == core):
                            continue
                        candidate = np.concatenate((prefix, panel), axis=1)
                        candidate = candidate.T.copy() if transposed else candidate
                        if _key(candidate) not in seen:
                            pair = {"input": candidate, "output": pair["output"]}
                            found = True
                            break
                    if found:
                        break
                _require(found, "uniqueness fallback exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training_data.append(_decode(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_hole(mask):
    padded = np.pad(mask, 1)
    return len(_components(~padded)) > 1


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, train, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            data = _decode(inp)
            canonical = out.T if data["transposed"] else out
            radius = sum(data["counts"])
            _require(canonical.shape == tuple(n+2*radius for n in data["mask"].shape), "casting extent")
            expected = _dilation_render(data)
            _require(np.array_equal(expected, canonical), "casting colors/contours")
            _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
            _require(np.array_equal(canonical[radius:-radius, radius:-radius] == data["core"],
                                    data["mask"]), "core preservation")
            (train if split == "train" else tests).append(data)
    _require({d["transposed"] for d in train} == {False, True}, "both orientations demonstrated")
    _require(any(_asymmetric(d["mask"]) and not all(d["mask"][r, c]
                 for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))) for d in train),
             "asymmetric missing-corner core witness")
    _require(any(_has_hole(d["mask"]) for d in train), "hole witness")
    _require(any(d["panel_shape"] != d["mask"].shape for d in train), "crop witness")
    lanes = [(n, int(m.sum())) for d in train for n, m in zip(d["counts"], d["tallies"])]
    _require(any(a == b and n != k for n, a in lanes for k, b in lanes), "equal area/different counts")
    _require(any(area > n for n, area in lanes), "multicell counter witness")
    # Every following alternative must fail on at least one demonstration.
    failures = {name: False for name in ("reverse", "group_paints", "unit", "area", "lane_depth", "recipe_width",
                                       "box", "diamond", "exterior_only")}
    for data in train:
        code = _depth_code(data)
        expected = _dilation_render(data)
        failures["reverse"] |= not np.array_equal(expected, _dilation_render(data, code[::-1]))
        paint_totals = {}
        for c, n in zip(data["paints"], data["counts"]):
            paint_totals[c] = paint_totals.get(c, 0) + n
        grouped_code = [c for c, n in paint_totals.items() for _ in range(n)]
        failures["group_paints"] |= code != grouped_code
        failures["unit"] |= len(data["paints"]) != len(code)
        failures["area"] |= sum(int(m.sum()) for m in data["tallies"]) != len(code)
        depth_code = [c for c, n in zip(data["paints"], data["lane_depths"]) for _ in range(n)]
        width_code = [c for c in data["paints"] for _ in range(data["recipe_width"])]
        # Any depth-list change changes an exposed extremal ray, or its extent.
        failures["lane_depth"] |= code != depth_code
        failures["recipe_width"] |= code != width_code
        failures["box"] |= not data["mask"].all()
        failures["diamond"] |= not np.array_equal(expected, _distance_render(
            data["mask"], code, data["core"], data["background"], manhattan=True))
        failures["exterior_only"] |= not np.array_equal(expected, _dilation_render(data, exterior_only=True))
    _require(all(failures.values()), "training permits shortcut: " + str(failures))
    radius_max = max(sum(d["counts"]) for d in train)
    extent_max = max(max(d["mask"].shape) for d in train)
    _require(all(sum(d["counts"]) > radius_max or max(d["mask"].shape) > extent_max for d in tests),
             "missing structural test extrapolation")


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming audit of 42 seeds, counts, equivariance, corruption."""
    signatures, shapes, cores, recipes, colors = set(), set(), set(), set(), set()
    radii, lane_numbers, counts, shifts = set(), set(), set(), set()
    holes = simple = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        tds = [_decode(p["input"]) for p in puzzle["train"]]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                shapes.add((pair["input"].shape, pair["output"].shape))
                cores.add(_key(d["mask"]))
                normalized = {}
                recipes.add(tuple((normalized.setdefault(c, len(normalized)), n)
                                  for c, n in zip(d["paints"], d["counts"])))
                colors.update(map(int, np.unique(pair["input"])))
                radii.add(sum(d["counts"]))
                lane_numbers.add(len(d["counts"]))
                counts.update(d["counts"])
                holes += _has_hole(d["mask"])
                simple += bool(d["mask"].all())
                if split == "test":
                    shifts.add("radius" if sum(d["counts"]) > max(sum(x["counts"]) for x in tds) else "extent")
        if seed in (0, 7, 41):
            _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterminism")
            pair = puzzle["test"][0]
            _require(np.array_equal(solve(pair["input"].T), pair["output"].T), "transpose equivariance")
            perm = np.array([8, 3, 6, 1, 9, 0, 4, 7, 2, 5], dtype=np.int8)
            _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "color equivariance")
    _require(len(signatures) == 42, "distinct seeds")
    _require(len(shapes) >= 100 and len(cores) >= 100 and len(recipes) >= 65, "structural variation")
    _require(colors == set(range(10)), "palette variation")
    _require(lane_numbers == set(range(2, 6)) and counts == set(range(1, 6)), "lane/count variation")
    _require(len(radii) >= 9 and {2, 3, 8} <= radii and max(radii) >= 11, "radius variation")
    _require(shifts == {"radius", "extent"} and holes >= 42 and simple >= 10, "topology/shift variation")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit geometric endpoints supplement, rather than distort, random
    # coverage frequencies. These fixtures use only their supplied inputs.
    boundary = _make_pair(np.random.default_rng(800), [0, 5, 2, 1, 3, 4],
                          [5, 5, 2], np.ones((6, 6), dtype=bool), False)
    _require(boundary["output"].shape == (30, 30)
             and np.array_equal(solve(boundary["input"]), boundary["output"]), "radius/grid maximum")
    minimal = np.array([[2, 0, 0, 5, 0, 0, 0], [5, 5, 5, 5, 0, 4, 0],
                        [3, 0, 0, 5, 0, 0, 0]], dtype=np.int8)
    answer = np.array([[2, 2, 2, 2, 2], [2, 3, 3, 3, 2], [2, 3, 4, 3, 2],
                       [2, 3, 3, 3, 2], [2, 2, 2, 2, 2]], dtype=np.int8)
    _require(np.array_equal(solve(minimal), answer), "one-cell core and one-cell lanes")
    monochrome = _make_pair(np.random.default_rng(801), [0, 5, 2, 7],
                            [1, 3], np.ones((1, 2), dtype=bool), True)
    _require(np.array_equal(solve(monochrome["input"]), monochrome["output"]), "adjacent equal paints")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "binary_cores": len(cores), "normalized_recipes": len(recipes),
            "radii": sorted(radii), "lane_counts": sorted(lane_numbers),
            "object_counts": sorted(counts), "hole_examples": holes, "solid_examples": simple,
            "test_shifts": sorted(shifts), "boundary_fixtures": ["radius12-grid30", "one-cell", "equal-paints"],
            "independent_semantic_verification": False}
