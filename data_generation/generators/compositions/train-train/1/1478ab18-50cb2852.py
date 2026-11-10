"""Clipped windows: frame the intersection of a rectangle and an inferred triangle.

Only standard-library modules and NumPy are used. No parent code is imported.
The public solver reconstructs all roles and geometry from the input.
"""

import json
from collections import Counter

import numpy as np


ORIENTATIONS = ("UL", "UR", "LL", "LR")
HOST_SEEDS = (0, 792985797, 892938434, 394349309, 415210852, 96452773,
              805721664, 469515418, 589325327, 248590980, 992187049,
              108331588, 847541032, 306965022, 448398579, 497199932,
              935670661, 679846111, 668384999, 756222304, 429607055,
              566762438, 800834326, 867801692, 869655414, 950688774,
              117675431, 466674348, 96509449, 24002782, 323372317,
              370057291, 715981088, 146055733, 970728832, 481953825,
              486953987, 137981019, 116259674, 825169111, 412400633,
              821086967)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _quartet(points):
    rows, cols = zip(*points)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    side = r1 - r0
    if side < 3 or side != c1 - c0:
        return None
    corners = dict(zip(ORIENTATIONS, ((r0, c0), (r0, c1), (r1, c0), (r1, c1))))
    missing = [name for name, point in corners.items() if point not in points]
    if len(missing) != 1:
        return None
    orientation = missing[0]
    mr, mc = corners[orientation]
    dr = 1 if orientation[0] == "U" else -1
    dc = 1 if orientation[1] == "L" else -1
    if set(points) != (set(corners.values()) - {(mr, mc)}) | {(mr + dr, mc + dc)}:
        return None
    return dict(bounds=(r0, c0, r1, c1), side=side, orientation=orientation,
                missing=(mr, mc), direction=(dr, dc))


def _components(grid, excluded):
    """Find monochromatic four-connected material components, not a color bbox."""
    visited = np.zeros(grid.shape, dtype=bool)
    boxes = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in excluded:
                continue
            stack, points = [(r, c)], []
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and grid[yy, xx] == color):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            ys, xs = zip(*points)
            r0, c0, r1, c1 = min(ys), min(xs), max(ys), max(xs)
            _require(r1-r0 >= 2 and c1-c0 >= 2, "material must be at least 3 by 3")
            _require(len(points) == (r1-r0+1)*(c1-c0+1), "material must be solid rectangles")
            boxes.append((r0, c0, r1, c1, color))
    return sorted(boxes)


def _analyze(grid):
    grid = _grid(grid)
    corner_counts = Counter(map(int, (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])))
    background, n = corner_counts.most_common(1)[0]
    _require(n >= 3, "surrounding background must occupy at least three canvas corners")
    colors, counts = np.unique(grid, return_counts=True)
    candidates = []
    for color, count in zip(colors, counts):
        if color == background or count != 4:
            continue
        info = _quartet([tuple(map(int, p)) for p in np.argwhere(grid == color)])
        if info is not None:
            candidates.append((int(color), info))
    _require(len(candidates) == 1, "one unambiguous four-cell square cue is required")
    cue, info = candidates[0]
    boxes = _components(grid, {background, cue})
    _require(bool(boxes), "at least one material rectangle required")
    return dict(info, background=background, cue=cue, boxes=boxes)


def _triangle_masks(shape, info):
    rows, cols = np.indices(shape)
    mr, mc = info["missing"]
    dr, dc = info["direction"]
    u, v = (rows-mr)*dr, (cols-mc)*dc
    side = info["side"]
    closed = (u >= 0) & (v >= 0) & (u+v <= side)
    strict = (u > 0) & (v > 0) & (u+v < side)
    diagonal = (u >= 0) & (v >= 0) & (u+v == side)
    legs = ((u == 0) & (v >= 0) & (v <= side)) | ((v == 0) & (u >= 0) & (u <= side))
    return closed, strict, diagonal, legs


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erode each triangle/material intersection once using four cardinal neighbors.

    This is a direct input-only construction. Padding treats the outside of the
    canvas as outside the material, so canvas-touching perimeters are preserved.
    """
    grid = _grid(input_grid)
    info = _analyze(grid)
    triangle, _, _, _ = _triangle_masks(grid.shape, info)
    output = grid.copy()
    for r0, c0, r1, c1, _ in info["boxes"]:
        intersection = np.zeros(grid.shape, dtype=bool)
        intersection[r0:r1+1, c0:c1+1] = triangle[r0:r1+1, c0:c1+1]
        p = np.pad(intersection, 1, constant_values=False)
        interior = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
                    & p[1:-1, :-2] & p[1:-1, 2:])
        output[interior] = info["cue"]
    return output


def _analytic_output(grid, info):
    """Equivalent coordinate formula, used for generation and semantic validation."""
    _, strict, _, _ = _triangle_masks(grid.shape, info)
    output = grid.copy()
    for r0, c0, r1, c1, _ in info["boxes"]:
        view = output[r0+1:r1, c0+1:c1]
        view[strict[r0+1:r1, c0+1:c1]] = info["cue"]
    return output


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _adjacent(a, b):
    return (((a[2]+1 == b[0] or b[2]+1 == a[0]) and max(a[1], b[1]) <= min(a[3], b[3]))
            or ((a[3]+1 == b[1] or b[3]+1 == a[1]) and max(a[0], b[0]) <= min(a[2], b[2])))


def _box_stats(box, masks):
    closed, strict, diagonal, legs = masks
    r0, c0, r1, c1 = box[:4]
    inside = (slice(r0+1, r1), slice(c0+1, c1))
    area = (r1-r0-1)*(c1-c0-1)
    changed = int(strict[inside].sum())
    perimeter = strict[r0:r1+1, c0:c1+1].copy()
    perimeter[1:-1, 1:-1] = False
    return dict(changed=changed, full=changed == area, partial=0 < changed < area,
                inactive=changed == 0, diagonal=int(diagonal[inside].sum()),
                leg=int(legs[inside].sum()), outside=int((~closed[inside]).sum()),
                protected_perimeter=int(perimeter.sum()))


def _extent(rng, limit, large=False):
    maximum = min(14 if large else 11, limit)
    if not large and rng.random() < .32:
        return int(rng.integers(3, min(5, maximum)+1))
    return int(rng.integers(3, maximum+1))


def _sample_box(rng, mode, h, w, top, left, side, boxes):
    if mode == "cross":
        u = int(rng.integers(3, side-2))
        cy, cx = top+u, left+side-u
        bh, bw = int(rng.integers(5, min(13, h)+1)), int(rng.integers(5, min(13, w)+1))
        r, c = cy-int(rng.integers(1, bh-1)), cx-int(rng.integers(1, bw-1))
    elif mode == "full":
        bh, bw = int(rng.integers(3, min(8, side-2)+1)), int(rng.integers(3, min(8, side-2)+1))
        r = top+int(rng.integers(1, side-bh+1))
        c = left+int(rng.integers(1, side-bw+1))
    elif mode == "leg":
        bh, bw = int(rng.integers(5, min(11, h)+1)), int(rng.integers(5, min(11, w)+1))
        cy, cx = top, left+int(rng.integers(3, side-2))
        if rng.integers(2):
            cy, cx = top+int(rng.integers(3, side-2)), left
        r, c = cy-int(rng.integers(1, bh-1)), cx-int(rng.integers(1, bw-1))
    elif mode == "adjacent" and boxes:
        bh, bw = _extent(rng, h), _extent(rng, w)
        anchor = boxes[int(rng.integers(len(boxes)))]
        direction = int(rng.integers(4))
        if direction < 2:
            r = anchor[2]+1 if direction == 0 else anchor[0]-bh
            c = int(rng.integers(max(0, anchor[1]-bw+1), min(w-bw, anchor[3])+1))
        else:
            c = anchor[3]+1 if direction == 2 else anchor[1]-bw
            r = int(rng.integers(max(0, anchor[0]-bh+1), min(h-bh, anchor[2])+1))
    else:
        bh, bw = _extent(rng, h, True), _extent(rng, w, True)
        r = int(rng.choice((0, h-bh))) if rng.random() < .22 else int(rng.integers(h-bh+1))
        c = int(rng.choice((0, w-bw))) if rng.random() < .22 else int(rng.integers(w-bw+1))
    box = (r, c, r+bh-1, c+bw-1)
    if r < 0 or c < 0 or box[2] >= h or box[3] >= w:
        return None
    if any(_overlap(box, old) for old in boxes):
        return None
    return box


def _make_pair(rng, kind, orientation, roles, seen):
    background, cue = roles
    for _ in range(220):
        side = int(rng.integers(10 if kind == "full" else 8, 28))
        # A leg-crossing rectangle needs room on the missing corner's outer side.
        if kind == "leg" and side > 26:
            continue
        minimum = 2 if kind == "leg" else 0
        extra_y = int(rng.integers(minimum, min(12, 29-side)+1))
        extra_x = int(rng.integers(minimum, min(12, 29-side)+1))
        top = int(rng.integers(1 if kind == "leg" else 0, extra_y+1))
        left = int(rng.integers(1 if kind == "leg" else 0, extra_x+1))
        h, w = side+1+extra_y, side+1+extra_x
        info = dict(missing=(top, left), direction=(1, 1), side=side)
        masks = _triangle_masks((h, w), info)
        markers = {(top, left+side), (top+side, left), (top+side, left+side), (top+1, left+1)}
        count = int(rng.integers(2, 6))
        modes = ["cross", kind] if kind != "free" else ["cross", "free"]
        modes += [str(rng.choice(("free", "adjacent", "full", "leg"), p=(.48, .30, .12, .10)))
                  for _ in range(count-2)]
        boxes = []
        for mode in modes:
            for _attempt in range(160):
                box = _sample_box(rng, mode, h, w, top, left, side, boxes)
                if box is None or any(box[0] <= y <= box[2] and box[1] <= x <= box[3] for y, x in markers):
                    continue
                stat = _box_stats(box, masks)
                if mode == "cross" and not (stat["partial"] and stat["diagonal"] and stat["outside"] and stat["protected_perimeter"]):
                    continue
                if mode == "full" and not stat["full"]:
                    continue
                if mode == "leg" and not (stat["changed"] and stat["leg"] and stat["outside"]):
                    continue
                boxes.append(box)
                break
            else:
                break
        if len(boxes) != count:
            continue
        palette = [int(c) for c in rng.permutation(10) if c not in (background, cue)]
        pool_size = int(rng.integers(1, min(4, count)+1))
        if any(_adjacent(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:]):
            pool_size = max(pool_size, 3)
        pool, colors = palette[:pool_size], []
        for i, box in enumerate(boxes):
            forbidden = {colors[j] for j in range(i) if _adjacent(box, boxes[j])}
            options = [c for c in pool if c not in forbidden]
            if not options:
                options = [c for c in palette if c not in forbidden]
            colors.append(int(rng.choice(options)))
        grid = np.full((h, w), background, dtype=np.int8)
        for (r0, c0, r1, c1), color in zip(boxes, colors):
            grid[r0:r1+1, c0:c1+1] = color
        for point in markers:
            grid[point] = cue
        if sum(grid[y, x] == background for y, x in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))) < 3:
            continue
        # Transposition precedes flips: UL remains UL before the requested orientation.
        if rng.integers(2):
            grid = grid.T
        if orientation[0] == "L":
            grid = np.flipud(grid)
        if orientation[1] == "R":
            grid = np.fliplr(grid)
        grid = grid.copy()
        if _signature(grid) in seen:
            continue
        parsed = _analyze(grid)
        _require(parsed["orientation"] == orientation, "orientation construction")
        output = _analytic_output(grid, parsed)
        seen.add(_signature(grid))
        return {"input": grid, "output": output}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    held = str(rng.choice(ORIENTATIONS))
    available = [o for o in ORIENTATIONS if o != held]
    training = [str(rng.choice([o for o in available if o in ("UL", "LR")])),
                str(rng.choice([o for o in available if o in ("UR", "LL")]))]
    training += [str(rng.choice(available)) for _ in range(num_train-2)]
    rng.shuffle(training)
    testing = [held] + [str(rng.choice(ORIENTATIONS)) for _ in range(num_test-1)]
    kinds = list(rng.permutation(("full", "leg")))
    kinds += [str(rng.choice(("full", "leg", "free"))) for _ in range(num_train+num_test-2)]
    puzzle, seen, previous = {"train": [], "test": []}, set(), None
    for i, orientation in enumerate(training+testing):
        roles = tuple(map(int, rng.choice(10, 2, replace=False)))
        if i == 1:
            # The geometric cue, not a literal color, determines the paint role.
            roles = (previous[1], previous[0])
        previous = roles
        pair = _make_pair(rng, kinds[i], orientation, roles, seen)
        puzzle["train" if i < num_train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, orientations = set(), {"train": set(), "test": set()}
    training_features, training_roles = Counter(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "shape preservation")
            sig = _signature(grid)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _analyze(grid)
            _require(2 <= len(info["boxes"]) <= 5, "generated material count")
            orientations[split].add(info["orientation"])
            masks = _triangle_masks(grid.shape, info)
            stats = [_box_stats(b, masks) for b in info["boxes"]]
            _require(any(s["partial"] and s["diagonal"] and s["outside"] and s["protected_perimeter"] for s in stats),
                     "every scene needs a diagonal clipping witness and a protected material perimeter")
            _require(np.array_equal(output, _analytic_output(grid, info)), "coordinate clipping rule")
            _require(np.array_equal(output, solve(grid)), "intersection erosion disagrees")
            # Explicit counterfactuals check that neither parent's role is cosmetic.
            material = np.zeros(grid.shape, dtype=bool)
            material_interior = np.zeros(grid.shape, dtype=bool)
            for r0, c0, r1, c1, _ in info["boxes"]:
                material[r0:r1+1, c0:c1+1] = True
                material_interior[r0+1:r1, c0+1:c1] = True
            for name, wrong_mask in (("omit A", material_interior),
                                     ("omit B", material & masks[1]),
                                     ("omit virtual rim", material_interior & masks[0])):
                alternative = grid.copy()
                alternative[wrong_mask] = info["cue"]
                _require(not np.array_equal(output, alternative), f"no witness against: {name}")
            changed = grid != output
            _require(bool(changed.any()), "nonidentity")
            _require(bool(np.all(output[changed] == info["cue"])), "use visible cue as paint")
            _require(bool(np.all(output[grid == info["background"]] == info["background"])), "background preservation")
            _require(bool(np.all(output[grid == info["cue"]] == info["cue"])), "original marker preservation")
            for r0, c0, r1, c1, color in info["boxes"]:
                _require(bool(np.all(output[[r0, r1], c0:c1+1] == color)), "horizontal rectangle perimeter")
                _require(bool(np.all(output[r0:r1+1, [c0, c1]] == color)), "vertical rectangle perimeter")
            if split == "train":
                training_roles.append((info["background"], info["cue"]))
                training_features["full"] += any(s["full"] for s in stats)
                training_features["leg"] += any(s["changed"] and s["leg"] and s["outside"] for s in stats)
    _require(training_features["full"] and training_features["leg"], "training must expose whole and leg-clipped windows")
    _require(len(set(training_roles)) >= 2, "training cue roles must vary")
    _require(orientations["train"] & {"UL", "LR"} and orientations["train"] & {"UR", "LL"}, "demonstrate both diagonal slopes")
    _require(orientations["test"] - orientations["train"], "held-out test rotation")


def _fingerprint(puzzle):
    return tuple(_signature(np.asarray(pair[k])) for s in ("train", "test")
                 for pair in puzzle[s] for k in ("input", "output"))


def self_test():
    """Bounded 42-seed semantic, diversity, equivariance, and interface checks.

    Returns descriptive measurements as well as asserting meaningful coverage.
    This is generator-owned verification, not an independent ARC solver review.
    """
    shapes, layouts, palettes, orientations, sides, counts, extents = (set() for _ in range(7))
    observations = Counter()
    fingerprints = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                info = _analyze(grid)
                shapes.add(grid.shape)
                layouts.add((grid.shape, info["bounds"], tuple(b[:4] for b in info["boxes"])))
                palettes.add((info["background"], info["cue"], tuple(b[4] for b in info["boxes"])))
                orientations.add(info["orientation"])
                sides.add(info["side"])
                counts.add(len(info["boxes"]))
                stats = [_box_stats(b, _triangle_masks(grid.shape, info)) for b in info["boxes"]]
                for feature in ("full", "partial", "inactive", "diagonal", "leg"):
                    observations[feature] += sum(bool(s[feature]) for s in stats)
                observations["repeated_color"] += len({b[4] for b in info["boxes"]}) < len(info["boxes"])
                observations["adjacent"] += any(_adjacent(a, b) for i, a in enumerate(info["boxes"]) for b in info["boxes"][i+1:])
                for r0, c0, r1, c1, _ in info["boxes"]:
                    extents.add((r1-r0+1, c1-c0+1))
                    observations["canvas_edge"] += r0 == 0 or c0 == 0 or r1 == grid.shape[0]-1 or c1 == grid.shape[1]-1
                    observations["thin"] += r1-r0 == 2 or c1-c0 == 2
                # The actual transformation must commute with every right-angle rotation.
                for k in (1, 2, 3):
                    _require(np.array_equal(solve(np.rot90(grid, k)), np.rot90(output, k)), "rotation equivariance")
                _require(np.array_equal(solve(grid.T), output.T), "reflection equivariance")
                # Any bijective recoloring changes roles consistently, including paint.
                permutation = np.array((7, 2, 9, 1, 5, 0, 8, 4, 6, 3))
                _require(np.array_equal(solve(permutation[grid]), permutation[output]), "color equivariance")
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 45 and len(layouts) >= 180, "geometry diversity")
    _require(len(palettes) >= 100 and orientations == set(ORIENTATIONS), "role diversity")
    _require(len(sides) >= 15 and counts == {2, 3, 4, 5} and len(extents) >= 60, "scale/count/aspect diversity")
    _require(all(observations[k] > 0 for k in ("full", "partial", "inactive", "diagonal", "leg", "repeated_color", "adjacent", "canvas_edge", "thin")), "missing compatible regimes")
    _require(_fingerprint(generate_puzzle(1234)) == _fingerprint(generate_puzzle(1234)), "determinism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(901+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                layouts=len(layouts), palettes=len(palettes), square_spans=sorted(sides),
                rectangle_counts=sorted(counts), rectangle_extents=len(extents),
                observations=dict(observations), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
