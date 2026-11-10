"""Enclosed cross-stitch receivers. Only Python's standard library and NumPy."""

from itertools import combinations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"illegal {name}: {value!r}")
    return int(value)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(np.all((grid >= 0) & (grid <= 9)), "ARC palette required")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _components(grid, background):
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if seen[r, c] or color == background:
                continue
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < height and 0 <= xx < width and not seen[yy, xx]
                            and grid[yy, xx] == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            result.append((color, cells))
    return result


def _analyze(input_grid):
    """Input-only structural decoding; no generation parameters are consulted."""
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    _require(all(np.all(edge == background)
                 for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])), "clear border")
    probes, probe_colors, objects = [], set(), []
    for color, cells in _components(grid, background):
        if len(cells) == 1:
            probes.extend(cells)
            probe_colors.add(color)
            continue
        rs, cs = zip(*cells)
        top, bottom, left, right = min(rs), max(rs) + 1, min(cs), max(cs) + 1
        box = grid[top:bottom, left:right]
        _require(min(box.shape) >= 2, "rectangular objects need two-dimensional extent")
        _require(np.all((box == color) | (box == background)), "foreign cells inside object")
        _require(all(np.all(edge == color)
                     for edge in (box[0], box[-1], box[:, 0], box[:, -1])), "closed rectangle")
        missing = np.argwhere(box == background)
        hole = None
        if len(missing):
            a, b = missing.min(axis=0)
            z, w = missing.max(axis=0) + 1
            _require(0 < a < z < box.shape[0] and 0 < b < w < box.shape[1], "internal cavity")
            _require(len(missing) == (z - a) * (w - b), "rectangular cavity")
            hole = (top + int(a), left + int(b), top + int(z), left + int(w))
        _require(len(cells) == box.size - len(missing), "one connected rectangular object")
        objects.append({"box": (top, left, bottom, right), "color": color, "hole": hole})
    _require(len(probes) >= 2 and len(probe_colors) == 1, "one singleton probe color")
    probe_color = next(iter(probe_colors))
    _require(2 <= len(objects) <= 5, "object count")
    _require(all(obj["color"] != probe_color for obj in objects), "probe/object palette collision")
    targets = [obj for obj in objects if obj["hole"] is not None]
    _require(len(targets) == 1, "unique cavity required")
    for index, first in enumerate(objects):
        a, b, z, w = first["box"]
        for second in objects[index + 1:]:
            c, d, v, u = second["box"]
            _require(z < c or v < a or w < d or u < b, "objects require a background gap")
    for r, c in probes:
        for obj in objects:
            a, b, z, w = obj["box"]
            _require(not (a - 1 <= r <= z and b - 1 <= c <= w), "probe too close to object")
    _require(all(max(abs(r - y), abs(c - x)) >= 2
                 for i, (r, c) in enumerate(probes) for y, x in probes[i + 1:]), "probe isolation")
    target = targets[0]
    top, left, bottom, right = target["hole"]
    rows = sorted({r for r, _ in probes if top <= r < bottom})
    cols = sorted({c for _, c in probes if left <= c < right})
    _require(rows and cols, "both axis supports required")
    _require(all(b - a >= 2 for values in (rows, cols) for a, b in zip(values, values[1:])),
             "adjacent supported coordinates conflict with accents")
    neutral = [(r, c) for r, c in probes if not top <= r < bottom and not left <= c < right]
    return {"background": background, "probe": probe_color, "probes": probes,
            "objects": objects, "target": target, "rows": rows, "cols": cols,
            "neutral": neutral}


def _render(grid, hole, rows, cols, probe_color, stitch_color):
    result = grid.copy()
    top, left, bottom, right = hole
    for r in rows:
        result[r, left:right] = probe_color
    for c in cols:
        result[top:bottom, c] = probe_color
    for r in rows:
        for c in cols:
            result[r, c] = stitch_color
            for dr in (-1, 1):
                for dc in (-1, 1):
                    if top <= r + dr < bottom and left <= c + dc < right:
                        result[r + dr, c + dc] = stitch_color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    a = _analyze(grid)
    return _render(grid, a["target"]["hole"], a["rows"], a["cols"],
                   a["probe"], a["target"]["color"])


def _axis(rng, size, count, edge=False, interior=False):
    candidates = [v for v in combinations(range(size), count)
                  if all(b - a >= 2 for a, b in zip(v, v[1:]))
                  and (not edge or 0 in v or size - 1 in v)
                  and (not interior or (count == 1 and edge)
                       or any(0 < a < size - 1 for a in v))]
    return candidates[int(rng.integers(len(candidates)))]


def _apart(box, old):
    a, b, z, w = box
    return all(z < c or v < a or w < d or u < b for c, d, v, u in old)


def _layout(rng, outer, matched, count):
    oh, ow = outer
    if matched:
        axes = [axis for axis, size in enumerate(outer) if 2 * size + 3 <= 30]
        axis = int(rng.choice(axes))
        gap = int(rng.integers(1, min(4, 28 - 2 * outer[axis]) + 1))
        minimum = 2 * outer[axis] + gap + 2
        shape = [int(rng.integers(max(18, oh + 4), 31)),
                 int(rng.integers(max(18, ow + 4), 31))]
        shape[axis] = int(rng.integers(max(18, minimum), 31))
        start = int(rng.integers(1, shape[axis] - 2 * outer[axis] - gap))
        positions = []
        for j in range(2):
            other = 1 - axis
            pos = [0, 0]
            pos[axis] = start + j * (outer[axis] + gap)
            pos[other] = int(rng.integers(1, shape[other] - outer[other]))
            positions.append((pos[0], pos[1], pos[0] + oh, pos[1] + ow))
        if rng.integers(2):
            positions.reverse()
        boxes = positions
    else:
        shape = [int(rng.integers(max(22, oh + 6), 31)),
                 int(rng.integers(max(22, ow + 6), 31))]
        r = int(rng.integers(1, shape[0] - oh))
        c = int(rng.integers(1, shape[1] - ow))
        boxes = [(r, c, r + oh, c + ow)]
    while len(boxes) < count:
        minimum = 4 if len(boxes) == 1 else 2
        choices = []
        for _ in range(12):
            height, width = (int(v) for v in rng.integers(minimum, 9, size=2))
            choices = [(r, c, r + height, c + width)
                       for r in range(1, shape[0] - height)
                       for c in range(1, shape[1] - width)
                       if _apart((r, c, r + height, c + width), boxes)]
            if choices:
                break
        if not choices:
            return None
        boxes.append(choices[int(rng.integers(len(choices)))])
    return tuple(shape), boxes


def _place_probes(rng, shape, boxes, hole, rows, cols, training_index):
    top, left, bottom, right = hole
    available = [(r, c) for r in range(1, shape[0] - 1)
                 for c in range(1, shape[1] - 1)
                 if all(not (a - 1 <= r <= z and b - 1 <= c <= w)
                        for a, b, z, w in boxes)
                 and (not top <= r < bottom or r in rows)
                 and (not left <= c < right or c in cols)]
    points = []

    def add(candidates):
        candidates = [p for p in candidates
                      if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2 for q in points)]
        if not candidates:
            return False
        points.append(candidates[int(rng.integers(len(candidates)))])
        return True

    for j, r in enumerate(rows):
        candidates = [p for p in available if p[0] == r]
        # The first two demonstrations collectively reveal all four probe sides.
        if training_index == 0:
            candidates = [p for p in candidates if p[1] < boxes[0][1]]
        elif training_index == 1:
            candidates = [p for p in candidates
                          if (p[1] >= boxes[0][3] if j == 0 else p[1] < boxes[0][1])]
        if not add(candidates):
            return None
    for j, c in enumerate(cols):
        candidates = [p for p in available if p[1] == c]
        if training_index == 0:
            candidates = [p for p in candidates
                          if (p[0] < boxes[0][0] if j == 0 else p[0] >= boxes[0][2])]
        if not add(candidates):
            return None
    # Redundant support: essential in demonstration 2, optional elsewhere.
    for _ in range(1 if training_index == 1 else int(rng.integers(0, 3))):
        if not add([p for p in available if p[0] in rows or p[1] in cols]):
            return None
    # The first solid foil has projections meeting inside it as well.
    a, b, z, w = boxes[1]
    if not add([p for p in available if a < p[0] < z - 1 and not b <= p[1] < w]):
        return None
    if not add([p for p in available if b < p[1] < w - 1 and not a <= p[0] < z]):
        return None
    # Truly irrelevant evidence: neither coordinate reaches the receiver.
    for _ in range(int(rng.integers(1, 3))):
        if not add([p for p in available if not top <= p[0] < bottom and not left <= p[1] < right]):
            return None
    return points


def _make_pair(rng, roles, split, index):
    background, probe, stitch = roles
    is_test = split == "test"
    for _ in range(240):
        if is_test:
            hh, hw = (int(v) for v in rng.integers(9, 14, size=2))
            nr, nc = (int(v) for v in rng.integers(2, 5, size=2))
            if nr == nc == 2:
                if rng.integers(2):
                    nr = 3
                else:
                    nc = 3
        else:
            hh, hw = (int(v) for v in rng.integers(6, 11, size=2))
            if index == 0:
                hh = int(rng.integers(5, 10))
                nr, nc = 1, 2
            elif index == 1:
                nr, nc = 2, 2
            else:
                nr, nc = (int(v) for v in rng.integers(1, 3, size=2))
        walls = [int(v) for v in rng.integers(1, 3, size=4)]
        oh, ow = hh + walls[0] + walls[2], hw + walls[1] + walls[3]
        matched = (not is_test and index < 2) or bool(rng.random() < .28)
        if matched and min(oh, ow) > 13:
            continue
        count = int(rng.integers(3, 6))
        layout = _layout(rng, (oh, ow), matched, count)
        if layout is None:
            continue
        shape, boxes = layout
        a, b, _, _ = boxes[0]
        hole = (a + walls[0], b + walls[1], a + walls[0] + hh, b + walls[1] + hw)
        if not is_test and index == 0:
            rows = (hole[0] + int(rng.integers(1, hh - 1)),)
        else:
            rows = tuple(hole[0] + r for r in _axis(rng, hh, nr,
                         edge=(not is_test and index == 1) or bool(rng.random() < .6), interior=True))
        cols = tuple(hole[1] + c for c in _axis(rng, hw, nc,
                     edge=(not is_test and index == 0) or bool(rng.random() < .6), interior=True))
        points = _place_probes(rng, shape, boxes, hole, rows, cols, index if not is_test else -1)
        if points is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        body_palette = [c for c in range(10) if c not in (background, probe)]
        for j, (r, c, z, w) in enumerate(boxes):
            if j == 0 or (j == 1 and not is_test and index == 1):
                color = stitch
            elif j == 1 and not is_test and index == 0:
                color = int(rng.choice([v for v in body_palette if v != stitch]))
            else:
                color = int(rng.choice(body_palette))
            grid[r:z, c:w] = color
        top, left, bottom, right = hole
        grid[top:bottom, left:right] = background
        for r, c in points:
            grid[r, c] = probe
        output = _render(grid, hole, rows, cols, probe, stitch)
        if not np.any(output[top:bottom, left:right] == background):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError(f"could not construct {split} example {index}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(10)]
    puzzle, seen = {"train": [], "test": []}, set()
    ordinal = 0
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            roles = tuple(palette[(ordinal + offset) % 10] for offset in (0, 3, 6))
            for _ in range(10):
                pair = _make_pair(rng, roles, split, index)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction failed")
            puzzle[split].append(pair)
            seen.add(key)
            ordinal += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    grid, out = _grid(pair["input"]), _grid(pair["output"])
    _require(grid.shape == out.shape, "canvas conservation")
    a = _analyze(grid)
    top, left, bottom, right = a["target"]["hole"]
    rows, cols = set(a["rows"]), set(a["cols"])
    _require(a["neutral"], "irrelevant probe evidence missing")
    # A cellwise oracle checks the renderer's loops without calling solve/_render.
    expected = grid.copy()
    plain = grid.copy()
    for r in range(top, bottom):
        for c in range(left, right):
            center = r in rows and c in cols
            diagonal = any(r + d in rows for d in (-1, 1)) and any(c + d in cols for d in (-1, 1))
            line = r in rows or c in cols
            expected[r, c] = (a["target"]["color"] if center or diagonal
                              else a["probe"] if line else a["background"])
            plain[r, c] = a["probe"] if line else a["background"]
    _require(np.array_equal(out, expected), "cross-stitch formula")
    _require(not np.array_equal(out, grid), "projection ablation must change the answer")
    _require(not np.array_equal(out, plain), "plain-lattice shortcut must differ")
    mask = np.ones(grid.shape, dtype=bool)
    mask[top:bottom, left:right] = False
    _require(np.array_equal(grid[mask], out[mask]), "changed cells outside receiver")
    _require(np.any(out[top:bottom, left:right] == a["background"]), "visible unpainted cavity")
    _require(all(out[r, c] == a["target"]["color"] for r in rows for c in cols), "all Cartesian centers")
    solids = [obj for obj in a["objects"] if obj["hole"] is None]
    decoys = []
    for obj in solids:
        y, x, z, w = obj["box"]
        supported_r = [r for r, _ in a["probes"] if y < r < z - 1]
        supported_c = [c for _, c in a["probes"] if x < c < w - 1]
        if supported_r and supported_c:
            # Ignoring enclosure and drawing the same projected lattice here
            # necessarily overwrites a body cell with the distinct probe color.
            _require(grid[supported_r[0], supported_c[0]] != a["probe"], "foil palette")
            _require(np.array_equal(grid[y:z, x:w], out[y:z, x:w]), "solid foil changed")
            decoys.append(obj)
    _require(decoys, "a solid must have its own paired projection support")
    a["decoys"] = decoys
    return a


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _check_pair(pair)
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve")
            analyses[split].append(a)
    for index in (0, 1):
        a = analyses["train"][index]
        y, x, z, w = a["target"]["box"]
        matching = [obj for obj in a["decoys"]
                    if obj["box"][2] - obj["box"][0] == z - y
                    and obj["box"][3] - obj["box"][1] == w - x]
        _require(matching, "same-size aligned solid foil in first two demonstrations")
        if index == 1:
            _require(any(obj["color"] == a["target"]["color"] for obj in matching), "same-color foil")
    first, second = analyses["train"][:2]
    _require((len(first["rows"]), len(first["cols"])) == (1, 2), "first demonstration")
    _require((len(second["rows"]), len(second["cols"])) == (2, 2), "Cartesian-product demonstration")
    sides = set()
    for a in (first, second):
        top, left, bottom, right = a["target"]["hole"]
        y, x, z, w = a["target"]["box"]
        for r, c in a["probes"]:
            if top <= r < bottom:
                sides.add("left" if c < x else "right")
            if left <= c < right:
                sides.add("top" if r < y else "bottom")
    _require(sides == {"left", "right", "top", "bottom"}, "all four probe sides demonstrated")
    _require(len(second["probes"]) - len(second["neutral"])
             > len(second["rows"]) + len(second["cols"]), "redundant support demonstration")
    t, l, b, r = first["target"]["hole"]
    _require(any(t < y < b - 1 and l < x < r - 1
                 for y in first["rows"] for x in first["cols"]), "complete interior motif")
    t, l, b, r = second["target"]["hole"]
    _require(any(y in (t, b - 1) for y in second["rows"]), "boundary clipping demonstrated")
    for a in analyses["train"]:
        _require(max(len(a["rows"]), len(a["cols"])) <= 2, "training coordinate range")
    for a in analyses["test"]:
        _require(2 <= min(len(a["rows"]), len(a["cols"]))
                 and 3 <= max(len(a["rows"]), len(a["cols"])) <= 4, "test aggregation shift")
    for role in ("background", "probe"):
        _require(first[role] != second[role], "pair-local color evidence")
    _require(first["target"]["color"] != second["target"]["color"], "receiver hue evidence")


def _signature(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded 42-seed audit plus metamorphic and count-contract checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42, "seed diversity")
    shapes, holes, walls, counts, layouts, palettes = set(), set(), set(), set(), set(), set()
    object_counts, probe_counts = set(), set()
    role_colors = {role: set() for role in ("background", "probe", "stitch")}
    clipped, overlapping, repeated, same_color, aspects = set(), 0, 0, 0, set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                a = _analyze(grid)
                top, left, bottom, right = a["target"]["hole"]
                y, x, z, w = a["target"]["box"]
                shapes.add(grid.shape)
                holes.add((bottom - top, right - left))
                walls.add((top - y, left - x, z - bottom, w - right))
                counts.add((len(a["rows"]), len(a["cols"])))
                object_counts.add(len(a["objects"]))
                probe_counts.add(len(a["probes"]))
                layouts.add((grid.shape, a["target"]["box"], a["target"]["hole"],
                             tuple(a["rows"]), tuple(a["cols"]), tuple(a["probes"])))
                palettes.add((a["background"], a["probe"], a["target"]["color"]))
                role_colors["background"].add(a["background"])
                role_colors["probe"].add(a["probe"])
                role_colors["stitch"].add(a["target"]["color"])
                aspects.add(np.sign((bottom - top) - (right - left)))
                repeated += len(a["probes"]) - len(a["neutral"]) > len(a["rows"]) + len(a["cols"])
                same_color += len({obj["color"] for obj in a["objects"]}) < len(a["objects"])
                accents = []
                for r in a["rows"]:
                    for c in a["cols"]:
                        cells = [(r + dr, c + dc) for dr in (-1, 1) for dc in (-1, 1)
                                 if top <= r + dr < bottom and left <= c + dc < right]
                        clipped.add(len(cells))
                        accents.extend(cells)
                overlapping += len(accents) > len(set(accents))
        # Metamorphic checks use one test from each seed, with no generation state.
        pair = puzzle["test"][0]
        grid, out = pair["input"], pair["output"]
        for k in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(grid, k)), np.rot90(out, k)), "rotation equivariance")
        _require(np.array_equal(solve(np.fliplr(grid)), np.fliplr(out)), "reflection equivariance")
        mapping = np.random.default_rng(10000 + len(shapes)).permutation(10)
        _require(np.array_equal(solve(mapping[grid]), mapping[out]), "palette equivariance")
        a = _analyze(grid)
        # Delete an irrelevant point: the cavity pattern stays unchanged.
        r, c = a["neutral"][0]
        edited, expected = grid.copy(), out.copy()
        edited[r, c] = expected[r, c] = a["background"]
        _require(np.array_equal(solve(edited), expected), "irrelevant-probe invariance")
        # Remove all probes supporting one row; its Cartesian centers disappear.
        edited = grid.copy()
        for r, c in a["probes"]:
            if r == a["rows"][0]:
                edited[r, c] = a["background"]
        changed = solve(edited)
        top, left, bottom, right = a["target"]["hole"]
        _require(not np.array_equal(changed[top:bottom, left:right], out[top:bottom, left:right]),
                 "row contribution is necessary")
        edited = grid.copy()
        for r, c in a["probes"]:
            if c == a["cols"][0]:
                edited[r, c] = a["background"]
        changed = solve(edited)
        _require(not np.array_equal(changed[top:bottom, left:right], out[top:bottom, left:right]),
                 "column contribution is necessary")
        # Add one valid irrelevant probe. It changes only its own exterior cell.
        candidates = [(r, c) for r in range(1, grid.shape[0] - 1)
                      for c in range(1, grid.shape[1] - 1)
                      if not top <= r < bottom and not left <= c < right
                      and np.all(grid[r - 1:r + 2, c - 1:c + 2] == a["background"])
                      and all(not (y <= r < z and x <= c < w)
                              for y, x, z, w in (obj["box"] for obj in a["objects"]))]
        _require(candidates, "room for a neutral addition diagnostic")
        r, c = candidates[0]
        edited, expected = grid.copy(), out.copy()
        edited[r, c] = expected[r, c] = a["probe"]
        _require(np.array_equal(solve(edited), expected), "irrelevant-probe addition invariance")
        # Demonstration 2 always has a removable redundant support. Removing
        # it preserves both coordinate sets and thus the complete interior.
        pair = puzzle["train"][1]
        grid, out = pair["input"], pair["output"]
        a = _analyze(grid)
        removed = False
        for r, c in a["probes"]:
            remaining = [p for p in a["probes"] if p != (r, c)]
            t, l, b, w = a["target"]["hole"]
            rs = sorted({y for y, _ in remaining if t <= y < b})
            cs = sorted({x for _, x in remaining if l <= x < w})
            if (r in a["rows"] or c in a["cols"]) and rs == a["rows"] and cs == a["cols"]:
                edited, expected = grid.copy(), out.copy()
                edited[r, c] = expected[r, c] = a["background"]
                _require(np.array_equal(solve(edited), expected), "redundant-probe removal invariance")
                removed = True
                break
        _require(removed, "removable redundant support diagnostic")
    _require(len(shapes) >= 45 and len(holes) >= 35 and len(walls) >= 12, "geometric diversity")
    _require(len(layouts) == 210 and len(palettes) >= 100, "structural/palette diversity")
    _require(len(counts) >= 10 and aspects == {-1, 0, 1}, "count/aspect diversity")
    _require(object_counts == {3, 4, 5} and len(probe_counts) >= 7, "object/probe count diversity")
    _require(clipped == {1, 2, 4} and overlapping >= 15 and repeated >= 42 and same_color >= 42,
             "family interaction coverage")
    _require(all(colors == set(range(10)) for colors in role_colors.values()), "role palette coverage")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + 10 * nt + ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_signature(generate_puzzle(591719129)) == _signature(generate_puzzle(591719129)),
             "determinism")
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes), "hole_shapes": len(holes),
            "wall_profiles": len(walls), "coordinate_count_pairs": len(counts),
            "layouts": len(layouts), "palettes": len(palettes),
            "object_counts": sorted(object_counts), "probe_counts": sorted(probe_counts),
            "overlapping_accent_pairs": overlapping, "redundant_probe_pairs": repeated,
            "same_color_object_pairs": same_color}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
