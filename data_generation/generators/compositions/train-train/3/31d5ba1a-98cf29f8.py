"""Bridge-guided rectangles leave the parity of every unit-position stamp.

The solver uses only the displayed scene.  Generation uses an integer coverage
formula, while solve explicitly toggles the individual translated rectangles.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _solid_box(mask):
    points = np.argwhere(mask)
    if not len(points):
        return None
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0) + 1)
    if int(mask.sum()) != (r1 - r0) * (c1 - c0):
        return None
    return r0, r1, c0, c1


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        component = np.zeros_like(mask)
        rr, cc = zip(*cells)
        component[rr, cc] = True
        result.append(component)
    return result


def _body_and_guide(component, hub):
    # Every cell of a side-2-or-larger solid rectangle belongs to a full 2x2.
    # No cell of a one-wide straight guide does. This recovers even thin bodies.
    square = (component[:-1, :-1] & component[1:, :-1]
              & component[:-1, 1:] & component[1:, 1:])
    core = np.zeros_like(component)
    core[:-1, :-1] |= square
    core[1:, :-1] |= square
    core[:-1, 1:] |= square
    core[1:, 1:] |= square
    body = _solid_box(core)
    if body is None or min(body[1] - body[0], body[3] - body[2]) < 2:
        raise ValueError("a mover needs a solid rectangular body with sides >=2")
    tail = np.argwhere(component & ~core)
    if not len(tail):
        raise ValueError("a mover needs at least one visible guide cell")
    r0, r1, c0, c1 = body
    hr0, hr1, hc0, hc1 = hub
    candidates = []
    if np.all(tail[:, 0] == tail[0, 0]):
        row = int(tail[0, 0])
        cols = np.sort(tail[:, 1])
        lo, hi = int(cols[0]), int(cols[-1]) + 1
        if hi - lo == len(tail) and r0 <= row < r1 and hr0 <= row < hr1:
            if lo == c1 and hi == hc0:
                candidates.append((0, 1))
            if hi == c0 and lo == hc1:
                candidates.append((0, -1))
    if np.all(tail[:, 1] == tail[0, 1]):
        col = int(tail[0, 1])
        rows = np.sort(tail[:, 0])
        lo, hi = int(rows[0]), int(rows[-1]) + 1
        if hi - lo == len(tail) and c0 <= col < c1 and hc0 <= col < hc1:
            if lo == r1 and hi == hr0:
                candidates.append((1, 0))
            if hi == r0 and lo == hr1:
                candidates.append((-1, 0))
    if len(candidates) != 1:
        raise ValueError("guide must uniquely connect the body to a facing hub edge")
    dr, dc = candidates[0]
    gap = len(tail)
    final = (r0 + gap * dr, r1 + gap * dr,
             c0 + gap * dc, c1 + gap * dc)
    fr0, fr1, fc0, fc1 = final
    height, width = component.shape
    if not (1 <= fr0 < fr1 < height and 1 <= fc0 < fc1 < width):
        raise ValueError("destination must preserve the background border")
    if max(fr0, hr0) < min(fr1, hr1) and max(fc0, hc0) < min(fc1, hc1):
        raise ValueError("a stamp cannot overlap the hub")
    return {"body": body, "direction": (dr, dc), "gap": gap, "final": final}


def _parse(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
        raise ValueError("the input needs a uniform background border")
    colors = set(map(int, np.unique(grid))) - {background}
    if len(colors) != 2:
        raise ValueError("expected background, hub color, and ink color")
    hubs = []
    for color in sorted(colors):
        box = _solid_box(grid == color)
        if box is not None and min(box[1] - box[0], box[3] - box[2]) >= 2:
            hubs.append((color, box))
    if len(hubs) != 1:
        raise ValueError("exactly one non-background color must form a solid hub")
    hub_color, hub = hubs[0]
    ink = next(iter(colors - {hub_color}))
    moves = [_body_and_guide(comp, hub) for comp in _components(grid == ink)]
    return {"shape": grid.shape, "background": background, "hub_color": hub_color,
            "ink": ink, "hub": hub, "moves": moves}


def _paint(info, mask):
    result = np.full(info["shape"], info["background"], dtype=np.int8)
    r0, r1, c0, c1 = info["hub"]
    result[r0:r1, c0:c1] = info["hub_color"]
    result[mask] = info["ink"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """XOR all body footprints at positions 0 through guide length, inclusive."""
    info = _parse(input_grid)
    parity = np.zeros(info["shape"], dtype=bool)
    for move in info["moves"]:
        r0, r1, c0, c1 = move["body"]
        dr, dc = move["direction"]
        for step in range(move["gap"] + 1):
            rr, cc = step * dr, step * dc
            parity[r0 + rr:r1 + rr, c0 + cc:c1 + cc] ^= True
    return _paint(info, parity)


def _coverage(info, move):
    """Count covering positions algebraically, without iterating stamp positions."""
    rows, cols = np.indices(info["shape"])
    r0, r1, c0, c1 = move["body"]
    dr, dc = move["direction"]
    if dc:
        local = cols - c0 if dc == 1 else c1 - 1 - cols
        depth = c1 - c0
        cross = (r0 <= rows) & (rows < r1)
    else:
        local = rows - r0 if dr == 1 else r1 - 1 - rows
        depth = r1 - r0
        cross = (c0 <= cols) & (cols < c1)
    lower = np.maximum(0, local - depth + 1)
    upper = np.minimum(move["gap"], local)
    return np.maximum(0, upper - lower + 1) * cross


def _coverages(info):
    return [_coverage(info, move) for move in info["moves"]]


def _integer(value, label, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {label}")
    return int(value)


def _rand(rng, low, high):
    return int(rng.integers(low, high + 1))


def _canvas(rng, hub, bodies, guides, palette):
    rmin = min([hub[0]] + [box[0] for box in bodies])
    rmax = max([hub[1]] + [box[1] for box in bodies])
    cmin = min([hub[2]] + [box[2] for box in bodies])
    cmax = max([hub[3]] + [box[3] for box in bodies])

    def padding(extent):
        extra = _rand(rng, 0, min(6, 28 - extent))
        before = _rand(rng, 1, extra + 1)
        return before, extra + 2 - before

    top, bottom = padding(rmax - rmin)
    left, right = padding(cmax - cmin)
    rr, cc = top - rmin, left - cmin
    bg, hub_color, ink = palette
    grid = np.full((rmax - rmin + top + bottom,
                    cmax - cmin + left + right), bg, dtype=np.int8)

    def draw(box, color):
        r0, r1, c0, c1 = box
        grid[r0 + rr:r1 + rr, c0 + cc:c1 + cc] = color

    draw(hub, hub_color)
    for box in bodies + guides:
        draw(box, ink)
    return grid


def _scene(rng, number, palette, lesson=None):
    sh, sw = _rand(rng, 2, 7), _rand(rng, 2, 7)
    hub = (0, sh, 0, sw)
    if number == 1:
        height = _rand(rng, 2, 9)
        row = _rand(rng, 1 - height, sh - 1)
        depth = _rand(rng, 5, 8) if lesson == "path" else _rand(rng, 2, 8)
        gap = 2 if lesson == "path" else _rand(rng, 1, 8)
        attach = _rand(rng, max(0, row), min(sh, row + height) - 1)
        bodies = [(row, row + height, -gap - depth, -gap)]
        guides = [(attach, attach + 1, -gap, 0)]
        grid = _canvas(rng, hub, bodies, guides, palette)
        if lesson != "path" and _rand(rng, 0, 1):
            grid = grid.T
        return grid.copy()

    # Both movers face the SAME hub. Their swept footprints cross outside its
    # upper-left corner, while their original bodies and guides remain separate.
    low = 2 if lesson == "corner" else 1
    over_h, over_v = _rand(rng, low, 4), _rand(rng, low, 4)
    inside_h, inside_v = _rand(rng, 1, sh), _rand(rng, 1, sw)
    if lesson == "corner":
        # An even number of positions complements the odd-position first lesson.
        gap_h = int(rng.choice([g for g in (3, 5) if g > over_v]))
        depth_h = _rand(rng, gap_h + 2, 8)
    else:
        gap_h, depth_h = _rand(rng, over_v + 1, 8), _rand(rng, 2, 8)
    gap_v, depth_v = _rand(rng, over_h + 1, 8), _rand(rng, 2, 8)
    bodies = [(-over_h, inside_h, -gap_h - depth_h, -gap_h),
              (-gap_v - depth_v, -gap_v, -over_v, inside_v)]
    ah, av = _rand(rng, 0, inside_h - 1), _rand(rng, 0, inside_v - 1)
    guides = [(ah, ah + 1, -gap_h, 0), (-gap_v, 0, av, av + 1)]
    return _canvas(rng, hub, bodies, guides, palette)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    stable_palette = bool(_rand(rng, 0, 2) == 0)
    rotation = _rand(rng, 0, 3)
    reflection = _rand(rng, 0, 2)  # reverse rows, columns, or both at test
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("path" if index == 0 else "corner" if index == 1 else None
                      ) if split == "train" else None
            number = (1 if lesson == "path" else 2 if lesson == "corner"
                      or (split == "test" and index == 0)
                      else int(rng.choice([1, 2], p=[.4, .6])))
            for _ in range(100):
                palette = base_palette if stable_palette else tuple(
                    map(int, rng.choice(10, size=3, replace=False)))
                grid = _scene(rng, number, palette, lesson)
                if split == "test":
                    if reflection in (0, 2):
                        grid = np.flip(grid, axis=0)
                    if reflection in (1, 2):
                        grid = np.flip(grid, axis=1)
                grid = np.rot90(grid, rotation).copy()
                key = _key(grid)
                if key in seen:
                    continue
                seen.add(key)
                info = _parse(grid)
                counts = sum(_coverages(info), np.zeros(grid.shape, dtype=int))
                output = _paint(info, (counts % 2) == 1)
                puzzle[split].append({"input": grid, "output": output})
                break
            else:
                raise RuntimeError("could not construct a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablations(info, local):
    union = np.logical_or.reduce([count > 0 for count in local])
    independent = np.logical_or.reduce([(count % 2) == 1 for count in local])
    endpoints = np.zeros(info["shape"], dtype=bool)
    static = np.zeros_like(endpoints)
    docked = np.zeros_like(endpoints)
    for move in info["moves"]:
        r0, r1, c0, c1 = move["body"]
        static[r0:r1, c0:c1] ^= True
        for box in (move["body"], move["final"]):
            a, b, c, d = box
            endpoints[a:b, c:d] ^= True
        a, b, c, d = move["final"]
        docked[a:b, c:d] = True
    return {"no_xor": union, "no_motion": static,
            "dock_only": docked, "endpoints_only": endpoints,
            "independent_sweeps": independent}


def validate_generated_puzzle(puzzle):
    """Check visible geometry, parity, interaction, and the demonstrated curriculum."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split schema")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, train_dirs, test_dirs = set(), set(), set()
    witnesses = {"no_xor": False, "no_motion": False, "dock_only": False,
                 "endpoints_only": False, "independent_sweeps": False}
    train_counts, train_multi, test_multi = set(), False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            before = inp.copy()
            info = _parse(inp)
            local = _coverages(info)
            counts = sum(local, np.zeros(inp.shape, dtype=int))
            parity = (counts % 2) == 1
            expected = _paint(info, parity)
            _require(np.array_equal(out, expected), "output violates accumulated parity")
            _require(np.array_equal(solve(inp), expected), "toggle solver disagrees with coverage formula")
            _require(np.array_equal(inp, before), "solver mutated its input")
            _require(not np.array_equal(inp, out), "unchanged scene")
            _require(np.array_equal(inp == info["hub_color"], out == info["hub_color"]),
                     "stationary hub changed")
            _require(np.all(out[counts == 0] == np.where(
                inp[counts == 0] == info["hub_color"], info["hub_color"], info["background"])),
                "material appeared outside the sweeps")
            _require(np.all(counts[0] == 0) and np.all(counts[-1] == 0)
                     and np.all(counts[:, 0] == 0) and np.all(counts[:, -1] == 0),
                     "stamp crossed the boundary")
            _require(not np.any(counts[inp == info["hub_color"]]), "stamp crossed the hub")
            (train_dirs if split == "train" else test_dirs).update(
                move["direction"] for move in info["moves"])
            if split == "train":
                train_counts.update(map(int, np.unique(counts)))
                train_multi |= len(info["moves"]) > 1
                for name, mask in _ablations(info, local).items():
                    witnesses[name] |= not np.array_equal(mask, parity)
            else:
                test_multi |= len(info["moves"]) > 1
    _require({0, 1, 2, 3}.issubset(train_counts), "training misses odd/even coverage evidence")
    _require(all(witnesses.values()), "training does not distinguish an intended mechanism")
    _require(not test_multi or train_multi, "multiple movers not demonstrated")
    _require(any(dr for dr, dc in train_dirs) and any(dc for dr, dc in train_dirs),
             "both travel axes must be demonstrated")
    _require(bool(test_dirs - train_dirs), "missing direction generalization at test")


def self_test():
    """Bounded 42-seed variation audit plus geometry/palette equivariance checks."""
    shapes, normalized, palettes, bodies, gaps, directions = (set() for _ in range(6))
    backgrounds, numbers, relations, intersections, signatures = (set() for _ in range(5))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                shapes.add(inp.shape)
                palette = (info["background"], info["hub_color"], info["ink"])
                palettes.add(palette)
                backgrounds.add(info["background"])
                role_grid = np.zeros(inp.shape, dtype=np.int8)
                role_grid[inp == info["hub_color"]] = 1
                role_grid[inp == info["ink"]] = 2
                normalized.add(_key(role_grid))
                numbers.add(len(info["moves"]))
                for move in info["moves"]:
                    r0, r1, c0, c1 = move["body"]
                    bodies.add((r1 - r0, c1 - c0))
                    gaps.add(move["gap"])
                    directions.add(move["direction"])
                    depth = c1 - c0 if move["direction"][1] else r1 - r0
                    relations.add(int(np.sign(depth - (move["gap"] + 1))))
                local = _coverages(info)
                if len(local) == 2:
                    intersections.add(int(np.count_nonzero((local[0] % 2) & (local[1] % 2))))
                # These tests depend on visible inputs, never generation metadata.
                k = (seed % 3) + 1
                _require(np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k)),
                         "rotation equivariance failed")
                _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)),
                         "reflection equivariance failed")
                permutation = np.roll(np.arange(10), seed % 9 + 1)
                _require(np.array_equal(solve(permutation[inp]), permutation[out]),
                         "palette equivariance failed")
    _require(len(signatures) == 42, "seed diversity failed")
    _require(len(shapes) >= 30 and len(normalized) >= 180, "geometry variation too small")
    _require(len(palettes) >= 30 and len(backgrounds) >= 7, "palette variation too small")
    _require(len(bodies) >= 30 and set(range(1, 9)).issubset(gaps), "shape/gap variation missing")
    _require(directions == {(0, 1), (0, -1), (1, 0), (-1, 0)}, "direction missing")
    _require(numbers == {1, 2} and relations == {-1, 0, 1}, "trajectory regimes missing")
    _require(len(intersections) >= 3 and min(intersections) > 0, "corner interference missing")
    # Hand-built cases exercise the stated rule beyond the sampler's ranges.
    witness = np.zeros((5, 11), dtype=np.int8)
    witness[1:4, 8:10] = 1
    witness[1:3, 1:6] = 2
    witness[1, 6:8] = 2
    _require(solve(witness)[2, 1:8].tolist() == [2, 0, 2, 2, 2, 0, 2],
             "known three-stamp cross-section failed")

    thin = np.zeros((8, 8), dtype=np.int8)
    thin[1:3, 3:6] = 1
    thin[4:6, 2:4] = 2
    thin[3, 3] = 2  # vertical, one-cell guide and a 2x2 body

    large = np.zeros((30, 30), dtype=np.int8)
    large[3:6, 25:28] = 1
    large[2:14, 2:13] = 2
    large[4, 13:25] = 2  # depth 11, height 12, guide length 12

    four = np.zeros((30, 30), dtype=np.int8)
    four[13:17, 13:17] = 1
    for box in ((11, 16, 1, 4), (1, 4, 11, 16),
                (15, 20, 25, 29), (25, 29, 15, 20),
                (14, 15, 4, 13), (4, 13, 14, 15),
                (16, 17, 17, 25), (17, 25, 16, 17)):
        a, b, c, d = box
        four[a:b, c:d] = 2
    _require(len(_parse(four)["moves"]) == 4, "additional movers not supported")
    for fixture in (witness, thin, large, four):
        info = _parse(fixture)
        counts = sum(_coverages(info), np.zeros(fixture.shape, dtype=int))
        answer = _paint(info, (counts % 2) == 1)
        _require(np.array_equal(solve(fixture), answer), "extended geometry failure")
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(fixture, k)), np.rot90(answer, k)),
                     "extended direction failure")
    for a, b in ((2, 1), (10, 4), (7, 3)):
        p, q = generate_puzzle(103, a, b), generate_puzzle(103, a, b)
        _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                     for k in ("input", "output")), "determinism failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "color_normalized_inputs": len(normalized),
            "role_palettes": len(palettes), "body_shapes": len(bodies),
            "guide_lengths": sorted(gaps), "directions": sorted(directions),
            "corner_cancellation_sizes": sorted(intersections),
            "extended_fixtures": 4,
            "checks": "coverage formula, ablations, rotation, reflection, recoloring, counts, determinism, hand fixtures"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
