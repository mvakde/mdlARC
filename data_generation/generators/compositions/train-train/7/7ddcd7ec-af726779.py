"""Diagonal sources feeding a gap-centering cellular process.

Only NumPy and the standard library are used.  The input carries every role:
background, monochrome square/ray color, seed-line color, and flow direction.
"""

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or any(not 1 <= n <= 30 for n in grid.shape)
            or grid.dtype.kind not in "iu" or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return grid


def _signature(grid):
    return (grid.shape, grid.astype(np.int8).tobytes())


def _decode(input_grid):
    """Rotate to downward flow, checking the complete input grammar."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3 or int(max(counts)) * 2 <= grid.size:
        raise ValueError("input needs three colors and a majority background")
    background = int(colors[np.argmax(counts)])
    cores = []
    for color in colors:
        if color == background:
            continue
        m = grid == color
        matches = m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]
        cores.extend((int(color), int(r), int(c)) for r, c in np.argwhere(matches))
    if len(cores) != 1:
        raise ValueError("input must have exactly one monochrome nonbackground 2x2 core")
    ray_color = cores[0][0]
    seed_color = next(int(c) for c in colors if c not in (background, ray_color))
    if np.count_nonzero(grid == seed_color) < 3:
        raise ValueError("seed line must contain at least three cells")
    candidates = []
    for rotation in range(4):
        g = np.rot90(grid, rotation)
        seed_positions = np.argwhere(g == seed_color)
        if len(set(map(int, seed_positions[:, 0]))) != 1:
            continue
        seed_row = int(seed_positions[0, 0])
        m = g == ray_color
        core = np.argwhere(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:])[0]
        r, c = map(int, core)
        if seed_row < r + 3:
            continue
        expected = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        directions = []
        for side, column in ((-1, c - 1), (1, c + 2)):
            if 0 <= column < g.shape[1] and g[r + 2, column] == ray_color:
                directions.append(side)
                expected.add((r + 2, column))
        if not directions or expected != set(map(tuple, np.argwhere(m))):
            continue
        info = {"background": background, "ray_color": ray_color,
                "seed_color": seed_color, "core": (r, c),
                "seed_row": seed_row, "directions": tuple(directions)}
        rays = _rays(g.shape, info)
        if np.any(rays & (g == seed_color)):
            continue
        if seed_row >= g.shape[0] - 1:
            continue
        candidates.append((g, info, rotation))
    if len(candidates) != 1:
        raise ValueError("seed line and core must identify one flow direction")
    return candidates[0]


def _rays(shape, info):
    mask = np.zeros(shape, dtype=bool)
    r, c = info["core"]
    for side in info["directions"]:
        column = c - 1 if side == -1 else c + 2
        for row in range(r + 2, shape[0]):
            if not 0 <= column < shape[1]:
                break
            mask[row, column] = True
            column += side
    return mask


def _next(active, require_empty=True):
    result = np.zeros_like(active, dtype=bool)
    result[1:-1] = active[:-2] & active[2:]
    if require_empty:
        result[1:-1] &= ~active[1:-1]
    return result


def _render(g, info, *, feed_rays=True, draw_rays=True, births=True,
            require_empty=True):
    """Canonical simulation; optional switches are counterfactual audits only."""
    out = g.copy()
    rays = _rays(g.shape, info)
    if draw_rays:
        out[rays] = info["ray_color"]
    row0 = info["seed_row"]
    active = g[row0] == info["seed_color"]
    if feed_rays:
        active |= rays[row0]
    if births:
        for row in range(row0 + 1, len(g)):
            born = _next(active, require_empty)
            out[row, born] = (info["ray_color"] if (row - row0) % 2
                              else info["seed_color"])
            if draw_rays:
                out[row, rays[row]] = info["ray_color"]
            active = born | rays[row] if feed_rays else born
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible objects, colors and geometry in this input."""
    g, info, rotation = _decode(input_grid)
    return np.rot90(_render(g, info), -rotation).copy()


def _features(input_grid, output_grid=None):
    g, info, rotation = _decode(input_grid)
    actual = _render(g, info) if output_grid is None else np.rot90(_grid(output_grid), rotation)
    rays = _rays(g.shape, info)
    b = info["background"]
    start = info["seed_row"]
    unfed = _render(g, info, feed_rays=False)
    ray_only = _render(g, info, births=False)
    loose = _render(g, info, require_empty=False)
    changed = actual != g
    offray = changed & ~rays
    live_rows = np.flatnonzero(np.any(offray, axis=1))
    seed = g[start] == info["seed_color"]
    suppressed = (actual == b) & (unfed != b) & ~rays
    created = (actual != b) & (unfed == b) & ~rays
    ray_lengths, ray_endings = [], []
    for side in info["directions"]:
        c = info["core"][1]
        column = c - 1 if side == -1 else c + 2
        row = info["core"][0] + 2
        cells = []
        while 0 <= row < len(g) and 0 <= column < g.shape[1]:
            cells.append((row, column))
            row, column = row + 1, column + side
        ray_lengths.append(sum(r >= start for r, _ in cells))
        row_exit, column_exit = row >= len(g), not 0 <= column < g.shape[1]
        ray_endings.append("corner" if row_exit and column_exit else "row" if row_exit else "column")
    result = dict(info)
    result.update({"shape": input_grid.shape, "rotation": rotation,
                   "horizon": len(g) - start - 1,
                   "gap": start - info["core"][0] - 2,
                   "feedback_difference": int(np.count_nonzero((actual != unfed) & ~rays)),
                   "birth_difference": int(np.count_nonzero(actual != ray_only)),
                   "exclusion_difference": int(np.count_nonzero(actual != loose)),
                   "solid_triplet": bool(np.any(seed[:-2] & seed[1:-1] & seed[2:])),
                   "last_birth": int(max(live_rows) - start) if len(live_rows) else 0,
                   "phase_return": bool(np.any(offray & (actual == info["seed_color"]))),
                   "offray_cells": int(np.count_nonzero(offray)),
                   "seed_count": int(sum(seed)),
                   "feedback_creations": int(np.count_nonzero(created)),
                   "feedback_suppressions": int(np.count_nonzero(suppressed)),
                   "ray_lengths": tuple(ray_lengths), "ray_endings": tuple(ray_endings)})
    # Separate local replay: includes forced cells in every preceding frontier.
    assert actual.shape == g.shape
    assert np.array_equal(actual[:start + 1], _render(g, info, births=False)[:start + 1])
    previous = (g[start] == info["seed_color"]) | rays[start]
    for row in range(start + 1, len(g)):
        centers = np.zeros(g.shape[1], dtype=bool)
        for column in range(1, g.shape[1] - 1):
            centers[column] = previous[column - 1] and not previous[column] and previous[column + 1]
        expected = np.full(g.shape[1], b, dtype=actual.dtype)
        expected[centers] = info["ray_color"] if (row - start) % 2 else info["seed_color"]
        expected[rays[row]] = info["ray_color"]
        assert np.array_equal(actual[row], expected), "frontier recurrence or color parity violated"
        previous = actual[row] != b
    return result


def _make_mask(rng, width, kind):
    mask = np.zeros(width, dtype=bool)
    if kind == "random":
        mask = rng.random(width) < float(rng.uniform(0.22, 0.65))
    elif kind == "dense":
        length = int(rng.integers(6, width + 1))
        left = int(rng.integers(0, width - length + 1))
        mask[left:left + length] = True
        holes = rng.choice(np.arange(left + 1, left + length - 1),
                           int(rng.integers(1, max(2, length // 2))), replace=False)
        mask[holes] = False
    elif kind == "chain":
        depth = int(rng.integers(1, min(9, (width - 1) // 2) + 1))
        left = int(rng.integers(0, width - 2 * depth))
        mask[left:left + 2 * depth + 1:2] = True
        if rng.random() < 0.5:
            mask |= rng.random(width) < 0.12
    elif kind == "fragments":
        for _ in range(int(rng.integers(2, 5))):
            length = int(rng.integers(2, min(width, 10) + 1))
            left = int(rng.integers(0, width - length + 1))
            mask[left:left + length:int(rng.integers(1, 3))] = True
    elif kind == "sparse":
        mask[rng.choice(width, int(rng.integers(3, min(7, width) + 1)), replace=False)] = True
    else:
        raise ValueError("unknown pattern kind")
    return mask


def _make_pair(rng, palette, rotation, directions, horizon, long_source,
               diagnostic, used):
    background, ray_color, seed_color = palette
    for _ in range(300):
        gap = int(rng.integers(1, 6 if long_source else 5))
        feeder = int(rng.choice(directions))
        clearance = {side: (9 if long_source and side == feeder else 2)
                     for side in directions}
        low = gap + 1 + clearance[-1] if -1 in directions else 0
        extra_right = gap + 3 + clearance[1] if 1 in directions else 2
        minimum_width = max(14, low + extra_right)
        width = int(rng.integers(minimum_width, 31))
        core_col = int(rng.integers(low, width - extra_right + 1))
        core_row = int(rng.integers(0, 5))
        seed_row = core_row + 2 + gap
        height = seed_row + horizon + 1
        assert height <= 30
        info = {"background": background, "ray_color": ray_color,
                "seed_color": seed_color, "core": (core_row, core_col),
                "seed_row": seed_row, "directions": directions}
        rays = _rays((height, width), info)
        source_column = core_col - 1 - gap if feeder == -1 else core_col + 2 + gap
        gap_column = source_column - feeder
        partner_column = source_column - 2 * feeder
        if not 0 <= partner_column < width or rays[seed_row, partner_column]:
            continue
        kind = str(rng.choice(("random", "dense", "chain", "fragments", "sparse")))
        seed = _make_mask(rng, width, kind)
        # A one-hole bracket containing a ray is the feedback witness.
        seed[rays[seed_row]] = False
        seed[gap_column] = False
        seed[partner_column] = True
        if diagnostic:
            forbidden = set(np.flatnonzero(rays[seed_row])) | {gap_column}
            options = [c for c in range(width - 2)
                       if not forbidden.intersection(range(c, c + 3))]
            left = int(rng.choice(options))
            seed[left:left + 3] = True
        if sum(seed) < 3:
            options = np.flatnonzero(~seed & ~rays[seed_row])
            options = options[options != gap_column]
            seed[rng.choice(options, 3 - int(sum(seed)), replace=False)] = True
        g = np.full((height, width), background, dtype=np.int8)
        g[core_row:core_row + 2, core_col:core_col + 2] = ray_color
        for side in directions:
            g[core_row + 2, core_col - 1 if side == -1 else core_col + 2] = ray_color
        g[seed_row, seed] = seed_color
        canonical_output = _render(g, info)
        inp = np.rot90(g, rotation).copy()
        out = np.rot90(canonical_output, rotation).copy()
        if _signature(inp) in used:
            continue
        feature = _features(inp, out)
        if (feature["feedback_difference"] == 0 or not feature["phase_return"]
                or feature["birth_difference"] == 0):
            continue
        if diagnostic and feature["exclusion_difference"] == 0:
            continue
        if long_source and feature["last_birth"] < 9:
            continue
        used.add(_signature(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a distinct, informative pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    local_palette = bool(rng.integers(0, 2))
    rotation = int(rng.integers(0, 4))
    single = (int(rng.choice((-1, 1))),)
    plans = [((-1, 1), True), (single, True)]
    while len(plans) < num_train:
        directions = ((-1, 1), (-1,), (1,))[int(rng.integers(3))]
        plans.append((directions, bool(rng.integers(2))))
    rng.shuffle(plans)
    used = set()
    puzzle = {"train": [], "test": []}
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            palette = tuple(map(int, rng.choice(10, 3, replace=False))) if local_palette else base_palette
            if split == "train":
                directions, diagnostic = plans[index]
                horizon = int(rng.integers(4, 9))
            else:
                directions = ((-1, 1), (-1,), (1,))[int(rng.integers(3))]
                diagnostic = bool(rng.integers(2))
                horizon = int(rng.integers(10, 17))
            pair = _make_pair(rng, palette, rotation, directions, horizon,
                              split == "test", diagnostic, used)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            assert np.array_equal(solve(inp), out)
            g, info, rotation = _decode(inp)
            canonical_out = np.rot90(out, rotation)
            assert np.array_equal(canonical_out[g != info["background"]], g[g != info["background"]])
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            f = _features(inp, out)
            assert f["birth_difference"] > 0 and f["feedback_difference"] > 0
            assert f["phase_return"]
            assert min(f["ray_lengths"]) >= 3, "each ray must cross the seed and continue twice"
            features[split].append(f)
    training, testing = features["train"], features["test"]
    assert len({f["rotation"] for f in training + testing}) == 1
    assert set().union(*(set(f["directions"]) for f in training)) == {-1, 1}
    assert any(len(f["directions"]) == 1 for f in training)
    assert any(len(f["directions"]) == 2 for f in training)
    assert sum(bool(f["solid_triplet"] and f["exclusion_difference"]) for f in training) >= 2
    assert all(4 <= f["horizon"] <= 8 for f in training)
    assert all(10 <= f["horizon"] <= 16 and f["last_birth"] >= 9 for f in testing)
    assert all(max(f["ray_lengths"]) >= 10 for f in testing)


def self_test():
    """Bounded generator-owned audit; not an independent ARC evaluation."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, normalized, geometries, palettes, rotations, seed_counts = set(), set(), set(), set(), set(), set()
    exclusions = feedback = suppressions = 0
    endings = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp, out)
                shapes.add(inp.shape)
                roles = {f["background"]: 0, f["ray_color"]: 1, f["seed_color"]: 2}
                normalized.add((inp.shape, tuple(roles[int(c)] for c in inp.flat)))
                geometries.add((f["core"], f["gap"], f["directions"], f["horizon"]))
                palettes.add((f["background"], f["ray_color"], f["seed_color"]))
                rotations.add(f["rotation"])
                seed_counts.add(f["seed_count"])
                exclusions += f["exclusion_difference"] > 0
                feedback += f["feedback_difference"] > 0
                suppressions += f["feedback_suppressions"] > 0
                endings.update(f["ray_endings"])
    assert len(shapes) >= 60 and len(normalized) >= 150
    assert len(geometries) >= 120 and len(palettes) >= 30
    assert rotations == {0, 1, 2, 3} and len(seed_counts) >= 12
    assert feedback == 210 and exclusions >= 84
    assert suppressions > 0 and endings == {"row", "column", "corner"}
    # Transformation invariance checks use the input alone, including a
    # color permutation that changes all literal role colors.
    probe = puzzles[0]["train"][0]
    original = probe["input"].copy()
    permutation = np.array([3, 7, 0, 9, 2, 8, 4, 1, 6, 5], dtype=np.int8)
    for rotation in range(4):
        inp = np.rot90(probe["input"], rotation)
        out = np.rot90(probe["output"], rotation)
        assert np.array_equal(solve(inp), out)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
    assert np.array_equal(probe["input"], original)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(90210 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    again = generate_puzzle(0)
    assert all(np.array_equal(x[key], y[key])
               for split in ("train", "test")
               for x, y in zip(puzzles[0][split], again[split])
               for key in ("input", "output"))
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "normalized_inputs": len(normalized), "geometries": len(geometries),
            "palettes": len(palettes), "seed_counts": len(seed_counts),
            "feedback_witnesses": feedback, "exclusion_witnesses": exclusions,
            "suppression_witnesses": suppressions, "ray_endings": sorted(endings)}


if __name__ == "__main__":
    print(self_test())
