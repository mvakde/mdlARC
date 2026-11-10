"""Magnetic cargo in a corner-seeded periodic sieve.

All coordinates used internally exclude the two rails.  Sampling constructs
the latent sieve first; solve recovers it solely from the visible corner seed.
"""

import json
import numpy as np


CORNERS = ("TL", "TR", "BL", "BR")


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        cells.add(p)
                        stack.append(p)
        result.append(cells)
    return result


def _four_connected(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _normal(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _lattice(motif, n, w, corner):
    ph, pw = motif.shape
    r0 = 0 if corner[0] == "T" else n - ph
    c0 = 0 if corner[1] == "L" else w - pw
    rr = (np.arange(n) - r0) % ph
    cc = (np.arange(w) - c0) % pw
    return motif[rr[:, None], cc[None, :]], r0, c0


def _parse(input_grid):
    grid = _grid(input_grid)
    horizontal = (grid.shape[0] >= 3 and np.all(grid[0] == grid[0, 0])
                  and np.all(grid[-1] == grid[-1, 0])
                  and grid[0, 0] != grid[-1, 0])
    vertical = (grid.shape[1] >= 3 and np.all(grid[:, 0] == grid[0, 0])
                and np.all(grid[:, -1] == grid[0, -1])
                and grid[0, 0] != grid[0, -1])
    if horizontal == vertical:
        raise ValueError("expected exactly one opposite-rail axis")
    a = grid if horizontal else grid.T
    interior = a[1:-1]
    rails = int(a[0, 0]), int(a[-1, 0])
    other = [int(x) for x in np.unique(a) if x not in rails]
    if len(other) != 2:
        raise ValueError("expected background, sieve and two rail colors")
    counts = [int(np.count_nonzero(interior == c)) for c in other]
    if counts[0] == counts[1]:
        raise ValueError("ambiguous background")
    background = other[int(np.argmax(counts))]
    wall_color = next(c for c in other if c != background)
    visible = interior == wall_color
    rr, cc = np.where(visible)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    n, w = interior.shape
    if ((r0 == 0) == (r1 == n) or (c0 == 0) == (c1 == w)):
        raise ValueError("sieve seed must touch exactly one corner")
    corner = ("T" if r0 == 0 else "B") + ("L" if c0 == 0 else "R")
    seed_block = interior[r0:r1, c0:c1]
    if not set(map(int, np.unique(seed_block))) <= {background, wall_color}:
        raise ValueError("cargo may not lie inside the seed rectangle")
    motif = seed_block == wall_color
    wall, _, _ = _lattice(motif, n, w, corner)
    objects = [(role, cells) for role, color in enumerate(rails)
               for cells in _components(interior == color)]
    if any(wall[r, c] for _, cells in objects for r, c in cells):
        raise ValueError("cargo overlaps the extrapolated sieve")
    return {"a": a, "horizontal": horizontal, "background": background,
            "wall_color": wall_color, "rails": rails, "corner": corner,
            "motif": motif, "visible": visible, "wall": wall,
            "objects": objects, "origin": (r0, c0)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the sieve, then slide each original rigid component to contact."""
    info = _parse(input_grid)
    a, wall = info["a"], info["wall"]
    n, w = wall.shape
    out = np.full(a.shape, info["background"], dtype=int)
    out[0], out[-1] = info["rails"][0], info["rails"][1]
    out[1:-1][wall] = info["wall_color"]
    occupied = set()
    for role, original in info["objects"]:
        step = -1 if role == 0 else 1
        cells = set(original)
        while True:
            next_cells = {(r + step, c) for r, c in cells}
            if any(r < 0 or r >= n or wall[r, c] for r, c in next_cells):
                break
            cells = next_cells
        if cells & occupied:
            raise ValueError("cargo destinations conflict")
        occupied.update(cells)
        for r, c in cells:
            out[r + 1, c] = info["rails"][role]
    return out if info["horizontal"] else out.T.copy()


def _clearance(cells, role, wall):
    """Ray calculation used by construction/checking, unlike solve's steps."""
    n, _ = wall.shape
    best = n
    for r, c in cells:
        if role == 0:
            obstacles = np.flatnonzero(wall[:r, c])
            distance = r - int(obstacles[-1]) - 1 if len(obstacles) else r
        else:
            obstacles = np.flatnonzero(wall[r + 1:, c])
            distance = int(obstacles[0]) if len(obstacles) else n - 1 - r
        best = min(best, distance)
    return int(best)


def _moved(cells, role, distance):
    dr = -distance if role == 0 else distance
    return {(r + dr, c) for r, c in cells}


def _sweep(cells, role, distance):
    return set().union(*(_moved(cells, role, d) for d in range(distance + 1)))


def _jump_distance(cells, role, wall):
    """Tempting wrong rule: ignore intermediate collisions and jump to a hole."""
    n = wall.shape[0]
    maximum = min(r for r, c in cells) if role == 0 else n - 1 - max(r for r, c in cells)
    for d in range(maximum, -1, -1):
        if all(not wall[r, c] for r, c in _moved(cells, role, d)):
            return d
    raise AssertionError("initial position was not free")


def _primitive(mask):
    h, w = mask.shape
    return not any(np.array_equal(mask, np.tile(mask[:p], (h // p, 1)))
                   for p in range(1, h) if h % p == 0) and not any(
        np.array_equal(mask, np.tile(mask[:, :p], (1, w // p)))
        for p in range(1, w) if w % p == 0)


def _motif(rng, ph, pw):
    for _ in range(300):
        mask = rng.random((ph, pw)) < float(rng.uniform(.18, .48))
        # A through-corridor and a sparse post column guarantee both kinds
        # of motion are possible. The exact motif is still freely composed.
        gap = int(rng.integers(1, pw - 1))
        mask[:, gap] = False
        if pw >= 5 and rng.random() < .45:
            mask[:, min(pw - 2, gap + 1)] = False
        mask[:, 0] = False
        mask[int(rng.integers(ph)), 0] = True
        mask[0, -1] = mask[-1, -1] = True
        if _primitive(mask):
            return mask
    raise RuntimeError("could not construct a primitive sieve")


def _shape(rng, kind, max_h=5, max_w=5):
    if kind == "pass":
        return {(r, 0) for r in range(int(rng.integers(1, max_h + 1)))}
    if kind == "diagonal":
        size = int(rng.integers(2, min(4, max_h, max_w) + 1))
        cells = {(r, r) for r in range(size)}
    else:
        h, w = int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_w + 1))
        mode = int(rng.integers(7))
        if mode == 0:
            cells = {(0, 0)}
        elif mode == 1:
            cells = {(r, 0) for r in range(h)}
        elif mode == 2:
            cells = {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)}
        elif mode == 3:
            cells = {(r, w // 2) for r in range(h)} | {(h // 2, c) for c in range(w)}
        elif mode == 4:
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
        elif mode == 5:
            cells = {(r, c) for r in range(h) for c in range(w)}
        else:
            cells = {(0, 0)}
            for _ in range(int(rng.integers(2, 11))):
                r, c = _pick(rng, sorted(cells))
                cells.add((r + int(rng.integers(-1, 2)), c + int(rng.integers(-1, 2))))
        if rng.random() < .5:
            cells = {(c, r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return set(_normal(cells))


def _near(first, second):
    return any(abs(r - s) <= 1 and abs(c - d) <= 1
               for r, c in first for s, d in second)


def _make_pair(rng, period, corner, horizontal, colors, split, first, pass_role):
    ph, pw = period
    for _ in range(90):
        repeats = 3 if first else int(rng.integers(2, 4))
        if split == "test":
            repeats = int(rng.integers(4, min(6, 28 // ph) + 1))
        n = repeats * ph + int(rng.integers(min(ph, 29 - repeats * ph)))
        lateral = int(rng.integers(2 if not first else 3, min(5, 30 // pw) + 1))
        w = lateral * pw + int(rng.integers(min(pw, 31 - lateral * pw)))
        motif = _motif(rng, ph, pw)
        wall, r0, c0 = _lattice(motif, n, w, corner)
        visible = np.zeros((n, w), dtype=bool)
        visible[r0:r0 + ph, c0:c0 + pw] = motif
        # The entire demonstration rectangle, including its blanks, is kept
        # clear so its tight accent box unambiguously defines the prototype.
        seed_rectangle = {(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)}
        count = int(rng.integers(3 if split == "train" else 4,
                                 7 if split == "train" else 9))
        placed, sweeps = [], set()
        slots = [(pass_role, "pass"), (1 - pass_role, "hidden"),
                 (pass_role, "diagonal" if first else "different")]
        slots += [(int(rng.integers(2)), "any") for _ in range(count - 3)]
        for role, kind in slots:
            accepted = False
            for attempt in range(1400):
                shape = _shape(rng, kind, min(5, n - 1), min(5, w))
                sh = 1 + max(r for r, c in shape)
                sw = 1 + max(c for r, c in shape)
                if sh > n or sw > w:
                    continue
                br, bc = int(rng.integers(n - sh + 1)), int(rng.integers(w - sw + 1))
                cells = {(br + r, bc + c) for r, c in shape}
                if cells & seed_rectangle or any(wall[r, c] for r, c in cells):
                    continue
                if any(role == old_role and _near(cells, old)
                       for old_role, old, _, _ in placed):
                    continue
                d = _clearance(cells, role, wall)
                maximum = min(r for r, c in cells) if role == 0 else n - 1 - max(r for r, c in cells)
                opposite = n - 1 - max(r for r, c in cells) if role == 0 else min(r for r, c in cells)
                if kind == "pass" and (d != maximum or d < 2 or maximum <= opposite):
                    continue
                if kind == "hidden" and (d < 1 or d == _clearance(cells, role, visible)
                                          or d == _jump_distance(cells, role, wall)):
                    continue
                if kind in ("diagonal", "different"):
                    if d == placed[0][2] or d >= maximum:
                        continue
                    if kind == "diagonal" and (d < 1 or len({
                            _clearance({cell}, role, wall) for cell in cells}) < 2):
                        continue
                swept = _sweep(cells, role, d)
                if swept & sweeps:
                    continue
                placed.append((role, cells, d, swept))
                sweeps.update(swept)
                accepted = True
                break
            if not accepted:
                break
        if len(placed) != count:
            continue
        background, wall_color, upper, lower = colors
        inp = np.full((n + 2, w), background, dtype=np.int8)
        out = inp.copy()
        inp[0] = out[0] = upper
        inp[-1] = out[-1] = lower
        inp[1:-1][visible] = wall_color
        out[1:-1][wall] = wall_color
        for role, cells, d, _ in placed:
            for r, c in cells:
                inp[r + 1, c] = (upper, lower)[role]
            for r, c in _moved(cells, role, d):
                out[r + 1, c] = (upper, lower)[role]
        other_counts = [np.count_nonzero(inp[1:-1] == c) for c in (background, wall_color)]
        if other_counts[0] <= other_counts[1]:
            continue
        if not horizontal:
            inp, out = inp.T.copy(), out.T.copy()
        return {"input": inp, "output": out}
    raise RuntimeError("could not pack noninteracting cargo into this sieve")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    recolor = bool(rng.random() < .4)
    mode = int(rng.integers(3))  # horizontal, vertical, or visibly mixed
    periods = [(h, w) for h in range(3, 6) for w in range(3, 7)]
    pool_size = min(num_train, int(rng.integers(2, 5)))
    period_pool = [periods[int(i)] for i in rng.permutation(len(periods))[:pool_size]]
    anchors = [CORNERS[int(i)] for i in rng.permutation(4)]
    orientations = [True, False] if mode == 2 else [mode == 0]
    result, used, taught = {"train": [], "test": []}, set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                period = period_pool[i] if i < pool_size else _pick(rng, period_pool)
                corner = anchors[i % 4]
                horizontal = orientations[i % len(orientations)]
                taught.append((period, corner, horizontal))
            else:
                period = _pick(rng, [x[0] for x in taught])
                corner = _pick(rng, [x[1] for x in taught])
                horizontal = _pick(rng, [x[2] for x in taught])
            colors = tuple(map(int, rng.choice(10, 4, replace=False))) if recolor else palette
            if not recolor and rng.random() < .4:
                colors = colors[:2] + (colors[3], colors[2])
            for _ in range(20):
                pair = _make_pair(rng, period, corner, horizontal, colors, split,
                                  split == "train" and i == 0, i % 2)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain a distinct input")
    validate_generated_puzzle(result)
    return result


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    assert inp.shape == out.shape
    info = _parse(inp)
    a, wall = info["a"], info["wall"]
    b = out if info["horizontal"] else out.T
    n, w = wall.shape
    ph, pw = info["motif"].shape
    assert 3 <= ph <= 5 and 3 <= pw <= 6
    assert n >= 2 * ph and w >= 2 * pw
    assert info["motif"].any() and not info["motif"].all()
    assert _primitive(info["motif"])
    assert np.any(~info["motif"].any(axis=0))
    assert np.any(info["motif"].any(axis=0) & ~info["motif"].all(axis=0))
    assert 3 <= len(info["objects"]) <= 8
    assert {role for role, cells in info["objects"]} == {0, 1}
    assert np.all(b[0] == info["rails"][0]) and np.all(b[-1] == info["rails"][1])
    assert np.array_equal(b[1:-1] == info["wall_color"], wall)
    expected = np.full_like(a, info["background"])
    expected[0], expected[-1] = info["rails"]
    expected[1:-1][wall] = info["wall_color"]
    swept_union = set()
    objects = []
    for role, cells in info["objects"]:
        d = _clearance(cells, role, wall)
        destination = _moved(cells, role, d)
        swept = _sweep(cells, role, d)
        assert not swept & swept_union
        swept_union.update(swept)
        assert all(not wall[r, c] for r, c in swept)
        maximum = min(r for r, c in cells) if role == 0 else n - 1 - max(r for r, c in cells)
        for r, c in destination:
            expected[r + 1, c] = info["rails"][role]
        objects.append({"role": role, "cells": cells, "shape": _normal(cells),
                        "distance": d, "blocked": d < maximum,
                        "wrong_nearest": maximum > (n - 1 - max(r for r, c in cells)
                                                     if role == 0 else min(r for r, c in cells)),
                        "hidden_matters": d != _clearance(cells, role, info["visible"]),
                        "jump_differs": d != _jump_distance(cells, role, wall),
                        "diagonal_matters": not _four_connected(cells) and len({
                            _clearance({cell}, role, wall) for cell in cells}) > 1})
    assert np.array_equal(expected, b)
    assert np.array_equal(solve(inp), out)
    assert any(o["distance"] > 0 for o in objects)
    info["objects"] = objects
    info["repeats"] = n // ph
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            info = _analyze(pair)
            # Counterexamples are guaranteed, rather than hoped for in a batch.
            assert any(o["hidden_matters"] and o["jump_differs"] and o["distance"] > 0
                       for o in info["objects"])
            assert any(not o["blocked"] and o["distance"] >= 2 and o["wrong_nearest"]
                       for o in info["objects"])
            assert any(len(set(o["distance"] for o in info["objects"] if o["role"] == role)) >= 2
                       for role in (0, 1))
            assert info["repeats"] in ((2, 3) if split == "train" else (4, 5, 6))
            analyses[split].append(info)
    train, test = analyses["train"], analyses["test"]
    assert any(o["diagonal_matters"] for a in train for o in a["objects"])
    for key in ("corner", "horizontal"):
        assert {a[key] for a in test} <= {a[key] for a in train}
    assert {a["motif"].shape for a in test} <= {a["motif"].shape for a in train}


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(p[s], q[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    analyses = [_analyze(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    objects = [o for a in analyses for o in a["objects"]]
    assert len({_key(p["test"][0]["input"]) for p in puzzles}) == 42
    assert len({a["a"].shape for a in analyses}) >= 70
    assert {a["corner"] for a in analyses} == set(CORNERS)
    assert {a["horizontal"] for a in analyses} == {True, False}
    assert {a["background"] for a in analyses} == set(range(10))
    assert len({a["motif"].shape for a in analyses}) >= 10
    assert len({(a["motif"].shape, a["motif"].tobytes()) for a in analyses}) >= 80
    assert len({o["shape"] for o in objects}) >= 60
    assert {len(a["objects"]) for a in analyses} == set(range(3, 9))
    assert any(o["distance"] == 0 for o in objects)
    assert any(o["distance"] >= 12 for o in objects)
    assert any(a["wall"].shape[0] % a["motif"].shape[0] == 0 and
               a["wall"].shape[1] % a["motif"].shape[1] == 0 for a in analyses)
    assert any(a["wall"].shape[0] % a["motif"].shape[0] != 0 and
               a["wall"].shape[1] % a["motif"].shape[1] != 0 for a in analyses)
    assert _equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    # Exercise equivariance on novel inputs, without using construction state.
    permutation = np.random.default_rng(8401).permutation(10)
    for puzzle in puzzles[:6]:
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
                assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
                assert np.array_equal(solve(permutation[a]), permutation[b])
    # All legal count combinations, including the minimum two demonstrations.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(analyses),
            "canonical_shapes": len({a["a"].shape for a in analyses}),
            "motifs": len({(a["motif"].shape, a["motif"].tobytes()) for a in analyses}),
            "cargo_shapes": len({o["shape"] for o in objects})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
