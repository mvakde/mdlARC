"""A phase-normalized porous shuttle slides through a field of fixed posts.

Only standard library and NumPy are used. Solving reads the displayed lane,
source swatch and dock; generation metadata is never used by solve().
"""
from __future__ import annotations

from itertools import permutations
import hashlib
import json

import numpy as np


_ORDERS = tuple((transpose, rows, cols) for transpose in (False, True)
                for rows in permutations(range(3)) for cols in permutations(range(3)))
_DIRECT = (False, (1, 2, 0), (1, 2, 0))
_HOST_SEEDS = (0,654961082,273020838,240006298,217340107,255461656,44812960,217157512,887664726,527478417,133561820,257070929,76959525,927058557,350685401,821011157,410672793,205063579,902322086,11178368,775730261,773298495,222883514,755454878,974192737,207686003,602546652,510055686,215922979,701282904,386360145,905237014,179359204,789993461,729937843,394541206,367074519,215448406,279883034,894464391,91391865,4346518)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _full_box(mask):
    return (mask.shape == (3, 3) and 1 < int(mask.sum()) < 9
            and all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())))


_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> i) & 1 for i in range(9)],
                                           dtype=bool).reshape(3, 3)))


def _ordered(mask, order=_DIRECT):
    transpose, rows, cols = order
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


def _tile(mask, k, order=_DIRECT):
    tile = _ordered(mask, order)
    return tile[np.arange(k)[:, None] % 3, np.arange(k)[None, :] % 3]


def _bbox(mask):
    cells = np.argwhere(mask)
    _require(len(cells) > 0, "empty object")
    top, left = map(int, cells.min(axis=0))
    bottom, right = map(int, cells.max(axis=0) + 1)
    return top, left, bottom, right


def _position(info, step):
    r, c = info["start"]
    dr, dc = info["direction"]
    return r + step * dr, c + step * dc


def _material(shape, info, step, footprint):
    result = np.zeros(shape, dtype=bool)
    r, c = _position(info, step)
    k = info["k"]
    result[r:r+k, c:c+k] = footprint
    return result


def _parse(value):
    a = _grid(value)
    wall = int(a[0, 0])
    _require(bool(np.all(a[0] == wall) and np.all(a[-1] == wall)
                  and np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall)),
             "outer frame must be uniform")
    roles = set(map(int, np.unique(a))) - {wall}
    _require(len(roles) == 3, "need wall, floor, swatch and dock")
    top, left, bottom, right = _bbox(a != wall)
    h, w = bottom-top, right-left
    k, length = min(h, w), max(h, w)
    _require(3 <= k <= 9 and length >= 2*k+6, "lane has invalid proportions")
    vertical = h > w
    first = (top, left)
    last = (bottom-k, right-k)
    goals = []
    for color in roles:
        r, c, b, d = _bbox(a == color)
        if (b-r, d-c) == (k, k) and int((a == color).sum()) == k*k and (r, c) in (first, last):
            goals.append((color, (r, c)))
    _require(len(goals) == 1, "dock is not one terminal solid square")
    goal, target = goals[0]
    start = last if target == first else first
    sign = 1 if target == last else -1
    direction = (sign, 0) if vertical else (0, sign)
    swatches = []
    for color in roles - {goal}:
        r, c, b, d = _bbox(a == color)
        if (b-r, d-c) == (3, 3) and 1 < int((a == color).sum()) < 9:
            if start[0] <= r <= b <= start[0]+k and start[1] <= c <= d <= start[1]+k:
                swatches.append((color, a[r:b, c:d] == color))
    _require(len(swatches) == 1, "source must contain one full-box non-solid swatch")
    accent, mask = swatches[0]
    _require(_full_box(mask), "swatch does not span its bounding patch")
    floor = next(iter(roles - {goal, accent}))
    # The band has exactly a square source bay and a solid terminal dock.
    sr, sc = start
    _require(not np.any(a[sr:sr+k, sc:sc+k] == wall), "source bay contains posts")
    middle_floor = a == floor
    middle_floor[sr:sr+k, sc:sc+k] = False
    tr, tc = target
    middle_floor[tr:tr+k, tc:tc+k] = False
    _require(bool(middle_floor.any()), "middle floor is completely obscured")
    return {"wall": wall, "floor": floor, "accent": accent, "goal": goal,
            "mask": mask.copy(), "k": k, "length": length, "vertical": vertical,
            "start": start, "target": target, "direction": direction,
            "distance": length-k, "bounds": (top, left, bottom, right)}


def _travel(a, info, footprint):
    """Unit translations; holes are transparent, so only material hits walls."""
    for step in range(1, info["distance"]+1):
        r, c = _position(info, step)
        k = info["k"]
        if np.any((a[r:r+k, c:c+k] == info["wall"]) & footprint):
            return step-1
    return info["distance"]


def _render(a, info, step, footprint):
    out = a.copy()
    out[a == info["accent"]] = info["floor"]
    if step == info["distance"]:
        out[a == info["goal"]] = info["floor"]
    r, c = _position(info, step)
    k = info["k"]
    out[r:r+k, c:c+k][footprint] = info["accent"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the lane, tile the swatch, and stop at the first collision."""
    a = _grid(input_grid)
    info = _parse(a)
    footprint = _tile(info["mask"], info["k"])
    return _render(a, info, _travel(a, info, footprint), footprint)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _orientation_candidates(mask):
    correct = _ordered(mask)
    return frozenset(i for i, order in enumerate(_ORDERS) if np.array_equal(_ordered(mask, order), correct))


# A bounded immutable lookup avoids recomputing 72 motif symmetries in every
# generation/validation call. It is never consulted by input-only solve().
_AGREEMENTS = {mask.tobytes(): _orientation_candidates(mask) for mask in _MASKS}


def _agreement(mask):
    return _AGREEMENTS[mask.tobytes()]


def _front(footprint, vertical, sign):
    lines = footprint.T if vertical else footprint
    return tuple((int(np.flatnonzero(row)[-1 if sign > 0 else 0])
                  if row.any() else -1) for row in lines)


def _hit_times(shape, info, footprint):
    times = np.full(shape, info["distance"]+1, dtype=np.int16)
    for step in range(info["distance"]+1):
        ink = _material(shape, info, step, footprint)
        times[ink] = np.minimum(times[ink], step)
    return times


def _make_pair(rng, palette, k, vertical, sign, mask, blocked, used, witness=False):
    wall, floor, accent, goal = palette
    # At most 80 transient candidates, no saved retries or execution traces.
    for _ in range(80):
        length = int(rng.integers(2*k+6, 29))
        h, w = (length, k) if vertical else (k, length)
        pads = []
        for dim in (h, w):
            budget = min(6, 30-dim)
            total = int(rng.integers(2, budget+1))
            before = int(rng.integers(1, total))
            pads.append((before, total-before))
        top, left = pads[0][0], pads[1][0]
        a = np.full((h+sum(pads[0]), w+sum(pads[1])), wall, dtype=np.int8)
        a[top:top+h, left:left+w] = floor
        first, last = (top, left), (top+h-k, left+w-k)
        start, target = (first, last) if sign > 0 else (last, first)
        sr, sc = start
        tr, tc = target
        a[tr:tr+k, tc:tc+k] = goal
        mr, mc = int(rng.integers(k-2)), int(rng.integers(k-2))
        a[sr+mr:sr+mr+3, sc+mc:sc+mc+3][mask] = accent
        info = _parse(a)
        footprint = _tile(mask, k)
        middle = (a == floor)
        middle[sr:sr+k, sc:sc+k] = False
        # The dock was already excluded because it is not floor-colored.
        phase_times = None
        if blocked and witness:
            actual_times = _hit_times(a.shape, info, footprint)
            unshifted = _tile(mask, k, (False, (0,1,2), (0,1,2)))
            phase_times = _hit_times(a.shape, info, unshifted)
            critical = middle & (actual_times >= 2) & (actual_times <= length-2*k+1)
            critical &= actual_times != phase_times
            critical_cells = np.argwhere(critical)
            if not len(critical_cells):
                continue
            p = tuple(critical_cells[int(rng.integers(len(critical_cells)))])
            stop = int(actual_times[p])-1
        elif blocked:
            # Stopping square stays wholly before the dock, avoiding an
            # unexplained partial goal erasure on failure.
            stop = int(rng.integers(1, length-2*k+1))
        else:
            stop = info["distance"]
        swept = np.zeros(a.shape, dtype=bool)
        for step in range(stop+1):
            swept |= _material(a.shape, info, step, footprint)
        safe = np.argwhere(middle & ~swept)
        if witness and not blocked and len(safe) == 0:
            continue
        if blocked:
            collision = middle & ~swept & _material(a.shape, info, stop+1, footprint)
            candidates = np.argwhere(collision)
            if not len(candidates):
                continue
            if not witness:
                p = tuple(candidates[int(rng.integers(len(candidates)))])
            a[p] = wall
            if witness and phase_times[p] > stop+1:
                # Keep the unshifted counterfactual strictly later; extra
                # pegs cannot accidentally make its stop equal the answer.
                safe = np.argwhere(middle & ~swept & (phase_times > stop+1))
        # Floor must remain visible outside the two bays. Otherwise in a
        # tiny k=3 source its complement could look like another swatch.
        available_floor = np.argwhere(middle & (a == floor))
        reserve = tuple(available_floor[int(rng.integers(len(available_floor)))])
        safe = np.array([q for q in safe if tuple(q) != reserve], dtype=int).reshape(-1, 2)
        # Posts can form disconnected comb teeth, clusters, or sparse pegs.
        # Some are inside the final holes; some lie farther downstream.
        if len(safe):
            number = int(rng.integers(1 if witness else 0, min(18, len(safe))+1))
            for i in rng.choice(len(safe), size=number, replace=False):
                a[tuple(safe[int(i)])] = wall
        actual = _travel(a, info, footprint)
        if actual != stop:
            continue
        if witness and blocked:
            unshifted = mask[np.arange(k)[:, None] % 3, np.arange(k)[None, :] % 3]
            if _travel(a, info, unshifted) == actual:
                continue
        if _key(a) in used:
            continue
        used.add(_key(a))
        return {"input": a, "output": _render(a, info, actual, footprint)}
    raise AssertionError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    # Regimes alter visible sampling only, never the rule or color meanings.
    regime = int(rng.integers(3))
    train_range, test_range = (((4, 6), (7, 9)), ((6, 8), (3, 5)), ((4, 7), (4, 7)))[regime]
    axis = bool(rng.integers(2))
    # A successful hollow shuttle MUST pass actual posts, defeating a solid
    # envelope or "first post" shortcut. An empty internal source row/column
    # becomes an empty phase-zero material line after cyclic normalization.
    success_masks = [mask for mask in _MASKS if not (mask[:, 1].any() if axis else mask[1].any())]
    first = success_masks[int(rng.integers(len(success_masks)))]
    first_agree = _agreement(first)
    eligible = [mask for mask in _MASKS if len(first_agree & _agreement(mask)) == 1]
    second = eligible[int(rng.integers(len(eligible)))]
    specs = [(first, axis, False, True), (second, not axis, True, True)]
    specs += [(_MASKS[int(rng.integers(len(_MASKS)))], bool(rng.integers(2)), bool(rng.integers(2)), False)
              for _ in range(n-2)]
    # General success may be unobstructed; the first witness always has pegs.
    specs = [specs[int(i)] for i in rng.permutation(n)]
    puzzle = {"train": [], "test": []}
    used = set()
    train_masks = set()
    for split, count in (("train", n), ("test", m)):
        lo, hi = train_range if split == "train" else test_range
        for i in range(count):
            if split == "train":
                mask, vertical, blocked, witness = specs[i]
            else:
                choices = [mask for mask in _MASKS if mask.tobytes() not in train_masks]
                mask = choices[int(rng.integers(len(choices)))]
                vertical, blocked, witness = bool(rng.integers(2)), bool(rng.integers(2)), False
            # k>=4 in training exposes at least a partial second period.
            k = int(rng.integers(lo, hi+1))
            if witness:
                # Each essential demonstration contains two complete tile
                # periods on both axes, including when only two are asked for.
                k = int(rng.integers(max(6, lo), hi+1))
            sign = 1 if rng.integers(2) else -1
            if witness and blocked:
                candidates = [candidate for candidate in eligible
                              if _front(_tile(candidate, k), vertical, sign) !=
                              _front(_tile(candidate, k, (False, (0,1,2), (0,1,2))), vertical, sign)]
                _require(bool(candidates), "no phase-sensitive orientation witness")
                mask = candidates[int(rng.integers(len(candidates)))]
            puzzle[split].append(_make_pair(rng, palette, k, vertical,
                                           sign, mask, blocked, used, witness))
            if split == "train":
                train_masks.add(mask.tobytes())
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "shape changed")
    info = _parse(a)
    footprint = _tile(info["mask"], info["k"])
    step = _travel(a, info, footprint)
    _require(np.array_equal(b, _render(a, info, step, footprint)), "incorrect material movement")
    _require(np.array_equal(b == info["wall"], a == info["wall"]), "posts/walls changed")
    _require(int((b == info["accent"]).sum()) == int(footprint.sum()), "material count changed")
    _require(step > 0, "generated shuttle did not move")
    success = step == info["distance"]
    if success:
        _require(not np.any(b == info["goal"]), "reached dock did not disappear")
    else:
        _require(np.array_equal(a == info["goal"], b == info["goal"]), "blocked dock changed")
        _require(step <= info["length"]-2*info["k"], "blocked envelope overlaps dock")
        _require(np.any(_material(a.shape, info, step+1, footprint) & (a == info["wall"])),
                 "stop lacks a next-step collision")
    solid = _travel(a, info, np.ones_like(footprint))
    identity = _tile(info["mask"], info["k"], (False, (0, 1, 2), (0, 1, 2)))
    return dict(info, step=step, success=success, solid_step=solid,
                identity_step=_travel(a, info, identity), shape=a.shape)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, palettes, infos = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            palettes.add(tuple(info[role] for role in ("wall", "floor", "accent", "goal")))
            infos[split].append(info)
    _require(len(palettes) == 1, "within-puzzle palette changed")
    train = infos["train"]
    _require(any(i["success"] and i["solid_step"] < i["step"] for i in train),
             "training needs a porous success through real posts")
    _require(any(not i["success"] and i["identity_step"] != i["step"] for i in train),
             "training needs a phase-sensitive collision")
    orders = set(range(len(_ORDERS)))
    for info in train:
        orders &= _agreement(info["mask"])
    _require(len(orders) == 1 and _ORDERS[next(iter(orders))] == _DIRECT,
             "training motifs do not distinguish cyclic orientation")
    train_masks = {i["mask"].tobytes() for i in train}
    _require(all(i["mask"].tobytes() not in train_masks for i in infos["test"]), "test motif is familiar")
    sizes = {i["k"] for i in train}
    _require(all(4 <= k <= 8 for k in sizes), "training scale outside sampler")
    # Other legal family inputs need not obey these sampling shifts; solve
    # supports every k=3..9 and every full-box mask independently of split.
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded checks: 42 puzzles, all counts, negative cases and exact fixtures.

    No files, logs, saved puzzle corpus, or execution traces are created.
    The returned descriptive report is optional; the host may ignore it.
    """
    # Literal hand-computed horizontal fixture: M=101/000/111 becomes
    # T=000/111/011. Its phase-zero row passes a peg at (1,7); the peg at
    # (2,10) first hits the rightmost material at step 7, so it stops at 6.
    a = np.full((5, 22), 5, dtype=np.int8)
    a[1:4, 1:21] = 0
    a[1:4, 18:21] = 8
    a[1:4, 1:4] = np.array([[2,0,2],[0,0,0],[2,2,2]])
    a[1,7] = a[2,10] = 5
    expected = a.copy()
    expected[expected == 2] = 0
    # With a 3-wide panel the first collision is at origin column 8,
    # hence last-safe origin 7 (translation 6), not at the swatch boundary.
    expected[2,7:10] = 2
    expected[3,8:10] = 2
    _require(np.array_equal(solve(a), expected), "literal collision fixture")
    # Remove the blocking peg: the empty line passes its remaining post and
    # the dock becomes exactly the tiled material plus floor-colored holes.
    clear = a.copy()
    clear[2,10] = 0
    end = clear.copy()
    end[end == 2] = 0
    end[end == 8] = 0
    end[2,18:21] = 2
    end[3,19:21] = 2
    _require(np.array_equal(solve(clear), end), "literal porous arrival fixture")
    motifs, shapes, sizes, directions, densities, structures = set(), set(), set(), set(), set(), set()
    fingerprints, test_outcomes, palettes = set(), set(), set()
    larger = smaller = phase_changes = porous = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic seed")
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        larger += min(i["k"] for i in test) > max(i["k"] for i in train)
        smaller += max(i["k"] for i in test) < min(i["k"] for i in train)
        for split, infos in (("train", train), ("test", test)):
            for pair, info in zip(puzzle[split], infos):
                motifs.add(info["mask"].tobytes())
                shapes.add(info["shape"])
                sizes.add(info["k"])
                directions.add(info["direction"])
                densities.add(int(info["mask"].sum()))
                palettes.add(tuple(info[r] for r in ("wall", "floor", "accent", "goal")))
                phase_changes += info["identity_step"] != info["step"]
                porous += info["solid_step"] < info["step"]
                norm = np.zeros(info["shape"], dtype=np.int8)
                for code, role in enumerate(("wall", "floor", "accent", "goal")):
                    norm[np.asarray(pair["input"]) == info[role]] = code
                structures.add(_key(norm))
                if split == "test":
                    test_outcomes.add(info["success"])
    _require(len(fingerprints) == 42, "whole puzzle diversity collapsed")
    _require(len(motifs) >= 70 and len(structures) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 50 and sizes == set(range(3, 10)), "scale variation collapsed")
    _require(directions == {(-1,0),(1,0),(0,-1),(0,1)}, "directions collapsed")
    _require(densities == set(range(2, 9)), "motif density variation collapsed")
    _require(test_outcomes == {False, True} and larger >= 5 and smaller >= 5, "test shift coverage")
    _require(phase_changes >= 42 and porous >= 42 and len(palettes) >= 35, "mechanism coverage")
    legal = [(n, m) for n in range(2, 11) for m in range(1, 5) if n > m]
    for n, m in legal:
        p = generate_puzzle(8181, n, m)
        _require((len(p["train"]), len(p["test"])) == (n, m), "counts ignored")
    for n, m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(2.0,1),(True,1),(4,False),(None,1),(4,None)):
        try:
            generate_puzzle(3, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative checks use only one live mutated copy and leave no files.
    for mutation in ("wall", "material", "dock"):
        p = generate_puzzle(0)
        target_pair = next(pair for pair in p["train"] if np.any(pair["output"] == _parse(pair["input"])["goal"]))
        info = _parse(target_pair["input"])
        role = {"wall": "wall", "material": "accent", "dock": "goal"}[mutation]
        where = tuple(np.argwhere(target_pair["output"] == info[role])[0])
        target_pair["output"][where] = info["floor"]
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "motifs": len(motifs), "shapes": len(shapes),
            "sizes": sorted(sizes), "directions": len(directions), "densities": sorted(densities),
            "structures": len(structures), "palettes": len(palettes), "larger_test_puzzles": larger,
            "smaller_test_puzzles": smaller, "phase_sensitive_pairs": phase_changes,
            "porous_advantage_pairs": porous, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
