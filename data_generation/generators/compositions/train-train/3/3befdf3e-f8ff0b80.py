"""Collective growth: global color populations arbitrate local arm crossings.

No parent generator is imported. The solver reads only the supplied grid.
"""

from collections import Counter
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _masks(shape, obj):
    r, c, k = obj[:3]
    s = k + 2
    footprint = np.zeros(shape, dtype=bool)
    footprint[r:r+s, c:c+s] = True
    arms = np.zeros(shape, dtype=bool)
    arms[r-k:r, c:c+s] = True
    arms[r+s:r+s+k, c:c+s] = True
    arms[r:r+s, c-k:c] = True
    arms[r:r+s, c+s:c+s+k] = True
    return footprint, arms


def _inspect(value):
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    _require(all(bool(np.all(edge == bg)) for edge in
                 (a[0], a[-1], a[:, 0], a[:, -1])),
             "input must have a uniform background border")
    seen = np.zeros(a.shape, dtype=bool)
    objects = []
    for sr, sc in np.argwhere(a != bg):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        cells, stack = [], [(sr, sc)]
        seen[sr, sc] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= nr < h and 0 <= nc < w and not seen[nr, nc]
                        and a[nr, nc] != bg):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        r = min(x[0] for x in cells)
        c = min(x[1] for x in cells)
        s = max(x[0] for x in cells) - r + 1
        _require(s >= 3 and max(x[1] for x in cells) - c + 1 == s
                 and len(cells) == s*s, "components must be solid square footprints")
        p = a[r:r+s, c:c+s]
        frame, core = int(p[0, 0]), int(p[1, 1])
        _require(len({bg, frame, core}) == 3, "roles must be distinct")
        _require(all(bool(np.all(edge == frame)) for edge in
                     (p[0], p[-1], p[:, 0], p[:, -1]))
                 and bool(np.all(p[1:-1, 1:-1] == core)),
                 "each square needs a one-cell frame and uniform center")
        k = s - 2
        _require(r >= k and c >= k and r+s+k <= h and c+s+k <= w,
                 "all four full arms must fit")
        objects.append((r, c, k, frame, core))
    _require(3 <= len(objects) <= 9, "need 3..9 framed squares")
    cores = {o[4] for o in objects}
    frames = {o[3] for o in objects}
    _require(len(cores) == 1 and not cores & frames,
             "all centers share a color absent from the frames")
    _require(2 <= len(frames) <= 4, "need 2..4 competing frame colors")
    footprints = np.zeros(a.shape, dtype=bool)
    claims = {color: np.zeros(a.shape, dtype=bool) for color in frames}
    for obj in objects:
        fp, arm = _masks(a.shape, obj)
        footprints |= fp
        claims[obj[3]] |= arm
    _require(not any(bool(np.any(mask & footprints)) for mask in claims.values()),
             "arms may cross other arms but never an input square")
    scores = {color: int(np.count_nonzero(a == color)) for color in frames}
    _require(len(set(scores.values())) == len(scores),
             "competing colors must have unequal original populations")
    multiplicity = sum(mask.astype(np.int8) for mask in claims.values())
    _require(bool(np.any(multiplicity >= 2)), "need a cross-color arm overlap")
    return {"grid": a, "background": bg, "core": next(iter(cores)),
            "objects": objects, "scores": scores, "claims": claims,
            "footprints": footprints, "multiplicity": multiplicity}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Swap every square, grow all arms, and resolve overlaps by input counts."""
    info = _inspect(input_grid)
    out = info["grid"].copy()
    for r, c, k, frame, core in info["objects"]:
        s = k + 2
        out[r:r+s, c:c+s] = core
        out[r+1:r+s-1, c+1:c+s-1] = frame
    # Union within a color first: the number of claims at a cell is irrelevant.
    for color in sorted(info["scores"], key=info["scores"].get):
        out[info["claims"][color]] = color
    return out


def _legal_placement(shape, obj, placed, remote=False):
    r, c, k = obj
    s = k + 2
    h, w = shape
    if r < k or c < k or r+s+k > h or c+s+k > w:
        return False
    fp, arm = _masks(shape, obj)
    for old in placed:
        rr, cc, kk = old[:3]
        ss = kk + 2
        # Background separation makes foreground-component parsing unambiguous.
        if r <= rr+ss and r+s >= rr and c <= cc+ss and c+s >= cc:
            return False
        old_fp, old_arm = _masks(shape, old)
        if np.any(fp & (old_fp | old_arm)) or np.any(arm & old_fp):
            return False
        if remote and np.any(arm & old_arm):
            return False
    return True


def _place_one(rng, shape, k, placed, prefer_contact=False, remote=False):
    h, w = shape
    positions = [(r, c, k) for r in range(k, h-2*k-1)
                 for c in range(k, w-2*k-1)]
    rng.shuffle(positions)
    fallback = None
    old_arms = np.zeros(shape, dtype=bool)
    for old in placed:
        old_arms |= _masks(shape, old)[1]
    for obj in positions:
        if not _legal_placement(shape, obj, placed, remote):
            continue
        if not prefer_contact or np.any(_masks(shape, obj)[1] & old_arms):
            return obj
        if fallback is None:
            fallback = obj
    return fallback


def _contact_edges(shape, objects):
    arms = [_masks(shape, obj)[1] for obj in objects]
    return [(i, j) for i in range(len(objects)) for j in range(i)
            if np.any(arms[i] & arms[j])]


def _draw(shape, geometry, labels, palette):
    bg, core, *colors = palette
    grid = np.full(shape, bg, dtype=np.int8)
    for (r, c, k), label in zip(geometry, labels):
        s = k + 2
        grid[r:r+s, c:c+s] = colors[label]
        grid[r+1:r+s-1, c+1:c+s-1] = core
    return grid


def _orient(rng, arrays):
    turns = int(rng.integers(0, 4))
    mirror = bool(rng.integers(0, 2))
    return [np.fliplr(np.rot90(a, turns)).copy() if mirror
            else np.rot90(a, turns).copy() for a in arrays]


def _witness_pair(rng):
    """Same arena twice; recoloring remote donors reverses both local contests."""
    for attempt in range(160):
        a = int(rng.integers(1, 3))
        b = int(rng.integers(2, 4))
        c = int(rng.integers(4, 6))
        donors = [int(x) for x in rng.integers(1, 3, size=3)]
        wa, wb, wc = a+1, b+1, c+1
        wd = [k+1 for k in donors]
        if not (wa+wd[0] < wc < wb+wd[1]+wd[2]
                and wb < wc < wa+sum(wd)):
            continue
        # Counting the entire two-color footprint, the centers, or the proposed
        # post-growth frame populations must not explain the same winners.
        body = lambda k: (k+2)**2
        grown = lambda k: 5*k*k + 8*k
        if not (min(body(b)+body(donors[1])+body(donors[2]),
                    body(a)+sum(body(k) for k in donors)) < body(c)
                and min(grown(b)+grown(donors[1])+grown(donors[2]),
                        grown(a)+sum(grown(k) for k in donors)) < grown(c)):
            continue
        # A is west of C; B is either north or east. Later D4 transforms remove
        # any privileged direction. The gaps vary whenever smaller arms allow it.
        sa, sb, s = a+2, b+2, c+2
        gap_a = int(rng.integers(c, c+a))
        gap_b = int(rng.integers(c, c+b))
        arena = [(int(rng.integers(0, s-sa+1)), -gap_a-sa, a),
                 (0, 0, b), (0, 0, c)]
        if rng.random() < .55:
            arena[1] = (-gap_b-sb, int(rng.integers(0, s-sb+1)), b)
        else:
            arena[1] = (int(rng.integers(0, s-sb+1)), s+gap_b, b)
        min_r = min(r-k for r, _, k in arena)
        min_c = min(col-k for _, col, k in arena)
        max_r = max(r+2*k+2 for r, _, k in arena)
        max_c = max(col+2*k+2 for _, col, k in arena)
        need_h, need_w = max_r-min_r, max_c-min_c
        if max(need_h, need_w) > 30:
            continue
        h = int(rng.integers(max(24, need_h), 31))
        w = int(rng.integers(max(24, need_w), 31))
        dr = -min_r + int(rng.integers(0, h-need_h+1))
        dc = -min_c + int(rng.integers(0, w-need_w+1))
        geometry = [(r+dr, col+dc, k) for r, col, k in arena]
        if not all(_legal_placement((h, w), obj, geometry[:i])
                   for i, obj in enumerate(geometry)):
            continue
        for k in donors:
            obj = _place_one(rng, (h, w), k, geometry, remote=True)
            if obj is None:
                break
            geometry.append(obj)
        if len(geometry) != 6:
            continue
        palette = [int(x) for x in rng.choice(10, 5, replace=False)]
        first = _draw((h, w), geometry, [0, 1, 2, 0, 1, 1], palette)
        second = _draw((h, w), geometry, [0, 1, 2, 0, 0, 0], palette)
        first, second = _orient(rng, [first, second])
        # Both changed local winner and a global-count / object-count reversal
        # follow algebraically from the inequalities above.
        _inspect(first)
        _inspect(second)
        return first, second, c
    raise RuntimeError("could not construct the contrasting demonstrations")


def _general_grid(rng, is_test, anchor_k):
    for attempt in range(180):
        n = int(rng.integers(7, 10) if is_test else rng.integers(4, 9))
        m = int(rng.choice([3, 4]) if is_test else rng.choice([2, 3]))
        sizes = [int(rng.integers(1, 4)) for _ in range(n)]
        if rng.random() < (.55 if is_test else .25):
            sizes[0] = min(6, anchor_k + int(is_test))
            # One larger square and a population of small ones fit naturally.
            sizes[1:] = [int(rng.integers(1, 3)) for _ in range(n-1)]
        sizes.sort(reverse=True)
        lower = max(19 if n < 7 else 23, 3*max(sizes)+2)
        h, w = (int(rng.integers(lower, 31)) for _ in range(2))
        geometry = []
        for k in sizes:
            obj = _place_one(rng, (h, w), k, geometry,
                             prefer_contact=bool(geometry and rng.random() < .85))
            if obj is None:
                break
            geometry.append(obj)
        if len(geometry) != n:
            continue
        edges = _contact_edges((h, w), geometry)
        if len(edges) < 2:
            continue
        palette = [int(x) for x in rng.choice(10, m+2, replace=False)]
        for _ in range(60):
            labels = list(range(m)) + [int(x) for x in rng.integers(0, m, n-m)]
            rng.shuffle(labels)
            populations = [sum(4*(k+1) for (_, _, k), label in zip(geometry, labels)
                               if label == color) for color in range(m)]
            if len(set(populations)) != m:
                continue
            color_edges = {tuple(sorted((labels[i], labels[j]))) for i, j in edges
                           if labels[i] != labels[j]}
            if len(color_edges) < m-1:
                continue
            # Every identity belongs to one connected competition graph.
            reached = {0}
            for _ in range(m):
                reached |= {c for edge in color_edges if reached.intersection(edge)
                            for c in edge}
            if len(reached) != m:
                continue
            grid = _draw((h, w), geometry, labels, palette)
            return _orient(rng, [grid])[0]
    raise RuntimeError("could not pack a competitive growth scene")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_))
                 and isinstance(value, (int, np.integer)) and lo <= value <= hi,
                 f"{name} must be an integer in {lo}..{hi}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    first, second, anchor = _witness_pair(rng)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(50):
                if split == "train" and index < 2:
                    grid = (first, second)[index]
                else:
                    grid = _general_grid(rng, split == "test", anchor)
                if _key(grid) not in used:
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
            used.add(_key(grid))
            puzzle[split].append({"input": grid, "output": solve(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _semantic_check(pair):
    """Check every claim cell using coordinate enumeration rather than painting."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have exactly input/output")
    info = _inspect(pair["input"])
    out = _grid(pair["output"])
    a = info["grid"]
    _require(out.shape == a.shape, "output must keep canvas dimensions")
    candidates = {}
    expected = np.full(a.shape, info["background"], dtype=a.dtype)
    for r, c, k, frame, core in info["objects"]:
        s = k+2
        for rr in range(r, r+s):
            for cc in range(c, c+s):
                expected[rr, cc] = (frame if r < rr < r+s-1 and c < cc < c+s-1
                                    else core)
        arm_cells = set()
        for depth in range(1, k+1):
            for offset in range(s):
                arm_cells.update(((r-depth, c+offset), (r+s-1+depth, c+offset),
                                  (r+offset, c-depth), (r+offset, c+s-1+depth)))
        _require(len(arm_cells) == 4*k*s, "wrong proposed arm area")
        for cell in arm_cells:
            _require(a[cell] == info["background"], "an arm hits a source square")
            candidates.setdefault(cell, set()).add(frame)
    for cell, colors in candidates.items():
        expected[cell] = max(colors, key=info["scores"].get)
    _require(np.array_equal(out, expected), "incorrect swap, arm, or collision result")
    _require(not np.array_equal(a, out), "identity pair")
    for color, score in info["scores"].items():
        perimeter_sum = sum(4*(o[2]+1) for o in info["objects"] if o[3] == color)
        _require(score == perimeter_sum, "score must count every original frame cell")
    info["candidates"] = candidates
    return info


def _witnesses(info):
    """Observable counterexamples at actual cross-color arm intersections."""
    objects, scores, shape = info["objects"], info["scores"], info["grid"].shape
    number = Counter(o[3] for o in objects)
    boxes = {}
    for color in scores:
        points = np.argwhere(info["grid"] == color)
        boxes[color] = int(np.prod(points.max(axis=0)-points.min(axis=0)+1))
    alternatives = [{color: sum(weight(o[2]) for o in objects if o[3] == color)
                     for color in scores}
                    for weight in (lambda k: (k+2)**2, lambda k: k*k,
                                   lambda k: 5*k*k+8*k)]
    small_wins = fewer_wins = bbox_reversal = False
    other_reversals = [False, False, False]
    for i, j in _contact_edges(shape, objects):
        p, q = objects[i], objects[j]
        if p[3] == q[3]:
            continue
        winner, loser = (p, q) if scores[p[3]] > scores[q[3]] else (q, p)
        small_wins |= winner[2] < loser[2]
        fewer_wins |= number[winner[3]] < number[loser[3]]
        bbox_reversal |= boxes[winner[3]] < boxes[loser[3]]
        for index, counts in enumerate(alternatives):
            other_reversals[index] |= counts[winner[3]] < counts[loser[3]]
    return tuple(bool(x) for x in (small_wins, fewer_wins, bbox_reversal,
                                  *other_reversals))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _semantic_check(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append(info)
    # The first two are deliberately a controlled comparison. This checks the
    # teaching evidence, separately from the per-grid semantic checks above.
    a, b = training[:2]
    _require(a["grid"].shape == b["grid"].shape
             and np.array_equal(a["footprints"], b["footprints"])
             and a["background"] == b["background"] and a["core"] == b["core"],
             "first demonstrations must preserve the geometric arena")
    flip = False
    for cell, colors in a["candidates"].items():
        if len(colors) >= 2 and b["candidates"].get(cell) == colors:
            flip |= max(colors, key=a["scores"].get) != max(colors, key=b["scores"].get)
    _require(flip, "remote population changes must reverse an unchanged local contest")
    evidence = [_witnesses(info) for info in training]
    _require(any(w[0] for w in evidence), "training must refute largest local square")
    _require(any(w[1] for w in evidence), "training must refute counting squares")
    _require(all(any(w[i] for w in evidence) for i in (3, 4, 5)),
             "training must refute footprint, center, and post-growth counting")


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in puzzle[s])


def self_test():
    """Bounded semantic and structural checks over 42 seeds, plus count edges."""
    shapes, structures, palettes, backgrounds, sizes = set(), set(), set(), set(), set()
    object_counts, clan_counts, collision_areas, test_orders = set(), set(), set(), set()
    witnesses = [0] * 6
    boundary, triple, same_color_crossings = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)),
                 "same seed must reproduce the whole puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _semantic_check(pair)
                a = info["grid"]
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                palettes.add(tuple(sorted(info["scores"])))
                object_counts.add(len(info["objects"]))
                clan_counts.add(len(info["scores"]))
                sizes.update(o[2] for o in info["objects"])
                collision_areas.add(int(np.count_nonzero(info["multiplicity"] >= 2)))
                triple += int(np.any(info["multiplicity"] >= 3))
                for flag, present in enumerate(_witnesses(info)):
                    witnesses[flag] += int(present)
                masks = [_masks(a.shape, o)[1] for o in info["objects"]]
                for i, j in _contact_edges(a.shape, info["objects"]):
                    same_color_crossings += int(info["objects"][i][3] == info["objects"][j][3])
                boundary += int(any(np.any(edge) for mask in masks
                                    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1])))
                roles = {}
                structures.add((a.shape, bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)))
                # Color permutation equivariance, including background and core.
                permutation = np.random.default_rng(seed+900).permutation(10)
                _require(np.array_equal(solve(permutation[a]), permutation[pair["output"]]),
                         "literal color identity affected the rule")
                # Geometric equivariance also catches directional overwrite bugs.
                _require(np.array_equal(solve(np.rot90(a)), np.rot90(pair["output"])),
                         "rotation changed the rule")
                if split == "test":
                    test_orders.add(tuple(sorted(info["scores"], key=info["scores"].get)))
                    _require(len(info["objects"]) >= 7, "test composition shift missing")
    _require(len(shapes) >= 35 and len(structures) >= 150, "insufficient geometric variation")
    _require(len(backgrounds) == 10 and len(palettes) >= 30, "insufficient palette variation")
    _require({1, 2, 3, 4, 5, 6} <= sizes, "scale relationship underexercised")
    _require({4, 5, 6, 7, 8, 9} <= object_counts and clan_counts == {2, 3, 4},
             "object or clan counts underexercised")
    _require(witnesses[0] >= 42 and witnesses[1] >= 42 and witnesses[2] >= 20,
             "shortcut counterexamples underexercised")
    _require(all(x >= 42 for x in witnesses[3:]), "population timing ambiguous")
    _require(boundary >= 10 and len(collision_areas) >= 10 and same_color_crossings >= 10,
             "boundary or overlap variation underexercised")
    # These valid, deliberately unsampled extremes check that solve's domain is
    # the specification, rather than the sampler's usual sizes and counts.
    for k in (7, 8, 9):
        a = _draw((30, 30), [(9, 9, k), (3, 6, 1), (24, 24, 1)],
                  [0, 1, 1], [0, 7, 4, 9])
        before = a.copy()
        result = solve(a)
        _semantic_check({"input": a, "output": result})
        _require(np.array_equal(a, before), "solve mutated its input")
    for a, b in ((2, 1), (10, 4), (6, 3)):
        p = generate_puzzle(123456, a, b)
        _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes),
            "color_normalized_inputs": len(structures), "backgrounds": len(backgrounds),
            "frame_palettes": len(palettes), "object_counts": sorted(object_counts),
            "clan_counts": sorted(clan_counts), "inner_sizes": sorted(sizes),
            "distinct_overlap_areas": len(collision_areas), "boundary_pairs": boundary,
            "triple_color_overlap_pairs": triple, "same_color_crossings": same_color_crossings,
            "shortcut_witness_pairs": dict(zip(("local_size", "object_count", "box_area",
                                                 "footprint_area", "center_area", "post_growth_count"), witnesses)),
            "distinct_test_priority_orders": len(test_orders)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
