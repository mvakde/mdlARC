"""Local cancellation tallies drawn as completed quadrants of quadrants.

No parent modules, source pairs, stored examples, or generation state are used
by solve(). See task_spec.md for the complete family contract.
"""
from __future__ import annotations

import numpy as np


UP = ((0, 1), (1, 0), (1, 1), (1, 2))
DOWN = ((0, 0), (0, 1), (0, 2), (1, 1))
NOISE = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
SLOTS = ((0, 0), (1, 0), (0, 1), (1, 1))
COUNT_PAIRS = tuple((a, b) for a in range(5) for b in range(5))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape), "grid dimensions")
    _require(np.issubdtype(array.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= array) & (array <= 9)), "ARC color range")
    return array


def _key(grid):
    array = np.asarray(grid)
    return array.shape, array.tobytes() if array.dtype == np.int8 else array.astype(np.int8).tobytes()


def _quadrant(point, shape):
    return 2 * int(point[0] >= shape[0] // 2) + int(point[1] >= shape[1] // 2)


def _components(grid, background):
    unseen = set(map(tuple, np.argwhere(grid != background).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, group = [start], []
        while stack:
            row, col = stack.pop()
            group.append((row, col))
            for dr, dc in STEPS:
                neighbor = (row + dr, col + dc)
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(tuple(sorted(group)))
    return result


def _normalized(group):
    row = min(r for r, _ in group)
    col = min(c for _, c in group)
    return frozenset((r - row, c - col) for r, c in group)


def _analyze_input(value):
    """Parse only visible input evidence; also reject out-of-family scenes."""
    grid = _grid(value)
    _require(all(n in (16, 20, 24, 28) for n in grid.shape), "canvas sides must be 16, 20, 24, or 28")
    hues, populations = np.unique(grid, return_counts=True)
    background = int(hues[np.argmax(populations)])
    _require(int(populations.max()) * 2 > grid.size, "background must be a strict majority")
    groups = _components(grid, background)
    dots = [g for g in groups if len(g) == 1]
    _require(len(dots) == 1, "exactly one isolated marker required")
    marker = dots[0][0]
    signal = int(grid[marker])
    _require(np.count_nonzero(grid == signal) == 1, "marker hue must be unique")
    counts = [[0, 0] for _ in range(4)]
    objects, noise = [], []
    allowed_noise = {frozenset(s) for s in NOISE}
    for group in groups:
        color = int(grid[group[0]])
        _require(all(int(grid[p]) == color for p in group), "components must be monochrome")
        if len(group) == 1:
            continue
        quadrants = {_quadrant(p, grid.shape) for p in group}
        _require(len(quadrants) == 1, "objects cannot cross a midpoint")
        q = next(iter(quadrants))
        shape = _normalized(group)
        if shape in (frozenset(UP), frozenset(DOWN)):
            orientation = 0 if shape == frozenset(UP) else 1
            counts[q][orientation] += 1
            objects.append((q, orientation, color, group))
        else:
            _require(shape in allowed_noise, "unsupported distractor shape")
            noise.append(group)
    _require(all(0 <= n <= 4 for pair in counts for n in pair), "at most four Ts per orientation per quadrant")
    _require(bool(objects), "at least one T required")
    q = _quadrant(marker, grid.shape)
    above, below = counts[q]
    return {"grid": grid, "background": background, "signal": signal,
            "marker": marker, "quadrant": q, "counts": counts,
            "above": above, "below": below, "difference": above - below,
            "objects": objects, "noise": noise}


def _render(shape, background, signal, quadrant, difference):
    output = np.full(shape, background, dtype=np.int8)
    h, w = shape[0] // 4, shape[1] // 4
    origin_r = (quadrant // 2) * 2 * h
    origin_c = (quadrant % 2) * 2 * w
    order = SLOTS if difference >= 0 else SLOTS[::-1]
    for dr, dc in order[:abs(difference)]:
        r, c = origin_r + dr * h, origin_c + dc * w
        output[r:r+h, c:c+w] = signal
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _analyze_input(input_grid)
    return _render(facts["grid"].shape, facts["background"], facts["signal"],
                   facts["quadrant"], facts["difference"])


def _count_plan(rng, number):
    contrasts = [(p, q) for p in COUNT_PAIRS for q in COUNT_PAIRS
                 if sum(p) == sum(q) and p[0] > p[1] and q[0] < q[1]
                 and p[0] - p[1] != q[1] - q[0]]
    first, second = contrasts[int(rng.integers(len(contrasts)))]
    train = [first, second]
    if rng.integers(2):
        train.reverse()
    if number >= 3:
        equal = int(rng.integers(1, 5))
        train.append((equal, equal))
    while len(train) < number:
        used_magnitudes = {abs(a-b) for a, b in train}
        candidates = [p for p in COUNT_PAIRS if p not in train
                      and abs(p[0]-p[1]) not in used_magnitudes]
        if not candidates:
            candidates = [p for p in COUNT_PAIRS if p not in train]
        train.append(candidates[int(rng.integers(len(candidates)))])
    return train


def _outside_plan(rng, local_difference, reverse):
    options = []
    for a in range(8):
        for b in range(8):
            global_difference = local_difference + a - b
            if not (3 <= a+b <= 9 and -4 <= global_difference <= 4):
                continue
            if global_difference == local_difference:
                continue
            if reverse and global_difference * local_difference >= 0:
                continue
            options.append((a, b))
    return options[int(rng.integers(len(options)))]


def _legal(cells, occupied):
    return not any(p in occupied or any((p[0]+dr, p[1]+dc) in occupied for dr, dc in STEPS)
                   for p in cells)


def _place(rng, shape, quadrant, offsets, occupied, mode):
    half_h, half_w = shape[0] // 2, shape[1] // 2
    r0, c0 = (quadrant // 2) * half_h, (quadrant % 2) * half_w
    box_h = max(r for r, _ in offsets) + 1
    box_w = max(c for _, c in offsets) + 1
    anchors = [(r, c) for r in range(r0, r0 + half_h - box_h + 1)
               for c in range(c0, c0 + half_w - box_w + 1)]
    choices = []
    local = [p for p in occupied if _quadrant(p, shape) == quadrant]
    for index in rng.permutation(len(anchors)):
        r, c = anchors[int(index)]
        cells = tuple((r+dr, c+dc) for dr, dc in offsets)
        if not _legal(cells, occupied):
            continue
        if mode == "scatter":
            return cells
        if mode == "edge":
            score = min(r-r0, c-c0, r0+half_h-r-box_h, c0+half_w-c-box_w)
        elif local:
            distance = min(abs(r-rr)+abs(c-cc) for rr, cc in local)
            score = distance if mode == "cluster" else -distance
        else:
            score = 0
        choices.append((score, cells))
        if len(choices) == 12:
            break
    if not choices:
        return None
    best = min(score for score, _ in choices)
    finalists = [cells for score, cells in choices if score == best]
    return finalists[int(rng.integers(len(finalists)))]


def _scene(rng, local_counts, quadrant, shape, background, signal, palette,
           multicolor, mode, reverse=False):
    outside = _outside_plan(rng, local_counts[0]-local_counts[1], reverse)
    descriptors = [(quadrant, o) for o in (0, 1) for _ in range(local_counts[o])]
    others = [q for q in range(4) if q != quadrant]
    exterior = [0] * outside[0] + [1] * outside[1]
    rng.shuffle(exterior)
    loads = {q: 0 for q in others}
    for orientation in exterior:
        available = [q for q in others if loads[q] < 4]
        q = available[int(rng.integers(len(available)))]
        loads[q] += 1
        descriptors.append((q, orientation))
    # Glyph hue is independent of class. In multicolor scenes one hue is
    # explicitly shared by both orientations, and another hue is also present.
    labels = [int(rng.choice(palette if multicolor else palette[:1])) for _ in descriptors]
    if multicolor and len(palette) > 1:
        shared = []
        for orientation in (0, 1):
            indices = [i for i, (_, o) in enumerate(descriptors) if o == orientation]
            if indices:
                labels[indices[0]] = palette[0]
                shared.append(indices[0])
        alternatives = [i for i in range(len(labels)) if i not in shared]
        if alternatives:
            labels[alternatives[0]] = palette[1]
    noises = [(int(rng.integers(4)), NOISE[int(rng.integers(len(NOISE)))],
               int(rng.choice(palette))) for _ in range(int(rng.integers(1, 5)))]
    for _ in range(32):
        grid = np.full(shape, background, dtype=np.int8)
        half_h, half_w = shape[0] // 2, shape[1] // 2
        marker = ((quadrant // 2)*half_h + int(rng.integers(half_h)),
                  (quadrant % 2)*half_w + int(rng.integers(half_w)))
        occupied = {marker}
        grid[marker] = signal
        complete = True
        for i in rng.permutation(len(descriptors)):
            q, orientation = descriptors[int(i)]
            cells = _place(rng, shape, q, UP if orientation == 0 else DOWN, occupied, mode)
            if cells is None:
                complete = False
                break
            occupied.update(cells)
            for cell in cells:
                grid[cell] = labels[int(i)]
        if not complete:
            continue
        for q, offsets, color in noises:
            cells = _place(rng, shape, q, offsets, occupied, mode)
            if cells is None:
                complete = False
                break
            occupied.update(cells)
            for cell in cells:
                grid[cell] = color
        if complete:
            return {"input": grid, "output": _render(shape, background, signal, quadrant,
                                                    local_counts[0]-local_counts[1])}
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)]
    background, signal = colors[:2]
    palette = colors[2:2+int(rng.integers(2, 5))]
    train_counts = _count_plan(rng, num_train)
    quadrants = [int(q) for q in rng.permutation(4)]
    held_quadrant = bool(rng.integers(2))
    train_q = quadrants[:2]
    while len(train_q) < num_train:
        if held_quadrant:
            train_q.append(quadrants[int(rng.integers(2))])
        elif len(train_q) < 4:
            train_q.append(quadrants[len(train_q)])
        else:
            train_q.append(int(rng.integers(4)))
    differences = {a-b for a, b in train_counts}
    candidates = [p for p in COUNT_PAIRS if p[0]-p[1] in differences and p not in train_counts]
    _require(bool(candidates), "no new count combination available")
    test_counts = [candidates[int(rng.integers(len(candidates)))]]
    while len(test_counts) < num_test:
        pool = [p for p in COUNT_PAIRS if p[0]-p[1] in differences]
        test_counts.append(pool[int(rng.integers(len(pool)))])
    unseen = [q for q in quadrants if q not in train_q]
    test_q = [unseen[0] if unseen else train_q[int(rng.integers(num_train))]]
    while len(test_q) < num_test:
        pool = [q for q in range(4) if q != test_q[-1]]
        test_q.append(pool[int(rng.integers(3))])
    puzzle, seen, training_shapes = {"train": [], "test": []}, set(), []
    modes = ("scatter", "cluster", "spread", "edge")
    for split, plans, qs in (("train", train_counts, train_q), ("test", test_counts, test_q)):
        for index, (counts, q) in enumerate(zip(plans, qs)):
            mode = modes[int(rng.integers(4))]
            for attempt in range(80):
                if split == "test":
                    compatible = [s for s in training_shapes if sum(counts) <= 6 or min(s) >= 20]
                    shape = compatible[int(rng.integers(len(compatible)))]
                else:
                    # High-count local censuses receive room; all scenes can
                    # otherwise be compact or roomy, independently of split.
                    side_pool = (20, 24, 28) if index == 0 or sum(counts) >= 7 else (16, 20, 24, 28)
                    shape = (int(rng.choice(side_pool)), int(rng.choice(side_pool)))
                pair = _scene(rng, counts, q, shape, background, signal, palette,
                              multicolor=(index != 0 or split == "test"), mode=mode,
                              reverse=(split == "train" and index == 0))
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("unable to place a distinct scene")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training_shapes.append(pair["input"].shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def _expected_by_tiles(facts):
    """Validator construction by cell membership rather than rectangle writes."""
    grid = facts["grid"]
    h, w = grid.shape[0] // 4, grid.shape[1] // 4
    q, difference = facts["quadrant"], facts["difference"]
    expected = np.full(grid.shape, facts["background"], dtype=np.int8)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if _quadrant((r, c), grid.shape) != q:
                continue
            local_r, local_c = (r // h) % 2, (c // w) % 2
            rank = 2 * local_c + local_r
            if difference > 0 and rank < difference or difference < 0 and rank >= 4+difference:
                expected[r, c] = facts["signal"]
    return expected


def _global_ablation(facts):
    difference = sum(a-b for a, b in facts["counts"])
    difference = min(4, max(-4, difference))
    return _render(facts["grid"].shape, facts["background"], facts["signal"],
                   facts["quadrant"], difference)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, roles, facts = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _analyze_input(inp)
            _require(out.shape == inp.shape, "output shape")
            _require(np.array_equal(out, _expected_by_tiles(f)), "incorrect cancellation mosaic")
            _require(np.count_nonzero(out == f["signal"]) == abs(f["difference"]) * inp.size // 16,
                     "signal area does not encode the surviving count")
            _require(bool(f["noise"]), "generated scene needs non-T clutter")
            _require(not np.array_equal(out, _global_ablation(f)), "global counting shortcut survives")
            roles.add((f["background"], f["signal"]))
            facts[split].append(f)
    _require(len(roles) == 1, "puzzle-level background and marker hues must be stable")
    train = facts["train"]
    ds = {f["difference"] for f in train}
    _require(any(d > 0 for d in ds) and any(d < 0 for d in ds), "both tally directions must be demonstrated")
    _require(len({abs(d) for d in ds if d}) >= 2, "majority direction alone must be insufficient")
    _require(len({f["quadrant"] for f in train}) >= 2, "two training quadrants required")
    _require(any(sum((a["above"], a["below"])) == sum((b["above"], b["below"]))
                 and a["difference"] * b["difference"] < 0
                 and abs(a["difference"]) != abs(b["difference"])
                 for i, a in enumerate(train) for b in train[i+1:]), "same-total orientation contrast")
    _require(any(f["above"] and f["below"] and f["difference"] for f in train), "cancellation must affect a training answer")
    if nt >= 3:
        _require(0 in ds, "tie demonstration required")
    train_counts = {(f["above"], f["below"]) for f in train}
    train_shapes = {f["grid"].shape for f in train}
    _require(any(min(shape) >= 20 for shape in train_shapes), "room for high-count recombinations must be demonstrated")
    _require((facts["test"][0]["above"], facts["test"][0]["below"]) not in train_counts,
             "first test must recombine local counts")
    for f in facts["test"]:
        _require(f["difference"] in ds, "test signed tally state was not taught")
        _require(f["grid"].shape in train_shapes, "test canvas shape was not taught")
        _require(f["above"] + f["below"] <= 6 or min(f["grid"].shape) >= 20,
                 "high-count test needs a compatible demonstrated canvas")
    return None


def _move_marker(facts, quadrant):
    grid = facts["grid"].copy()
    grid[facts["marker"]] = facts["background"]
    occupied = set(map(tuple, np.argwhere(grid != facts["background"]).tolist()))
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            p = (r, c)
            if p != facts["marker"] and _quadrant(p, grid.shape) == quadrant and _legal((p,), occupied):
                grid[p] = facts["signal"]
                return grid
    raise AssertionError("no place to move marker")


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded observable invariants, counterfactuals, and diversity: 42 seeds."""
    signatures, shapes, structures, palettes, count_plans = set(), set(), set(), set(), set()
    differences, quadrants, clutter_shapes, local_totals = set(), set(), set(), set()
    diagonal = mono = multi = unseen_q = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        per_split = {}
        for split in ("train", "test"):
            per_split[split] = []
            for pair in puzzle[split]:
                f = _analyze_input(pair["input"])
                per_split[split].append(f)
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "input-only solve")
                shapes.add(f["grid"].shape)
                palettes.add((f["background"], f["signal"]))
                differences.add(f["difference"])
                quadrants.add(f["quadrant"])
                local_totals.add(f["above"] + f["below"])
                clutter_shapes.update(_normalized(g) for g in f["noise"])
                structures.add((f["grid"].shape, f["marker"],
                                tuple((q, o, g) for q, o, _, g in f["objects"])))
                hues = {color for _, _, color, _ in f["objects"]}
                mono += len(hues) == 1
                multi += len(hues) > 1
                components = _components(f["grid"], f["background"])
                labels = {p: i for i, group in enumerate(components) for p in group}
                diagonal += any((r+1, c+dc) in labels and labels[(r+1, c+dc)] != label
                                for (r, c), label in labels.items() for dc in (-1, 1))
        train_facts = per_split["train"]
        count_plans.add(tuple((f["above"], f["below"]) for f in train_facts))
        train_q = {f["quadrant"] for f in train_facts}
        unseen_q += any(f["quadrant"] not in train_q for f in per_split["test"])
        f = train_facts[0]
        moved = _move_marker(f, f["quadrant"])
        _require(np.array_equal(solve(moved), puzzle["train"][0]["output"]), "within-quadrant marker invariance")
        moved = _move_marker(f, (f["quadrant"] + 1) % 4)
        _require(not np.array_equal(solve(moved), puzzle["train"][0]["output"]), "quadrant selection must matter")
        _require(np.array_equal(solve(np.rot90(f["grid"], 2)),
                                np.rot90(puzzle["train"][0]["output"], 2)),
                 "180-degree rotation must swap orientations and rotate the tally")
        # Erasing orientation and counting all selected Ts would fill four
        # blocks or a positive prefix; it must fail on some demonstration.
        _require(any(not np.array_equal(pair["output"], _render(g["grid"].shape,
                     g["background"], g["signal"], g["quadrant"], min(4, g["above"]+g["below"])))
                     for pair, g in zip(puzzle["train"], train_facts)), "orientation ablation")
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 12 and len(structures) >= 200, "geometry diversity")
    _require(len(palettes) >= 25 and len(count_plans) >= 30, "palette/count-plan diversity")
    _require(differences == set(range(-4, 5)) and quadrants == set(range(4)), "state/quadrant coverage")
    _require(len(local_totals) >= 8 and len(clutter_shapes) == len(NOISE), "count/clutter coverage")
    _require(diagonal >= 30 and mono >= 42 and multi >= 100 and unseen_q >= 10, "layout/presentation coverage")
    _require(_puzzle_key(generate_puzzle(156706536)) == _puzzle_key(generate_puzzle(156706536)), "determinism")
    large_tie = _scene(np.random.default_rng(0), (4, 4), 0, (20, 20), 0, 1,
                       [2, 3], True, "cluster")
    _require(large_tie is not None and np.all(solve(large_tie["input"]) == 0),
             "maximum-population tie must fit and cancel on a compatible canvas")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(98431082, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    point = (0, 0)
    broken["test"][0]["output"][point] = (int(broken["test"][0]["output"][point])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "structures": len(structures), "palettes": len(palettes),
            "count_plans": len(count_plans), "differences": sorted(differences),
            "diagonal_contact_pairs": diagonal, "monochrome_pairs": mono,
            "multicolor_pairs": multi, "puzzles_with_unseen_test_quadrant": unseen_q}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
