"""Exhaustion voting: square inventories select the electorate at every rank.

Only Python's standard library and NumPy are used. No parent code is imported.
The solver has no generation state or learned palette convention.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors outside 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(a, background):
    """Maximal monochromatic eight-connected components, from visible cells."""
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(a != background):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        color = int(a[r, c])
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and int(a[yy, xx]) == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        result.append((color, tuple(sorted(cells))))
    return result


def _analyze(input_grid):
    """Parse squares and strokes without any generation metadata."""
    a = _grid(input_grid)
    hist = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    _require(int(hist[bg]) * 2 > a.size, "background must occupy over half the grid")
    classes, by_color, strokes, squares = {}, {}, [], []
    components = _components(a, bg)
    # Require a background moat even between differently colored objects.
    labels = np.full(a.shape, -1, dtype=int)
    for index, (_, cells) in enumerate(components):
        for r, c in cells:
            labels[r, c] = index
    for index, (_, cells) in enumerate(components):
        for r, c in cells:
            neighborhood = labels[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            _require(np.all((neighborhood == -1) | (neighborhood == index)), "objects touch")
    for color, cells in components:
        rows, cols = zip(*cells)
        hh, ww = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
        if hh == ww and len(cells) == hh * ww and 2 <= hh <= 5:
            side = hh
            if side not in classes:
                _require(color not in by_color, "color used by different square sizes")
                classes[side] = {"color": color, "count": 0, "runs": []}
                by_color[color] = side
            _require(classes[side]["color"] == color, "same-size squares have different colors")
            classes[side]["count"] += 1
            squares.append((side, color, cells))
        else:
            n = len(cells)
            consecutive = hh == n and len(set(rows)) == n
            dr = consecutive and len({r - c for r, c in cells}) == 1
            dl = consecutive and len({r + c for r, c in cells}) == 1
            _require(2 <= n <= 8 and dr != dl, "component is neither a square nor a straight diagonal")
            strokes.append((color, 1 if dr else -1, n, cells))
    _require(2 <= len(classes) <= 4, "need two to four square sizes")
    for color, direction, length, cells in strokes:
        _require(color in by_color, "stroke has no same-colored square class")
        classes[by_color[color]]["runs"].append((direction, length))
    for data in classes.values():
        _require(1 <= data["count"] <= 5, "square multiplicity outside 1..5")
        _require(1 <= len(data["runs"]) <= 3, "each square class needs one to three strokes")
    _require({d for _, d, _, _ in strokes} == {-1, 1}, "both diagonal directions required")
    facts = {"background": bg, "classes": classes, "squares": squares,
             "strokes": strokes, "shape": a.shape}
    stages = _stages(classes)
    _require(all(s["vote"] != 0 for s in stages), "tied active vote")
    _require(sum(max(s["active"]) + 1 for s in stages) <= 30, "output too wide")
    return facts


def _stages(classes):
    result = []
    for rank in range(1, max(v["count"] for v in classes.values()) + 1):
        active = sorted(s for s, data in classes.items() if data["count"] >= rank)
        runs = [(d, length, classes[s]["count"] - rank + 1)
                for s in active for d, length in classes[s]["runs"]]
        vote = sum(d * length for d, length, _ in runs)
        result.append({"rank": rank, "active": active, "runs": runs, "vote": vote})
    return result


def _render(facts):
    classes = facts["classes"]
    stages = _stages(classes)
    width = sum(max(s["active"]) + 1 for s in stages)
    out = np.full((max(classes) + 1, width), facts["background"], dtype=np.int8)
    cursor = 0
    for stage in stages:
        tile_side = max(stage["active"])
        for side in reversed(stage["active"]):
            offset = 0 if stage["vote"] > 0 else tile_side - side
            out[:side, cursor + offset:cursor + offset + side] = classes[side]["color"]
        cursor += tile_side + 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct all rank-specific electorates and their nested composites."""
    return _render(_analyze(input_grid))


def _witnesses(classes):
    stages = _stages(classes)
    informative = [s for s in stages if len(s["active"]) >= 2]
    longest = fewer = weighted = color_majority = largest = False
    for stage in informative:
        win = 1 if stage["vote"] > 0 else -1
        winners = [n for d, n, _ in stage["runs"] if d == win]
        losers = [n for d, n, _ in stage["runs"] if d != win]
        if winners and losers:
            longest |= max(losers) > max(winners)
            fewer |= len(winners) < len(losers)
        weighted |= sum(d * n * copies for d, n, copies in stage["runs"]) * win < 0
        local = {s: sum(d * n for d, n in classes[s]["runs"]) for s in stage["active"]}
        color_vote = sum((v > 0) - (v < 0) for v in local.values())
        color_majority |= color_vote * win < 0
        largest |= local[max(stage["active"])] * win < 0
    directions = {1 if s["vote"] > 0 else -1 for s in informative}
    return {"flip": directions == {-1, 1}, "longest": longest,
            "fewer": fewer, "weighted": weighted,
            "color_majority": color_majority, "largest": largest,
            "initial": 1 if stages[0]["vote"] > 0 else -1}


def _sample_symbolic(rng, mode, initial):
    """Sample object facts, never complete parent-style puzzles."""
    for _ in range(20000):
        if mode in ("longest", "fewer"):
            number = 3
            counts = [int(x) for x in rng.permutation((1, 2, 3))]
        else:
            number = int(rng.choice((2, 3, 4), p=(0.2, 0.5, 0.3)))
            upper = 3 if mode == "extra" else int(rng.integers(4, 6))
            counts = [int(x) for x in rng.integers(1, upper + 1, size=number)]
            if mode == "test":
                counts[int(rng.integers(number))] = upper
            if len(set(counts)) == 1:
                continue
        sides = sorted(int(x) for x in rng.choice(np.arange(2, 6), size=number, replace=False))
        if sum(s * s * n for s, n in zip(sides, counts)) > 205:
            continue
        classes = {}
        for side, count in zip(sides, counts):
            runs = [(int(rng.choice((-1, 1))), int(rng.integers(2, 9)))
                    for _ in range(int(rng.integers(1, 4)))]
            classes[side] = {"count": count, "runs": runs}
        if sum(length for data in classes.values() for _, length in data["runs"]) > 72:
            continue
        if {d for data in classes.values() for d, _ in data["runs"]} != {-1, 1}:
            continue
        stages = _stages(classes)
        if any(s["vote"] == 0 for s in stages):
            continue
        witness = _witnesses(classes)
        if mode in ("longest", "fewer"):
            if not witness["flip"] or witness["initial"] != initial or not witness[mode]:
                continue
            if mode == "longest" and not witness["weighted"]:
                continue
            if mode == "longest" and not witness["largest"]:
                continue
            if mode == "fewer" and not witness["color_majority"]:
                continue
        return classes
    raise RuntimeError("symbolic sampler exhausted")


def _offsets(kind, side, direction):
    if kind == "square":
        return [(r, c) for r in range(side) for c in range(side)]
    return [(i, i if direction == 1 else side - 1 - i) for i in range(side)]


def _pack(rng, classes, background, shape, edge):
    """Greedy placement with cell-wise moats; stroke bounding boxes may overlap."""
    h, w = shape
    items = []
    for side, data in classes.items():
        items.extend(("square", side, 0, data["color"]) for _ in range(data["count"]))
        items.extend(("stroke", n, d, data["color"]) for d, n in data["runs"])
    for _ in range(28):
        blocked = np.zeros(shape, dtype=bool)
        grid = np.full(shape, background, dtype=np.int8)
        order = list(items)
        rng.shuffle(order)
        order.sort(key=lambda item: -(item[1] ** 2 if item[0] == "square" else item[1] * 2))
        if edge:
            index = int(rng.integers(min(4, len(order))))
            order.insert(0, order.pop(index))
        success = True
        for index, (kind, side, direction, color) in enumerate(order):
            offsets = _offsets(kind, side, direction)
            oh, ow = h - side + 1, w - side + 1
            ok = np.ones((oh, ow), dtype=bool)
            for dr, dc in offsets:
                ok &= ~blocked[dr:dr + oh, dc:dc + ow]
            if index == 0 and edge:
                ok[1:-1, 1:-1] = False
            candidates = np.argwhere(ok)
            if not len(candidates):
                success = False
                break
            r, c = (int(x) for x in candidates[int(rng.integers(len(candidates)))])
            for dr, dc in offsets:
                y, x = r + dr, c + dc
                grid[y, x] = color
                blocked[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True
        if success:
            return grid
    raise RuntimeError("object packing failed")


def _make_pair(rng, parameters, mode, initial):
    for _ in range(80):
        classes = _sample_symbolic(rng, mode, initial)
        if parameters["palette_mode"] == "free":
            palette = [int(x) for x in rng.permutation(10)]
        elif parameters["palette_mode"] == "background":
            palette = [parameters["palette"][0]]
            palette.extend(int(x) for x in rng.permutation([c for c in range(10) if c != palette[0]]))
        else:
            palette = parameters["palette"]
        bg = palette[0]
        for side, data in classes.items():
            data["color"] = palette[side - 1]
        # Estimate area occupied by objects plus their separation, then choose
        # a shape independently of the eventual layout.
        footprint = sum((s + 1) ** 2 * v["count"] for s, v in classes.items())
        footprint += sum(4 * n + 2 for v in classes.values() for _, n in v["runs"])
        choices = [(h, w) for h in range(18, 31) for w in range(18, 31)
                   if footprint <= 0.9 * h * w]
        if not choices:
            continue
        shape = choices[int(rng.integers(len(choices)))]
        try:
            a = _pack(rng, classes, bg, shape, edge=bool(rng.random() < 0.6))
            facts = _analyze(a)
        except (RuntimeError, AssertionError):
            continue
        # Audit visible facts against intended objects, including every copy.
        if any(facts["classes"][s]["count"] != v["count"]
               or facts["classes"][s]["color"] != v["color"]
               or sorted(facts["classes"][s]["runs"]) != sorted(v["runs"])
               for s, v in classes.items()):
            continue
        return {"input": a, "output": _render(facts)}
    raise RuntimeError("pair construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    parameters = {"palette": [int(x) for x in rng.permutation(10)],
                  "palette_mode": str(rng.choice(("stable", "free", "background")))}
    first = int(rng.choice((-1, 1)))
    plan = [("longest", first), ("fewer", -first)]
    plan.extend(("extra", 0) for _ in range(num_train - 2))
    rng.shuffle(plan)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, items in (("train", plan), ("test", [("test", 0)] * num_test)):
        for mode, initial in items:
            for _ in range(20):
                pair = _make_pair(rng, parameters, mode, initial)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_output(facts, output):
    """Cellwise coverage audit, distinct from the painter used by solve."""
    classes = facts["classes"]
    stages = _stages(classes)
    bg = facts["background"]
    height = max(classes) + 1
    width = sum(max(s["active"]) + 1 for s in stages)
    _require(output.shape == (height, width), "wrong output dimensions")
    _require(np.all(output[-1] == bg), "missing bottom padding")
    cursor = 0
    for stage in stages:
        maximum = max(stage["active"])
        for r in range(height):
            for c in range(maximum + 1):
                eligible = []
                if c < maximum:
                    for side in stage["active"]:
                        in_column = c < side if stage["vote"] > 0 else c >= maximum - side
                        if r < side and in_column:
                            eligible.append(side)
                expected = classes[min(eligible)]["color"] if eligible else bg
                _require(int(output[r, cursor + c]) == expected, "wrong rank, corner, layer, or separator")
        cursor += maximum + 1
    _require(set(np.unique(output)) == {bg, *(v["color"] for v in classes.values())}, "lost payload color")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split must be a list")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            facts = _analyze(a)
            _check_output(facts, out)
            (training if split == "train" else testing).append(facts)
    witnesses = [_witnesses(f["classes"]) for f in training]
    _require({w["initial"] for w in witnesses if w["flip"]} == {-1, 1}, "need both initial directions and visible flips")
    _require(any(w["longest"] for w in witnesses), "missing longest-run counterexample")
    _require(any(w["fewer"] for w in witnesses), "missing run-count counterexample")
    _require(any(w["weighted"] for w in witnesses), "missing copy-weighting counterexample")
    _require(any(w["color_majority"] for w in witnesses), "missing color-majority counterexample")
    _require(any(w["largest"] for w in witnesses), "missing largest-class counterexample")
    max_train = max(v["count"] for f in training for v in f["classes"].values())
    _require(all(max(v["count"] for v in f["classes"].values()) > max_train for f in testing),
             "test must extend the demonstrated rank depth")


def _fingerprint(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test():
    """Bounded semantic, counterfactual, API, and 42-seed diversity checks."""
    shapes, profiles, palettes, layouts, outputs, run_profiles = (set() for _ in range(6))
    class_counts, lengths, counts, leaders, signs = (set() for _ in range(5))
    edge = flips = mid_dropout = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                facts = _analyze(a)
                cls = facts["classes"]
                stages = _stages(cls)
                shapes.add(a.shape)
                outputs.add(pair["output"].shape)
                profile = tuple((s, v["count"]) for s, v in sorted(cls.items()))
                profiles.add((split, profile))
                palettes.add((facts["background"], tuple((s, v["color"]) for s, v in sorted(cls.items()))))
                layouts.add(tuple((s, cells) for s, _, cells in facts["squares"]))
                run_profiles.add(tuple((s, tuple(sorted(v["runs"]))) for s, v in sorted(cls.items())))
                class_counts.add(len(cls))
                counts.update(v["count"] for v in cls.values())
                lengths.update(n for v in cls.values() for _, n in v["runs"])
                maximum_count = max(v["count"] for v in cls.values())
                leaders.update(list(sorted(cls)).index(s) for s, v in cls.items() if v["count"] == maximum_count)
                signs.update(1 if s["vote"] > 0 else -1 for s in stages)
                flips += _witnesses(cls)["flip"]
                mid_dropout += any(v["count"] < min(cls[min(cls)]["count"], cls[max(cls)]["count"])
                                   for s, v in cls.items() if s not in (min(cls), max(cls)))
                edge += any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                            for _, cells in _components(a, facts["background"]) for r, c in cells)
                _require(np.array_equal(solve(a.tolist()), pair["output"]), "list input solver failure")
                # Padding/translation, and a global palette permutation, must
                # preserve the rule. Use only visible grids to transform them.
                if seed < 4 and split == "test":
                    permutation = np.roll(np.arange(10), 1 + seed)
                    _require(np.array_equal(solve(permutation[a]), permutation[pair["output"]]), "palette equivariance")
                    if a.shape[0] < 30 and a.shape[1] < 30:
                        padded = np.full((a.shape[0] + 1, a.shape[1] + 1), facts["background"], dtype=np.int8)
                        padded[1:, 1:] = a
                        _require(np.array_equal(solve(padded), pair["output"]), "position/padding dependence")
    _require(len(fingerprints) == 42, "seed collisions")
    _require(len(shapes) >= 45 and len(profiles) >= 70 and len(layouts) >= 180, "structural diversity")
    _require(len(palettes) >= 50 and len(run_profiles) >= 180 and len(outputs) >= 15, "palette/stroke/output diversity")
    _require(class_counts == {2, 3, 4} and counts == {1, 2, 3, 4, 5}, "class/count coverage")
    _require(lengths == set(range(2, 9)) and signs == {-1, 1}, "stroke coverage")
    _require({0, 1, 2}.issubset(leaders) and mid_dropout >= 10, "inventory exhaustion coverage")
    _require(edge >= 80 and flips >= 84, "edge/flip coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(8128 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (2.5, 1), (4, True), (None, 1), (4, None)):
        try:
            generate_puzzle(7, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_fingerprint(generate_puzzle(99, 2, 1)) == _fingerprint(generate_puzzle(99, 2, 1)), "nondeterminism")
    _hand_cases()


def _hand_cases():
    """Known numeric answers, plus edits isolating each inherited mechanism."""
    def scene(a_lengths=(4, 4), b_length=7, extra_a=False):
        a = np.zeros((28, 30), dtype=np.int8)
        a[0:2, 0:2] = 2
        for r, c in ((0, 5), (0, 10)):
            a[r:r + 3, c:c + 3] = 3
        for r, c in ((8, 0), (8, 6), (8, 12)):
            a[r:r + 4, c:c + 4] = 4
        for (r, c), length in zip(((0, 18), (6, 18)), a_lengths):
            for i in range(length):
                a[r + i, c + i] = 2
        for i in range(b_length):
            a[14 + i, 28 - i] = 3
        for i in range(3):
            a[23 + i, i] = 4
        if extra_a:
            a[26:28, 8:10] = 2
        return a

    def digits(rows):
        return np.array([[int(c) for c in row] for row in rows.split()], dtype=np.int8)

    original = scene()
    expected = digits("223404333044440 223404333044440 333404333044440 444404444044440 000000000000000")
    _require(np.array_equal(solve(original), expected), "hand-derived rank vote failed")
    # A single extra complete square (including a boundary square) keeps the
    # class active at rank 2; the signatures are identical to the first scene.
    inventory_edit = scene(extra_a=True)
    expected_inventory = digits("223402234044440 223402234044440 333403334044440 444404444044440 000000000000000")
    _require(np.array_equal(solve(inventory_edit), expected_inventory), "inventory eligibility failed")
    # Keep every square identical and change only stroke lengths: 7 versus 8
    # flips rank 1 while preserving every output dimension and multiplicity.
    vote_edit = scene(a_lengths=(2, 2), b_length=8)
    expected_vote = digits("432204333044440 432204333044440 433304333044440 444404444044440 000000000000000")
    _require(np.array_equal(solve(vote_edit), expected_vote), "stroke population failed")
    # Tied eligible totals are rejected explicitly, never broken by color.
    try:
        solve(scene(a_lengths=(2, 2)))
    except AssertionError:
        pass
    else:
        raise AssertionError("tied vote accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal explicit counts, semantic witnesses)")
