"""Mirror-repaired square inventories, serialized as occurrence-rank nests.

Only standard-library modules and NumPy are used. No file I/O is performed by
generation, solving, validation, or self_test. Black (0) always means erasure.
"""
from functools import lru_cache
from hashlib import sha256

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


@lru_cache(maxsize=64)
def _orbits(h, w, a, b):
    seen, groups = set(), []
    for r in range(h):
        rows = sorted({r, a - r} & set(range(h)))
        for c in range(w):
            if (r, c) in seen:
                continue
            cols = sorted({c, b - c} & set(range(w)))
            group = tuple((y, x) for y in rows for x in cols)
            seen.update(group)
            groups.append(group)
    return tuple(groups)


def _axis(grid, bg, dimension):
    """Largest reflected overlap with no visible disagreement; reject ties.

    A candidate must compare at least one visible nonbackground pair. This
    excludes folds supported only by an empty border. Fixed points do not vote.
    """
    x = grid if dimension == 0 else grid.T
    n = x.shape[0]
    candidates = []
    for fold in range(1, 2 * n - 2):
        left = np.arange(max(0, fold - n + 1), (fold + 1) // 2)
        if not len(left):
            continue
        u, v = x[left], x[fold - left]
        visible = (u != 0) & (v != 0)
        if (np.any(visible & (u != bg))
                and not np.any(visible & (u != v))):
            overlap = n - abs(fold - (n - 1))
            candidates.append((overlap, fold))
    if not candidates:
        raise ValueError("no supported reflection fold")
    best = max(length for length, fold in candidates)
    winners = [fold for length, fold in candidates if length == best]
    if len(winners) != 1:
        raise ValueError("ambiguous maximal-overlap reflection fold")
    return winners[0]


def _restore(grid, bg):
    a, b = (_axis(grid, bg, d) for d in (0, 1))
    result = grid.copy()
    for group in _orbits(*grid.shape, a, b):
        witnesses = {int(grid[p]) for p in group if grid[p] != 0}
        if len(witnesses) != 1:
            raise ValueError("a reflection orbit has no unanimous witness")
        color = next(iter(witnesses))
        for p in group:
            if grid[p] == 0:
                result[p] = color
    return result, (a, b)


def _components(grid, excluded):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in excluded:
                continue
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and grid[yy, xx] == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            ys, xs = zip(*cells)
            box = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            result.append((color, box, len(cells)))
    return result


def _inventory(grid, bg):
    components = _components(grid, {bg})
    classes, squares = {}, []
    for color, box, area in components:
        r0, r1, c0, c1 = box
        side = r1 - r0
        if side >= 2 and side == c1 - c0 and area == side * side:
            if side in classes and classes[side][0] != color:
                raise ValueError("one size class has two colors")
            classes.setdefault(side, [color, 0])[1] += 1
            squares.append((side, color, box))
    if (not 2 <= len(classes) <= 4 or not set(classes) <= {2, 3, 4, 5}
            or len({color for color, n in classes.values()}) != len(classes)):
        raise ValueError("invalid square size/color classes")
    by_color = {color: s for s, (color, n) in classes.items()}
    others = set(int(c) for c in np.unique(grid)) - {bg} - set(by_color)
    if len(others) != 1:
        raise ValueError("expected one sparse registration/noise color")
    noise = others.pop()
    fragments = []
    for color, box, area in components:
        if color == noise:
            continue
        r0, r1, c0, c1 = box
        hh, ww = r1 - r0, c1 - c0
        if hh == ww and hh == by_color[color] and area == hh * ww:
            continue
        if not (hh != ww and max(hh, ww) == by_color[color]
                and area == hh * ww
                and (r0 == 0 or c0 == 0 or r1 == grid.shape[0] or c1 == grid.shape[1])):
            raise ValueError("a payload component is neither a square nor a clipped fragment")
        fragments.append((color, box))
    m = grid == noise
    if np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]):
        raise ValueError("noise contains a solid 2x2")
    return classes, squares, fragments, noise


def _render(classes, bg):
    active_classes = {s: (color, n) for s, (color, n) in classes.items() if n > 0}
    if not active_classes:
        return np.full((1, 1), bg, dtype=np.int8)
    ranks = [sorted((s for s, (color, n) in active_classes.items() if n >= rank), reverse=True)
             for rank in range(1, max(n for color, n in active_classes.values()) + 1)]
    width = sum(sides[0] + 1 for sides in ranks)
    if width > 30:
        raise ValueError("rank serialization exceeds 30 columns")
    out = np.full((max(active_classes) + 1, width), bg, dtype=np.int8)
    cursor = 0
    for sides in ranks:
        for side in sides:
            out[:side, cursor:cursor + side] = active_classes[side][0]
        cursor += sides[0] + 1
    return out


def _analyze(input_grid):
    grid = _grid(input_grid)
    hist = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    if bg == 0 or sum(hist == hist[bg]) != 1 or not np.any(grid == 0):
        raise ValueError("input needs a unique nonblack dominant background and black erasures")
    restored, axes = _restore(grid, bg)
    classes, squares, fragments, noise = _inventory(restored, bg)
    return {"background": bg, "axes": axes, "restored": restored,
            "classes": classes, "squares": squares, "fragments": fragments, "noise": noise}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer everything from this grid; never uses a seed or generation state."""
    facts = _analyze(input_grid)
    return _render(facts["classes"], facts["background"])


def _apart(p, q):
    # Half-open rectangles must have at least one background cell between them.
    return p[1] < q[0] or q[1] < p[0] or p[3] < q[2] or q[3] < p[2]


@lru_cache(maxsize=96)
def _placements(h, w, a, b, side, count):
    result = []
    for r in range(1 - side, min(h - 1, (a - side + 1) // 2) + 1):
        for c in range(1 - side, min(w - 1, (b - side + 1) // 2) + 1):
            origins = sorted({(y, x) for y in (r, a - r - side + 1)
                              for x in (c, b - c - side + 1)})
            visible, full = [], []
            valid = True
            for y, x in origins:
                box = max(0, y), min(h, y + side), max(0, x), min(w, x + side)
                if box[0] >= box[1] or box[2] >= box[3]:
                    continue
                hh, ww = box[1] - box[0], box[3] - box[2]
                if hh == ww == side:
                    full.append(box)
                elif hh == ww or max(hh, ww) != side:
                    valid = False  # No ambiguous square-looking/corner crops.
                    break
                visible.append(box)
            if (valid and visible and len(full) == count
                    and all(_apart(p, q) for i, p in enumerate(visible) for q in visible[:i])):
                result.append((tuple(visible), tuple(full)))
    return tuple(result)


def _free(reserved, boxes):
    return all(not reserved[r0:r1, c0:c1].any() for r0, r1, c0, c1 in boxes)


def _paint(grid, reserved, boxes, color):
    h, w = grid.shape
    for r0, r1, c0, c1 in boxes:
        grid[r0:r1, c0:c1] = color
        reserved[max(0, r0 - 1):min(h, r1 + 1), max(0, c0 - 1):min(w, c1 + 1)] = True


def _profile(classes):
    return tuple((s, n) for s, (color, n) in sorted(classes.items()))


def _recipes(rng, style):
    k = int(rng.choice((2, 3, 4), p=(0.30, 0.55, 0.15)))
    if style == "large":
        k = 3
    sides = sorted(int(x) for x in rng.choice(np.arange(2, 6), k, replace=False))
    primary = sides[0] if style == "small" else (sides[-1] if style == "large" else int(rng.choice(sides)))
    recipes = {}
    for s in sides:
        if s == primary:
            recipes[s] = (4,) if style != "free" else ((4,), (4, 1), (4, 2))[int(rng.choice(3, p=(0.65, 0.15, 0.20)))]
        elif style != "free":
            recipes[s] = (int(rng.choice((1, 2), p=(0.25, 0.75))),)
        else:
            recipes[s] = ((1,), (2,), (4,), (2, 1), (2, 2))[int(rng.choice(5, p=(0.15, 0.40, 0.20, 0.15, 0.10)))]
    return recipes, primary


def _noise(rng, grid, bg, color, axes):
    groups = _orbits(*grid.shape, *axes)
    desired = int(rng.integers(8, 21))
    used = 0
    for index in rng.permutation(len(groups)):
        group = groups[int(index)]
        if len(group) < 2 or any(grid[p] != bg for p in group):
            continue
        for p in group:
            grid[p] = color
        m = grid == color
        if np.any(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]):
            for p in group:
                grid[p] = bg
        else:
            used += 1
            if used == desired:
                break
    return used >= 6


def _erase(rng, clean, axes, primary_boxes, hard):
    h, w = clean.shape
    mask = np.zeros((h, w), dtype=bool)
    protected = np.zeros_like(mask)
    number = 3 if hard else int(rng.integers(1, 4))
    chosen = rng.permutation(4)[:number]
    irregular = bool(rng.integers(2))
    for i in chosen:
        r0, r1, c0, c1 = primary_boxes[int(i)]
        protected[r0:r1, c0:c1] = True
        top, bottom, left, right = (int(x) for x in rng.integers(0, 3, size=4))
        # Widen the unknown region so its outline does not report square size.
        left += 1
        if irregular:
            for y in range(max(0, r0 - top), min(h, r1 + bottom)):
                extra = (y - r0) % 2
                mask[y, max(0, c0 - left):min(w, c1 + right + extra)] = True
        else:
            mask[max(0, r0 - top):min(h, r1 + bottom),
                 max(0, c0 - left):min(w, c1 + right)] = True
    mask |= protected
    for group in _orbits(h, w, *axes):
        if all(mask[p] for p in group):
            options = [p for p in group if not protected[p]]
            if not options:
                return None
            mask[options[int(rng.integers(len(options)))]] = False
    inp = clean.copy()
    inp[mask] = 0
    return inp


def _make_pair(rng, style, hard, excluded_profiles):
    for attempt in range(180):
        recipes, primary = _recipes(rng, style)
        colors = [int(x) for x in rng.permutation(np.arange(1, 10))]
        bg, noise = colors[:2]
        classes = {s: [color, sum(recipes[s])] for s, color in zip(sorted(recipes), colors[2:])}
        if _profile(classes) in excluded_profiles:
            continue
        try:
            output = _render(classes, bg)
        except ValueError:
            continue
        h, w = (int(x) for x in rng.integers(20, 31, size=2))
        if rng.random() < 0.22:
            w = h
        dr, dc = (int(x) for x in rng.integers(-3, 4, size=2))
        a, b = h - 1 + dr, w - 1 + dc
        clean = np.full((h, w), bg, dtype=np.int8)
        reserved = np.zeros((h, w), dtype=bool)
        items = [(s, copies, j) for s, recipe in recipes.items() for j, copies in enumerate(recipe)]
        items.sort(key=lambda x: (-x[0], -x[1]))
        primary_boxes = None
        success = True
        prefer_fragments = rng.random() < 0.45
        for s, copies, j in items:
            choices = [p for p in _placements(h, w, a, b, s, copies) if _free(reserved, p[0])]
            if not choices:
                success = False
                break
            if prefer_fragments:
                clipped = [p for p in choices if len(p[0]) > len(p[1])]
                choices = clipped or choices
            boxes, full = choices[int(rng.integers(len(choices)))]
            _paint(clean, reserved, boxes, classes[s][0])
            if s == primary and j == 0:
                primary_boxes = full
        if not success:
            continue
        # An optional reflected set of visibly clipped fragments, with no full
        # squares, reinforces that boundary area does not add to the ledger.
        if rng.random() < 0.45:
            s = int(rng.choice(sorted(classes)))
            choices = [p for p in _placements(h, w, a, b, s, 0) if _free(reserved, p[0])]
            if choices:
                boxes, full = choices[int(rng.integers(len(choices)))]
                _paint(clean, reserved, boxes, classes[s][0])
        if not _noise(rng, clean, bg, noise, (a, b)):
            continue
        inp = _erase(rng, clean, (a, b), primary_boxes, hard)
        if inp is None or np.count_nonzero((inp == 0) & (clean == bg)) < 2:
            continue
        try:
            facts = _analyze(inp)
            if (facts["axes"] != (a, b) or facts["classes"] != classes
                    or not np.array_equal(facts["restored"], clean)):
                continue
            return {"input": inp, "output": output}
        except ValueError:
            continue
    raise RuntimeError("bounded square-orbit construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    styles = ["small", "large"] + ["free"] * (nt - 2)
    rng.shuffle(styles)
    puzzle, seen, training_profiles = {"train": [], "test": []}, set(), set()
    hard_demo = int(rng.integers(nt))
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for retry in range(12):
                pair = _make_pair(rng, styles[index] if split == "train" else "free",
                                  split == "test" or index == hard_demo,
                                  training_profiles if split == "test" else set())
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        training_profiles.add(_profile(_analyze(pair["input"])["classes"]))
                    break
            else:
                raise RuntimeError("distinct-input retry budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _visible_classes(inp, facts):
    classes = {s: [color, 0] for s, (color, n) in facts["classes"].items()}
    by_color = {color: s for s, (color, n) in classes.items()}
    for color, (r0, r1, c0, c1), area in _components(inp, {0, facts["background"], facts["noise"]}):
        s = by_color[color]
        if r1 - r0 == c1 - c0 == s and area == s * s:
            classes[s][1] += 1
    return classes


def _features(inp, facts):
    a, b = facts["axes"]
    h, w = inp.shape
    diagonal = diagonal_payload = 0
    payload = {color for color, n in facts["classes"].values()}
    for r, c in zip(*np.nonzero(inp == 0)):
        rr, cc = a - int(r), b - int(c)
        if (0 <= rr < h and 0 <= cc < w and inp[rr, cc] != 0
                and inp[rr, c] == 0 and inp[r, cc] == 0):
            diagonal += 1
            diagonal_payload += int(facts["restored"][r, c]) in payload
    erased = sum(np.all(inp[r0:r1, c0:c1] == 0) for s, color, (r0, r1, c0, c1) in facts["squares"])
    edges = sum(r0 == 0 or c0 == 0 or r1 == h or c1 == w
                for s, color, (r0, r1, c0, c1) in facts["squares"])
    return {"diagonal_only": int(diagonal), "diagonal_payload": int(diagonal_payload),
            "erased_squares": int(erased),
            "edge_squares": int(edges), "fragments": len(facts["fragments"])}


def _check_bands(out, classes, bg):
    """Check each output cell via threshold/size membership, not paint order."""
    cursor = 0
    _require(out.shape[0] == max(classes) + 1, "wrong bottom margin")
    for rank in range(1, max(n for color, n in classes.values()) + 1):
        active = {s: color for s, (color, n) in classes.items() if n >= rank}
        largest = max(active)
        for r in range(out.shape[0]):
            for c in range(largest):
                covering = [s for s in active if r < s and c < s]
                expected = active[min(covering)] if covering else bg
                _require(out[r, cursor + c] == expected, "wrong overwrite band or padding")
        _require(np.all(out[:, cursor + largest] == bg), "missing separator")
        cursor += largest + 1
    _require(cursor == out.shape[1] and np.all(out[-1] == bg), "wrong output extent")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, profiles = set(), {"train": set(), "test": set()}
    hard_demo = small_tail = large_tail = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts = _analyze(inp)
            bg, classes = facts["background"], facts["classes"]
            restored = facts["restored"]
            _require(np.array_equal(inp[inp != 0], restored[inp != 0]), "visible cells changed")
            _require(all(1 <= n <= 6 for color, n in classes.values()), "multiplicity outside 1..6")
            f = _features(inp, facts)
            _require(f["erased_squares"] >= 1, "no wholly erased square")
            _require(np.any((inp == 0) & (restored == bg)), "mask directly reveals square outline")
            _require(not np.array_equal(_render(_visible_classes(inp, facts), bg), out),
                     "ignoring reflection leaves answer unchanged")
            _require(np.array_equal(solve(inp), out), "wrong output")
            _check_bands(out, classes, bg)
            profiles[split].add(_profile(classes))
            if split == "train":
                hard_demo |= f["diagonal_payload"] > 0
                smallest, largest = min(classes), max(classes)
                small_tail |= classes[smallest][1] > classes[largest][1]
                large_tail |= classes[largest][1] > classes[smallest][1]
            else:
                _require(f["diagonal_payload"] > 0, "test must exercise both folds on a square")
    _require(hard_demo and small_tail and large_tail, "training lacks distinguishing demonstrations")
    _require(not (profiles["train"] & profiles["test"]), "test inventory was demonstrated verbatim")


def self_test():
    # Small direct checks distinguish a complete border square from a crop.
    border_case = np.full((10, 12), 9, dtype=np.int8)
    border_case[:2, :2] = 2
    border_case[3:6, 5:8] = 3
    border_case[7:10, 11:12] = 3
    border_case[1, 8] = 8
    inventory, squares, fragments, noise = _inventory(border_case, 9)
    _require(inventory == {2: [2, 1], 3: [3, 1]} and len(fragments) == 1,
             "border-square/fragment distinction failed")
    groups = _orbits(5, 7, 5, 4)
    _require(((0, 5),) in groups and ((0, 0), (0, 4)) in groups,
             "singleton or one-fold fringe handling failed")
    rank_case = {2: [2, 5], 3: [3, 2], 5: [5, 3]}
    band_case = _render(rank_case, 9)
    _require(band_case.shape == (6, 24), "rank threshold width failed")
    _check_bands(band_case, rank_case, 9)
    shapes, profiles, palettes, axes, layouts, signatures = (set() for _ in range(6))
    features = {"edge_squares": 0, "fragments": 0, "diagonal_only": 0}
    classes_seen, counts_seen, bg_seen = set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                facts = _analyze(inp)
                f = _features(inp, facts)
                for key in features:
                    features[key] += f[key] > 0
                shapes.add(inp.shape)
                profiles.add(_profile(facts["classes"]))
                palettes.add(tuple((s, color) for s, (color, n) in sorted(facts["classes"].items())))
                a, b = facts["axes"]
                axes.add((int(np.sign(a - inp.shape[0] + 1)), int(np.sign(b - inp.shape[1] + 1)), a % 2, b % 2))
                layouts.add(tuple((s, box) for s, color, box in facts["squares"]))
                classes_seen.add(len(facts["classes"]))
                counts_seen.update(n for color, n in facts["classes"].values())
                bg_seen.add(facts["background"])
                digest.update(bytes.fromhex(_signature(inp)))
                digest.update(bytes.fromhex(_signature(out)))
                before = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, before), "solver mutates input")
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42, "cross-seed puzzle collapse")
    _require(len(shapes) >= 35 and len(profiles) >= 35 and len(layouts) >= 180, "structural variation collapsed")
    _require(len(palettes) >= 100 and len(bg_seen) >= 8 and len(axes) >= 16, "palette/fold variation collapsed")
    _require(classes_seen == {2, 3, 4} and counts_seen == set(range(1, 7)), "inventory coverage incomplete")
    _require(all(n >= 15 for n in features.values()), "edge, fragment, or two-fold coverage missing")
    first = generate_puzzle(0)
    again = generate_puzzle(0)
    for split in first:
        for p, q in zip(first[split], again[split]):
            _require(all(np.array_equal(p[k], q[k]) for k in p), "nondeterministic seed")
    for nt, nq in ((2, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(123 + nt, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a wrong output")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "inventory_profiles": len(profiles), "payload_palettes": len(palettes),
            "fold_descriptors": len(axes), "square_layouts": len(layouts),
            "class_counts": sorted(classes_seen), "multiplicities": sorted(counts_seen),
            "background_colors": sorted(bg_seen), "pairs_with_features": features}
