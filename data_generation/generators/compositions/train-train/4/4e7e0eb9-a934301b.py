"""Folded evidence: mirror-coordinate union followed by a shared count test.

Only the standard library and NumPy are used. The parser/solver requires no
generation settings, seed, training examples, or stored grids.
"""

from __future__ import annotations

import numpy as np


SHAPES = tuple((h, w) for h in range(2, 8) for w in range(2, 8) if h * w <= 30)
RESERVED_SHAPES = ((2, 5), (3, 5), (2, 6), (3, 6), (4, 5), (4, 6))
CONTRAST_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3),
                   (2, 4), (4, 2), (3, 4), (4, 3))
KEEP_KINDS = ("blank_pair", "agree", "unilateral", "single0", "single1")
ERASE_KINDS = ("disagree", "single2", "crowded", "overlap")
ALL_KINDS = KEEP_KINDS + ERASE_KINDS


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected an integer 2D grid, dimensions 1..30, colors 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _axis(grid):
    """Return a vertical-axis view, and the visible color convention."""
    candidates = []
    for orientation, canvas in (("v", grid), ("h", grid.T)):
        width = canvas.shape[1]
        if width < 7 or width % 2 != 1:
            continue
        mid = width // 2
        marker = int(canvas[0, mid])
        background = int(canvas[0, mid - 1])
        if (marker != background and np.all(canvas[:, mid] == marker)
                and np.all(canvas[:, mid - 1] == background)
                and np.all(canvas[:, mid + 1] == background)):
            candidates.append((orientation, canvas, background, marker))
    if len(candidates) != 1:
        raise ValueError("expected exactly one central axis with blank adjacent stripes")
    return candidates[0]


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < height and 0 <= xx < width
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    orientation, canvas, background, marker = _axis(grid)
    height, width = canvas.shape
    mid = width // 2
    mask = canvas != background
    mask[:, mid] = False
    orbits = {}
    for cells in _components(mask):
        ys, xs = zip(*cells)
        r, c = min(ys), min(xs)
        h, w = max(ys) - r + 1, max(xs) - c + 1
        if not (2 <= h <= 7 and 2 <= w <= 7 and h * w <= 30):
            raise ValueError("invalid rectangle dimensions")
        region = canvas[r:r + h, c:c + w]
        if len(cells) != h * w or np.any(region == background):
            raise ValueError("components must be completely filled rectangles")
        carriers = set(int(v) for v in np.unique(region)) - {marker}
        if len(carriers) != 1:
            raise ValueError("an impression needs one visible carrier color")
        carrier = carriers.pop()
        if c + w <= mid - 1:
            side, canonical_c, folded = 0, c, region
        elif c >= mid + 2:
            side, canonical_c, folded = 1, width - c - w, region[:, ::-1]
        else:
            raise ValueError("rectangles must avoid the axis and adjacent stripes")
        box = (r, canonical_c, h, w)
        if box not in orbits:
            orbits[box] = {"box": box, "carrier": carrier, "masks": [None, None]}
        orbit = orbits[box]
        if orbit["carrier"] != carrier or orbit["masks"][side] is not None:
            raise ValueError("inconsistent reflected impressions")
        orbit["masks"][side] = frozenset(
            (int(y), int(x)) for y, x in zip(*np.where(folded == marker)))
    objects = [orbits[box] for box in sorted(orbits)]
    if not objects:
        raise ValueError("input must contain at least one logical rectangle")
    for i, obj in enumerate(objects):
        r, c, h, w = obj["box"]
        for other in objects[i + 1:]:
            rr, cc, hh, ww = other["box"]
            if not (r + h < rr or rr + hh < r or c + w < cc or cc + ww < c):
                raise ValueError("folded footprints must have a blank-cell separation")
    return {"orientation": orientation, "canvas": canvas,
            "background": background, "marker": marker, "orbits": objects}


def _union(orbit):
    left, right = orbit["masks"]
    return (left or frozenset()) | (right or frozenset())


def _render(info, mode="rule"):
    """Reconstruct from parsed input; alternate modes are validator ablations."""
    canvas = info["canvas"]
    height, width = canvas.shape
    background, marker = info["background"], info["marker"]
    result = np.full((height, width), background, dtype=np.int8)
    result[:, width // 2] = marker
    for orbit in info["orbits"]:
        r, c, h, w = orbit["box"]
        left, right = orbit["masks"]
        marks = _union(orbit)
        count = len(marks)
        if mode == "no_reflection":
            # The wrong rule treats raw local offsets on opposite sides as
            # equal, rather than identifying them by the indicated reflection.
            raw_right = frozenset((y, w - 1 - x) for y, x in (right or ()))
            marks = (left or frozenset()) | raw_right
            count = len(marks)
        elif mode == "physical_count":
            count = sum(len(m) for m in (left, right) if m is not None)
        if mode != "no_threshold" and count >= 2:
            continue
        block = np.full((h, w), orbit["carrier"], dtype=np.int8)
        for y, x in marks:
            block[y, x] = marker
        result[r:r + h, c:c + w] = block
        far_c = width - c - w
        result[r:r + h, far_c:far_c + w] = block[:, ::-1]
    return result if info["orientation"] == "v" else result.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from a single input, without modifying that input."""
    return _render(_parse(input_grid))


def _independent_filter(info):
    """B's per-visible-component rule, without mirror union or completion."""
    result = info["canvas"].copy()
    width = result.shape[1]
    for orbit in info["orbits"]:
        r, c, h, w = orbit["box"]
        for side, marks in enumerate(orbit["masks"]):
            if marks is not None and len(marks) >= 2:
                x = c if side == 0 else width - c - w
                result[r:r + h, x:x + w] = info["background"]
    return result if info["orientation"] == "v" else result.T.copy()


def _count_arg(name, value, lower, upper):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _sample_marks(rng, h, w, n):
    if n == 0:
        return frozenset()
    cells = [(r, c) for r in range(h) for c in range(w)]
    pattern = int(rng.integers(4))
    if n == 1:
        if pattern == 0:
            cells = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        elif pattern == 1:
            cells = [(r, c) for r, c in cells if r in (0, h - 1) or c in (0, w - 1)]
        elif pattern == 2 and h > 2 and w > 2:
            cells = [(r, c) for r, c in cells if 0 < r < h - 1 and 0 < c < w - 1]
        return frozenset((_choose(rng, cells),))
    start = _choose(rng, cells)
    rest = [p for p in cells if p != start]
    rng.shuffle(rest)
    if pattern in (0, 1):
        rest.sort(key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]),
                  reverse=pattern == 1)
        return frozenset([start] + rest[:n - 1])
    return frozenset(_choose_subset(rng, cells, n))


def _choose_subset(rng, items, n):
    return [items[int(i)] for i in rng.choice(len(items), size=n, replace=False)]


def _masks(rng, kind, h, w):
    # Corner positions ensure asymmetric reflection witnesses on either axis.
    corner = (int(rng.choice([0, h - 1])), int(rng.choice([0, w - 1])))
    p = frozenset((corner,))
    empty = frozenset()
    if kind == "blank_pair":
        masks = [empty, empty]
    elif kind == "agree":
        masks = [p, p]
    elif kind == "disagree":
        q = frozenset(((corner[0], w - 1 - corner[1]),))
        masks = [p, q]
    elif kind == "unilateral":
        masks = [p, empty]
    elif kind.startswith("single"):
        n = int(kind[-1])
        masks = [_sample_marks(rng, h, w, n) if n != 1 else p, None]
    elif kind == "crowded":
        cap = min(5, h * w - 1)
        masks = [_sample_marks(rng, h, w, int(rng.integers(2, cap + 1))),
                 _sample_marks(rng, h, w, int(rng.integers(0, cap + 1)))]
    elif kind == "overlap":
        cells = [(r, c) for r in range(h) for c in range(w)]
        cap = min(4, h * w - 1)
        n0, n1 = (int(rng.integers(2, cap + 1)) for _ in range(2))
        remaining = [p for p in cells if p != corner]
        masks = [frozenset([corner] + _choose_subset(rng, remaining, n0 - 1)),
                 frozenset([corner] + _choose_subset(rng, remaining, n1 - 1))]
    else:
        raise ValueError("unknown sampling kind")
    if rng.random() < 0.5:
        masks.reverse()
    return masks


def _place(rng, height, capacity, shapes):
    """Pack logical footprints into one half, with no hidden regular slots."""
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0] * shapes[i][1])
    prefer_edge = rng.random() < 0.75
    for _ in range(12):
        occupied = np.zeros((height, capacity), dtype=bool)
        positions = [None] * len(shapes)
        for depth, i in enumerate(order):
            h, w = shapes[i]
            candidates = [(r, c) for r in range(height - h + 1)
                          for c in range(capacity - w + 1)]
            rng.shuffle(candidates)
            if depth == 0 and prefer_edge:
                candidates.sort(key=lambda p: not (
                    p[0] in (0, height - h) or p[1] == 0))
            position = None
            for r, c in candidates:
                if not np.any(occupied[max(0, r - 1):min(height, r + h + 1),
                                       max(0, c - 1):min(capacity, c + w + 1)]):
                    position = (r, c)
                    occupied[r:r + h, c:c + w] = True
                    break
            if position is None:
                break
            positions[i] = position
        else:
            return positions
    return None


def _pair(rng, index, target, settings):
    reserved = settings["reserved"]
    excluded = {reserved, reserved[::-1]}
    available = [s for s in SHAPES if target or s not in excluded]
    witness = not target and index < 2
    orientation = (settings["first_axis"] if index == 0 else
                   ("h" if settings["first_axis"] == "v" else "v")) if witness else _choose(rng, ("v", "h"))
    if witness:
        kinds = (["agree", "disagree", "single0", "single1"] if index == 0 else
                 ["agree", "disagree", "unilateral", "single2"])
        kinds += [_choose(rng, ALL_KINDS) for _ in range(int(rng.integers(0, 3)))]
    else:
        pool = _choose(rng, (KEEP_KINDS, ERASE_KINDS, ALL_KINDS, ALL_KINDS))
        kinds = [_choose(rng, pool) for _ in range(int(rng.integers(1, 8)))]
    if settings["fixed_roles"] is None:
        background, marker = (int(v) for v in rng.choice(10, size=2, replace=False))
    else:
        background, marker = settings["fixed_roles"]
    carrier_pool = [v for v in settings["carrier_order"] if v not in (background, marker)]
    vocabulary = _choose_subset(rng, carrier_pool, int(rng.integers(1, min(4, len(carrier_pool)) + 1)))
    weights = np.array([(h * w) ** -1.7 for h, w in available], dtype=float)
    weights /= weights.sum()
    for _ in range(100):
        height, half = int(rng.integers(14, 31)), int(rng.integers(8, 15))
        shapes = [available[int(rng.choice(len(available), p=weights))] for _ in kinds]
        if witness:
            shapes[0] = shapes[1] = settings["contrast"]
        if target:
            shapes[int(rng.integers(len(shapes)))] = reserved if rng.random() < 0.5 else reserved[::-1]
        positions = _place(rng, height, half - 1, shapes)
        if positions is not None:
            break
    else:
        raise RuntimeError("could not pack logical rectangles")
    width = 2 * half + 1
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[:, half] = marker
    out[:, half] = marker
    contrast_color = _choose(rng, vocabulary)
    for i, (kind, (h, w), (r, c)) in enumerate(zip(kinds, shapes, positions)):
        carrier = contrast_color if witness and i in (0, 1) else _choose(rng, vocabulary)
        masks = _masks(rng, kind, h, w)
        # Direct generation from the planned folded masks. No parsing/solve.
        for side, marks in enumerate(masks):
            if marks is None:
                continue
            patch = np.full((h, w), carrier, dtype=np.int8)
            for y, x in marks:
                patch[y, x] = marker
            x0 = c if side == 0 else width - c - w
            inp[r:r + h, x0:x0 + w] = patch if side == 0 else patch[:, ::-1]
        combined = set()
        for marks in masks:
            if marks is not None:
                combined.update(marks)
        if len(combined) <= 1:
            patch = np.full((h, w), carrier, dtype=np.int8)
            for y, x in combined:
                patch[y, x] = marker
            out[r:r + h, c:c + w] = patch
            x0 = width - c - w
            out[r:r + h, x0:x0 + w] = patch[:, ::-1]
    if orientation == "h":
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a deterministic puzzle; explicit counts are strictly validated."""
    num_train = _count_arg("num_train", num_train, 2, 10)
    num_test = _count_arg("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    fixed_roles = tuple(int(v) for v in rng.choice(10, size=2, replace=False))
    settings = {"reserved": _choose(rng, RESERVED_SHAPES),
                "contrast": _choose(rng, CONTRAST_SHAPES),
                "first_axis": _choose(rng, ("v", "h")),
                "fixed_roles": fixed_roles if rng.random() < 0.6 else None,
                "carrier_order": tuple(int(v) for v in rng.permutation(10))}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(40):
                pair = _pair(rng, index, split == "test", settings)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("could not generate a distinct input")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(info):
    found = set()
    contrasts = {}
    for orbit in info["orbits"]:
        masks = orbit["masks"]
        count = len(_union(orbit))
        present = [m for m in masks if m is not None]
        if len(present) == 1:
            found.add("single" + str(count))
        elif sorted(len(m) for m in present) == [0, 1]:
            found.add("unilateral")
        if len(present) == 2 and all(len(m) == 1 for m in present):
            kind = "agree" if count == 1 else "disagree"
            found.add(kind)
            shape_color = orbit["box"][2:] + (orbit["carrier"],)
            contrasts.setdefault(shape_color, set()).add(kind)
    found.add("contrast") if any(v == {"agree", "disagree"} for v in contrasts.values()) else None
    return found


def validate_generated_puzzle(puzzle):
    """Check actual grids and family witnesses, without seed or plan metadata."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {"train": [], "test": []}
    ablations = {"no_reflection": False, "no_threshold": False,
                 "physical_count": False, "independent_filter": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            expected = _render(info)
            assert np.array_equal(out, expected), "incorrect folded evidence output"
            oriented = out if info["orientation"] == "v" else out.T
            assert np.array_equal(oriented, oriented[:, ::-1]), "output must be mirror symmetric"
            mid = oriented.shape[1] // 2
            assert np.array_equal(oriented[:, mid], info["canvas"][:, mid])
            for orbit in info["orbits"]:
                r, c, h, w = orbit["box"]
                patch = oriented[r:r + h, c:c + w]
                marks = _union(orbit)
                if len(marks) >= 2:
                    assert np.all(patch == info["background"])
                else:
                    assert np.count_nonzero(patch == info["marker"]) == len(marks)
                    assert np.all((patch == info["marker"]) | (patch == orbit["carrier"]))
            infos[split].append(info)
            if split == "train":
                for mode in ablations:
                    wrong = _independent_filter(info) if mode == "independent_filter" else _render(info, mode)
                    ablations[mode] |= not np.array_equal(wrong, out)
    assert {i["orientation"] for i in infos["train"]} == {"v", "h"}
    all_witnesses = set().union(*(_witnesses(i) for i in infos["train"]))
    assert {"single0", "single1", "single2", "unilateral", "agree", "disagree"} <= all_witnesses
    for axis in ("v", "h"):
        assert any(i["orientation"] == axis and "contrast" in _witnesses(i)
                   for i in infos["train"]), "both axes need same-size opposing fates"
    assert all(ablations.values()), f"missing counterfactual witness: {ablations}"
    train_shapes = {o["box"][2:] for i in infos["train"] for o in i["orbits"]}
    for info in infos["test"]:
        assert any(max(o["box"][2:]) >= 5 and o["box"][2:] not in train_shapes
                   and o["box"][2:][::-1] not in train_shapes for o in info["orbits"]), "missing geometry shift"


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _literal_checks():
    # Hand-specified 11x11 example: agree; disagree; singleton 1; singleton 2.
    inp = np.array([[int(c) for c in row] for row in (
        "83300800338", "33300800333", "00000800000",
        "83300800833", "33300800333", "00000800000",
        "83300800000", "33300800000", "00000800000",
        "88300800000", "33300800000")], dtype=np.int8)
    out = np.array([[int(c) for c in row] for row in (
        "83300800338", "33300800333", "00000800000",
        "00000800000", "00000800000", "00000800000",
        "83300800338", "33300800333", "00000800000",
        "00000800000", "00000800000")], dtype=np.int8)
    assert np.array_equal(solve(inp), out)
    assert np.array_equal(solve(inp.T), out.T)
    # Removing one of the disagreeing marks changes deletion into repair.
    repaired = inp.copy()
    repaired[3, 8] = 3
    expected = out.copy()
    expected[3:5, :3] = np.array([[8, 3, 3], [3, 3, 3]])
    expected[3:5, 8:] = np.array([[3, 3, 8], [3, 3, 3]])
    assert np.array_equal(solve(repaired), expected)


def self_test():
    """Bounded 42-seed semantic/diversity audit; return compact measurements."""
    _literal_checks()
    signatures, dimensions, shapes, inventories, palettes, structures = (set() for _ in range(6))
    axes, union_counts, support_counts, mixtures = (set() for _ in range(4))
    observed = {k: 0 for k in ("edge", "all_keep", "all_erase", "overlap",
                                "role_changes", "smaller_test")}
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in puzzle[s]))
        roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                info = _parse(inp)
                roles.add((info["background"], info["marker"]))
                axes.add(info["orientation"])
                dimensions.add(inp.shape)
                inventories.add(len(info["orbits"]))
                palettes.add(tuple(int(v) for v in np.unique(inp)))
                names = {}
                normalized = tuple(names.setdefault(int(v), len(names)) for v in inp.flat)
                structures.add((inp.shape, normalized))
                counts = [len(_union(o)) for o in info["orbits"]]
                union_counts.update(counts)
                mixtures.add((sum(c <= 1 for c in counts), sum(c >= 2 for c in counts)))
                observed["all_keep"] += max(counts) <= 1
                observed["all_erase"] += min(counts) >= 2
                for orbit in info["orbits"]:
                    r, c, h, w = orbit["box"]
                    shapes.add((h, w))
                    support_counts.add(sum(m is not None for m in orbit["masks"]))
                    observed["edge"] += r == 0 or r + h == info["canvas"].shape[0] or c == 0
                    a, b = orbit["masks"]
                    observed["overlap"] += a is not None and b is not None and bool(a & b) and len(a | b) >= 2
                if seed < 4:
                    permutation = np.roll(np.arange(10, dtype=np.int8), seed + 1)
                    assert np.array_equal(solve(permutation[inp]), permutation[out])
                    assert np.array_equal(solve(inp.T), out.T)
        observed["role_changes"] += len(roles) > 1
        observed["smaller_test"] += puzzle["test"][0]["input"].size < max(p["input"].size for p in puzzle["train"])
    assert len(signatures) == 42
    assert len(dimensions) >= 40 and len(shapes) >= 25 and len(palettes) >= 30
    assert inventories == set(range(1, 8)) and support_counts == {1, 2} and axes == {"v", "h"}
    assert {0, 1, 2, 3, 4} <= union_counts and len(mixtures) >= 15
    assert len(structures) >= 190 and all(v > 0 for v in observed.values()), observed
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + i, nt, ne)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
    assert _same(generate_puzzle(271828), generate_puzzle(271828))
    assert _same(generate_puzzle(314159, 10, 4), generate_puzzle(314159, 10, 4))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt}/{ne}")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(dimensions), "rectangle_shapes": len(shapes),
            "inventories": sorted(inventories), "palettes": len(palettes),
            "normalized_structures": len(structures), "outcome_mixtures": len(mixtures),
            "union_counts": sorted(union_counts), "observed": observed}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
