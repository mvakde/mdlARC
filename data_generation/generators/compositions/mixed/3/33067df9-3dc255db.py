"""Exact-span lattice rectangles become tab-directed cargo tally frames.

Only the standard library and NumPy are used. No files are read or written.
The input reader knows the fixed parity channels and scale, never a seed.
"""
from dataclasses import dataclass, replace
from itertools import product

import numpy as np


_DIRECTIONS = ("U", "R", "D", "L")
_STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


@dataclass(frozen=True, order=True)
class Rect:
    r0: int
    r1: int
    c0: int
    c1: int
    color: int

    def cells(self):
        return tuple(product(range(self.r0, self.r1), range(self.c0, self.c1)))


@dataclass(frozen=True)
class Object:
    rect: Rect
    direction: str
    payload_color: int
    cargo: tuple


def _runs(row, bg):
    result, c = [], 0
    while c < len(row):
        color, end = int(row[c]), c + 1
        if color == bg:
            c = end
            continue
        while end < len(row) and int(row[end]) == color:
            end += 1
        result.append((c, end, color))
        c = end
    return result


def _rectangles(coarse, bg):
    """Partition maximal horizontal runs by EXACT consecutive span equality."""
    rows = [_runs(row, bg) for row in coarse]
    result = []
    for r, runs in enumerate(rows):
        previous = set(rows[r - 1]) if r else set()
        for left, right, color in runs:
            if (left, right, color) in previous:
                continue
            stop = r + 1
            while stop < len(rows) and (left, right, color) in rows[stop]:
                stop += 1
            result.append(Rect(r, stop, left, right, color))
    return sorted(result)


def _reference_rectangles(coarse, bg):
    """Cell-local check: find a sample's run, then its exact up/down extent."""
    found = set()
    height, width = coarse.shape
    for r, c in product(range(height), range(width)):
        color = int(coarse[r, c])
        if color == bg:
            continue
        left, right = c, c + 1
        while left > 0 and int(coarse[r, left - 1]) == color:
            left -= 1
        while right < width and int(coarse[r, right]) == color:
            right += 1

        def exact(y):
            return (np.all(coarse[y, left:right] == color)
                    and (left == 0 or int(coarse[y, left - 1]) != color)
                    and (right == width or int(coarse[y, right]) != color))

        top, bottom = r, r + 1
        while top > 0 and exact(top - 1):
            top -= 1
        while bottom < height and exact(bottom):
            bottom += 1
        found.add(Rect(top, bottom, left, right, color))
    return sorted(found)


def _tab(rect, direction):
    r = (rect.r0 + rect.r1 - 1) // 2
    c = (rect.c0 + rect.c1 - 1) // 2
    return {"U": (2 * rect.r0, 2 * c + 1),
            "D": (2 * rect.r1, 2 * c + 1),
            "L": (2 * r + 1, 2 * rect.c0),
            "R": (2 * r + 1, 2 * rect.c1)}[direction]


def _bounds(rect):
    # Inclusive endpoints: scale 5, two-cell outer margin, one-cell gaps.
    return 5 * rect.r0 + 2, 5 * rect.r1, 5 * rect.c0 + 2, 5 * rect.c1


def _tip(rect, direction):
    top, bottom, left, right = _bounds(rect)
    r = 5 * ((rect.r0 + rect.r1 - 1) // 2) + 3
    c = 5 * ((rect.c0 + rect.c1 - 1) // 2) + 3
    return {"U": (top - 1, c), "D": (bottom + 1, c),
            "L": (r, left - 1), "R": (r, right + 1)}[direction]


def _ray(rect, direction, quantity, shape):
    dr, dc = _STEP[direction]
    r, c = _tip(rect, direction)
    result = []
    for _ in range(quantity):
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        result.append((r, c))
    return tuple(result)


def _room(rect, direction, shape):
    r, c = _tip(rect, direction)
    return {"U": r, "D": shape[0] - 1 - r,
            "L": c, "R": shape[1] - 1 - c}[direction]


def _body(rect):
    top, bottom, left, right = _bounds(rect)
    return set(product(range(top, bottom + 1), range(left, right + 1)))


def _encode(coarse, bg, objects):
    rows, cols = coarse.shape
    inp = np.full((2 * rows + 1, 2 * cols + 1), bg, dtype=np.int8)
    inp[1::2, 1::2] = coarse
    for obj in objects:
        slot = _tab(obj.rect, obj.direction)
        if inp[slot] != bg:
            raise ValueError("two direction tabs occupy one source slot")
        inp[slot] = obj.rect.color
        for r, c in obj.cargo:
            inp[2 * r + 2, 2 * c + 2] = obj.payload_color
    return inp


def _render(objects, bg, shape, tally=True):
    """Explicit transient filled rectangles, cargo, hollowing, then export.

    The intermediate arrays stay in memory. Saved execution traces are not
    produced. Counterfactual checks may call this on deliberately wrong groups.
    """
    out = np.full(shape, bg, dtype=np.int8)
    for obj in objects:
        top, bottom, left, right = _bounds(obj.rect)
        out[top:bottom + 1, left:right + 1] = obj.rect.color
    if not tally:
        return out
    for obj in objects:
        for r, c in obj.cargo:
            out[5 * r + 3, 5 * c + 3] = obj.payload_color
    for obj in objects:
        top, bottom, left, right = _bounds(obj.rect)
        out[top + 1:bottom, left + 1:right] = bg
        out[_tip(obj.rect, obj.direction)] = obj.rect.color
    for obj in objects:
        for cell in _ray(obj.rect, obj.direction, len(obj.cargo), shape):
            out[cell] = obj.payload_color
    return out


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grid must be an integer 2D ARC grid")
    return a


def _checked_geometry(objects, shape):
    bodies, used = set(), set()
    for obj in objects:
        body = _body(obj.rect)
        if bodies & body:
            raise ValueError("decoded rectangle bodies overlap")
        bodies.update(body)
    for obj in objects:
        # Reserve the next background cell as evidence of count-based stopping.
        guard = set(_ray(obj.rect, obj.direction, len(obj.cargo) + 1, shape))
        guard.add(_tip(obj.rect, obj.direction))
        if guard & (bodies | used):
            raise ValueError("tip/ray corridors conflict")
        used.update(guard)


def _read_input(input_grid):
    inp = _array(input_grid)
    height, width = inp.shape
    if not (3 <= height <= 11 and 3 <= width <= 11 and height % 2 and width % 2):
        raise ValueError("expected a 1..5 by 1..5 parity-coded coarse lattice")
    bg = int(inp[0, 0])
    coarse = inp[1::2, 1::2].copy()
    rectangles = _rectangles(coarse, bg)
    if not rectangles:
        raise ValueError("empty rectangle code")
    objects = []
    for rect in rectangles:
        directions = [d for d in _DIRECTIONS if int(inp[_tab(rect, d)]) == rect.color]
        if len(directions) != 1:
            raise ValueError("each decoded rectangle needs one unambiguous side tab")
        cargo = tuple((r, c) for r, c in rect.cells() if inp[2 * r + 2, 2 * c + 2] != bg)
        colors = {int(inp[2 * r + 2, 2 * c + 2]) for r, c in cargo}
        if len(colors) != 1 or rect.color in colors:
            raise ValueError("each rectangle needs nonempty, uniform, distinct-color cargo")
        objects.append(Object(rect, directions[0], next(iter(colors)), cargo))
    if not np.array_equal(inp, _encode(coarse, bg, objects)):
        raise ValueError("unassigned source marks or cargo below blank labels")
    shape = (5 * coarse.shape[0] + 3, 5 * coarse.shape[1] + 3)
    _checked_geometry(objects, shape)
    return coarse, bg, objects, shape


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all labels, tabs and cargo from this input alone."""
    _, bg, objects, shape = _read_input(input_grid)
    return _render(objects, bg, shape)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _choose_directions(coarse, bg, quantities, rng, required=None, modes=None):
    rectangles = _rectangles(coarse, bg)
    shape = (5 * coarse.shape[0] + 3, 5 * coarse.shape[1] + 3)
    bodies = set().union(*(_body(rect) for rect in rectangles))
    required, modes = required or {}, modes or {}
    candidates = []
    for rect in rectangles:
        choices = []
        q = quantities[rect]
        for direction in rng.permutation(_DIRECTIONS):
            direction = str(direction)
            if rect in required and direction != required[rect]:
                continue
            room = _room(rect, direction, shape)
            if modes.get(rect) == "free" and room <= q:
                continue
            if modes.get(rect) == "clip" and not 0 < room < q:
                continue
            slot = _tab(rect, direction)
            if any(other != rect and other.color == rect.color
                   and slot in {_tab(other, d) for d in _DIRECTIONS} for other in rectangles):
                continue
            guard = set(_ray(rect, direction, q + 1, shape)) | {_tip(rect, direction)}
            if not guard & bodies:
                choices.append((direction, slot, guard))
        if not choices:
            return None
        candidates.append((rect, choices))
    candidates.sort(key=lambda item: len(item[1]))
    directions = {}
    nodes = 0

    def visit(index, occupied, slots):
        nonlocal nodes
        nodes += 1
        if nodes > 512:
            return False
        if index == len(candidates):
            return True
        rect, choices = candidates[index]
        for direction, slot, guard in choices:
            if slot in slots or guard & occupied:
                continue
            directions[rect] = direction
            if visit(index + 1, occupied | guard, slots | {slot}):
                return True
        directions.pop(rect, None)
        return False

    return directions if visit(0, set(), set()) else None


def _sample_cargo(rect, quantity, rng):
    cells = rect.cells()
    return tuple(sorted(cells[int(i)] for i in rng.choice(len(cells), quantity, replace=False)))


def _add_runs(coarse, bg, colors, rng, attempts):
    rows, cols = coarse.shape
    for _ in range(attempts):
        r0, c0 = int(rng.integers(rows)), int(rng.integers(cols))
        r1 = min(rows, r0 + int(rng.integers(1, 3)))
        c1 = int(rng.integers(c0 + 1, cols + 1))
        if np.all(coarse[r0:r1, c0:c1] == bg):
            coarse[r0:r1, c0:c1] = int(rng.choice(colors))


def _anchor_plan(rng, depth, directions=None):
    """Sample relational witnesses, not a stored coarse matrix template."""
    for _ in range(80):
        rows = int(rng.integers(max(4, depth + 1), 6))
        cols = int(rng.integers(4, 6))
        bg, frame, payload = (int(v) for v in rng.choice(10, 3, replace=False))
        horizontal = directions[0] if directions else str(rng.choice(("L", "R")))
        vertical = directions[1] if directions else str(rng.choice(("U", "D")))
        partial_row = 0 if vertical == "U" else rows - 1
        r0 = 1 if vertical == "U" else rows - depth - 1
        spans = [(left, right) for left in range(cols) for right in range(left + 2, cols + 1)
                 if (left > 0 if horizontal == "L" else right < cols)]
        left, right = spans[int(rng.integers(len(spans)))]
        other_spans = [(a, b) for a in range(cols) for b in range(a + 2, cols + 1)
                       if (a, b) != (left, right) and max(a, left) < min(b, right)]
        a, b = other_spans[int(rng.integers(len(other_spans)))]
        coarse = np.full((rows, cols), bg, dtype=np.int8)
        coarse[r0:r0 + depth, left:right] = frame
        coarse[partial_row, a:b] = frame
        main = Rect(r0, r0 + depth, left, right, frame)
        partial = Rect(partial_row, partial_row + 1, a, b, frame)
        extra_colors = [c for c in range(10) if c not in (bg, frame)]
        _add_runs(coarse, bg, extra_colors, rng, int(rng.integers(4)))
        rectangles = _rectangles(coarse, bg)
        if main not in rectangles or partial not in rectangles or len(rectangles) > 7:
            continue
        if any(r.r1 - r.r0 > max(depth, 2) for r in rectangles):
            continue
        quantities = {rect: int(rng.integers(1, min(10, len(rect.cells())) + 1)) for rect in rectangles}
        quantities[main] = 4 if depth == 2 else int(rng.integers(4, min(10, len(main.cells())) + 1))
        quantities[partial] = int(rng.integers(2, len(partial.cells()) + 1))
        chosen = _choose_directions(coarse, bg, quantities, rng,
            {main: horizontal, partial: vertical}, {main: "free", partial: "clip"})
        if chosen is None:
            continue
        objects = []
        for rect in rectangles:
            pc = payload if rect.color == frame else int(rng.choice([c for c in range(10) if c not in (bg, rect.color)]))
            objects.append(Object(rect, chosen[rect], pc, _sample_cargo(rect, quantities[rect], rng)))
        return coarse, bg, objects, main, (horizontal, vertical)
    raise RuntimeError("bounded witness construction exhausted")


def _ordinary_pair(rng):
    for _ in range(80):
        rows, cols = int(rng.integers(1, 6)), int(rng.integers(1, 6))
        bg = int(rng.integers(10))
        palette = [int(c) for c in rng.choice([c for c in range(10) if c != bg], int(rng.integers(1, 5)), replace=False)]
        coarse = np.full((rows, cols), bg, dtype=np.int8)
        _add_runs(coarse, bg, palette, rng, int(rng.integers(1, 8)))
        rectangles = _rectangles(coarse, bg)
        if not rectangles or len(rectangles) > 8 or any(r.r1 - r.r0 > 2 for r in rectangles):
            continue
        quantities = {r: int(rng.integers(1, len(r.cells()) + 1)) for r in rectangles}
        directions = _choose_directions(coarse, bg, quantities, rng)
        if directions is None:
            continue
        objects = [Object(r, directions[r], int(rng.choice([c for c in range(10) if c not in (bg, r.color)])),
                          _sample_cargo(r, quantities[r], rng)) for r in rectangles]
        return _make_pair(coarse, bg, objects)
    # A guaranteed small, count-dependent fallback, still sampled in color.
    bg, frame, payload = (int(v) for v in rng.choice(10, 3, replace=False))
    coarse = np.full((1, 2), bg, dtype=np.int8)
    coarse[0, 0] = frame
    rect = Rect(0, 1, 0, 1, frame)
    return _make_pair(coarse, bg, [Object(rect, "R", payload, ((0, 0),))])


def _make_pair(coarse, bg, objects):
    inp = _encode(coarse, bg, objects)
    shape = (5 * coarse.shape[0] + 3, 5 * coarse.shape[1] + 3)
    out = _render(objects, bg, shape)
    # Construction records do not reach the input reader.
    if not np.array_equal(solve(inp), out):
        raise AssertionError("source reader disagrees with construction")
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct examples; legal explicit counts are 2..10 and 1..4."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    coarse, bg, objects, main, directions = _anchor_plan(rng, 2)
    anchors = []
    for quantity in (2, 4):
        changed = [replace(o, cargo=_sample_cargo(main, quantity, rng)) if o.rect == main else o for o in objects]
        anchors.append(_make_pair(coarse, bg, changed))
    if rng.random() < .5:
        anchors.reverse()
    puzzle = {"train": anchors, "test": []}
    seen = {_key(p["input"]) for p in anchors}
    for split, count in (("train", nt), ("test", ns)):
        while len(puzzle[split]) < count:
            for _ in range(40):
                if split == "train":
                    pair = _ordinary_pair(rng)
                else:
                    field, background, objs, _, _ = _anchor_plan(rng, 3, directions)
                    pair = _make_pair(field, background, objs)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(coarse, bg):
    rows = [_runs(row, bg) for row in coarse]
    horizontal = any(b - a >= 2 for row in rows for a, b, _ in row)
    exact = any(set(a) & set(b) for a, b in zip(rows, rows[1:]))
    partial = any(color == other and (a, b) != (c, d) and max(a, c) < min(b, d)
                  for upper, lower in zip(rows, rows[1:])
                  for a, b, color in upper for c, d, other in lower)
    return horizontal, bool(exact), partial


def _split_ablation(objects):
    # Give this wrong rule the correct directions and colors for a fair test.
    return [replace(o, rect=Rect(r, r + 1, o.rect.c0, o.rect.c1, o.rect.color),
                    cargo=tuple(cell for cell in o.cargo if cell[0] == r))
            for o in objects for r in range(o.rect.r0, o.rect.r1)]


def _overlap_ablation(objects):
    """Wrongly fuse touching, same-color run stacks by their bounding box."""
    remaining, result = set(range(len(objects))), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = [first], [first]
        while stack:
            a = objects[stack.pop()].rect
            for index in sorted(remaining):
                b = objects[index].rect
                if (a.color == b.color and (a.r1 == b.r0 or b.r1 == a.r0)
                        and max(a.c0, b.c0) < min(a.c1, b.c1)):
                    remaining.remove(index)
                    component.append(index)
                    stack.append(index)
        members = [objects[i] for i in component]
        r = Rect(min(o.rect.r0 for o in members), max(o.rect.r1 for o in members),
                 min(o.rect.c0 for o in members), max(o.rect.c1 for o in members), members[0].rect.color)
        result.append(replace(members[0], rect=r, cargo=tuple(sorted({p for o in members for p in o.cargo}))))
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train_counts, evidence = set(), set(), [False, False, False]
    clipped, matched, split_changes, overlap_changes, b_changes = False, {}, False, False, False
    train_depth, test_depth, test_free = 0, 0, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            coarse, bg, objects, shape = _read_input(inp)
            assert _rectangles(coarse, bg) == _reference_rectangles(coarse, bg)
            assert out.shape == shape and np.array_equal(out, solve(inp))
            assert len(objects) <= 8
            for o in objects:
                top, bottom, left, right = _bounds(o.rect)
                assert np.all(out[top + 1:bottom, left + 1:right] == bg), "cargo not cleared"
                assert out[_tip(o.rect, o.direction)] == o.rect.color
                ray = _ray(o.rect, o.direction, len(o.cargo), shape)
                assert all(out[p] == o.payload_color for p in ray)
                room, q = _room(o.rect, o.direction, shape), len(o.cargo)
                depth = o.rect.r1 - o.rect.r0
                if split == "train":
                    train_depth = max(train_depth, depth)
                    clipped |= 0 < room < q
                    if room > q:
                        train_counts.add(q)
                        signature = (o.rect, o.direction, o.payload_color, coarse.shape, bg)
                        matched.setdefault(signature, set()).add(q)
                else:
                    test_depth = max(test_depth, depth)
                    test_free |= room > q and q >= 4
            if split == "train":
                evidence = [a or b for a, b in zip(evidence, _features(coarse, bg))]
                split_changes |= not np.array_equal(out, _render(_split_ablation(objects), bg, shape))
                overlap_changes |= not np.array_equal(out, _render(_overlap_ablation(objects), bg, shape))
                b_changes |= not np.array_equal(out, _render(objects, bg, shape, tally=False))
    assert all(evidence), "training misses horizontal, exact, or partial-overlap evidence"
    assert {2, 4} <= train_counts and any({2, 4} <= qs for qs in matched.values())
    assert clipped and split_changes and overlap_changes and b_changes
    assert train_depth <= 2 and test_depth >= 3 and test_free


def _solver_fixtures():
    """Hand-painted expectations cover sizes/depths outside the sampler.

    These fixtures call neither the rectangle partitioner nor output renderer
    to form their expected arrays. They are authored checks, not an external
    solver. Every full-lattice object has one cell of outward clearance.
    """
    for rows, cols, direction in product(range(1, 6), range(1, 6), _DIRECTIONS):
        inp = np.zeros((2 * rows + 1, 2 * cols + 1), dtype=np.int8)
        inp[1::2, 1::2] = 1
        inp[2::2, 2::2] = 2
        rm, cm = (rows - 1) // 2, (cols - 1) // 2
        tabs = {"U": (0, 2 * cm + 1), "D": (2 * rows, 2 * cm + 1),
                "L": (2 * rm + 1, 0), "R": (2 * rm + 1, 2 * cols)}
        inp[tabs[direction]] = 1
        expected = np.zeros((5 * rows + 3, 5 * cols + 3), dtype=np.int8)
        expected[2, 2:5 * cols + 1] = expected[5 * rows, 2:5 * cols + 1] = 1
        expected[2:5 * rows + 1, 2] = expected[2:5 * rows + 1, 5 * cols] = 1
        tips = {"U": (1, 5 * cm + 3), "D": (5 * rows + 1, 5 * cm + 3),
                "L": (5 * rm + 3, 1), "R": (5 * rm + 3, 5 * cols + 1)}
        r, c = tips[direction]
        expected[r, c] = 1
        dr, dc = _STEP[direction]
        expected[r + dr, c + dc] = 2
        assert np.array_equal(solve(inp), expected), "full-lattice size/orientation regression"
    # The same run returns after a blank row. It must start a fresh frame.
    inp = np.zeros((11, 11), dtype=np.int8)
    for r in (0, 2, 3):
        for c in (1, 2):
            inp[2 * r + 1, 2 * c + 1] = 1
            inp[2 * r + 2, 2 * c + 2] = 2
    inp[1, 2] = inp[5, 2] = 1
    expected = np.zeros((28, 28), dtype=np.int8)
    for top, bottom, tip_row, q in ((2, 5, 3, 2), (12, 20, 13, 4)):
        expected[top, 7:16] = expected[bottom, 7:16] = 1
        expected[top:bottom + 1, 7] = expected[top:bottom + 1, 15] = 1
        expected[tip_row, 6] = 1
        expected[tip_row, 6 - q:6] = 2
    assert np.array_equal(solve(inp), expected), "interrupted exact spans incorrectly coalesced"
    return 101


def self_test():
    """Bounded in-memory semantic/variation checks; no puzzle corpus is saved."""
    fingerprints, shapes, normalized, palettes = set(), set(), set(), set()
    counts, depths, widths, quantities, directions, modes, backgrounds = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = np.asarray(pair["input"])
                before = inp.copy()
                assert np.array_equal(solve(inp), pair["output"]) and np.array_equal(inp, before)
                coarse, bg, objects, shape = _read_input(inp)
                shapes.add((coarse.shape, shape))
                backgrounds.add(bg)
                palettes.add(tuple(sorted(int(c) for c in np.unique(inp))))
                roles = {bg: 0}
                labels = tuple(roles.setdefault(int(c), len(roles)) for c in coarse.flat)
                normalized.add((coarse.shape, labels))
                counts.add(len(objects))
                for o in objects:
                    q, room = len(o.cargo), _room(o.rect, o.direction, shape)
                    depths.add(o.rect.r1 - o.rect.r0)
                    widths.add(o.rect.c1 - o.rect.c0)
                    quantities.add(q)
                    directions.add(o.direction)
                    modes.add("free" if room > q else "exact" if room == q else "clipped")
    assert len(fingerprints) == 42 and len(normalized) >= 75
    assert len(shapes) >= 12 and len(palettes) >= 30 and len(backgrounds) == 10
    assert {1, 2, 3} <= depths and len(widths) >= 4 and len(counts) >= 4
    assert {1, 2, 3, 4} <= quantities and max(quantities) >= 7
    assert directions == set(_DIRECTIONS) and modes == {"free", "exact", "clipped"}
    # Minimal demonstrations must preserve all family evidence too.
    for seed in range(42):
        generate_puzzle(seed, 2, 1)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(7301 + 10 * nt + ns, nt, ns)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(53117, 7, 3), generate_puzzle(53117, 7, 3)
    assert all(np.array_equal(x[k], y[k]) for s in ("train", "test")
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
    # Color renaming includes background and role reuse; parity stays fixed.
    p = generate_puzzle(7)
    for s in ("train", "test"):
        for pair in p[s]:
            assert np.array_equal(solve((pair["input"] + 3) % 10), (pair["output"] + 3) % 10)
    fixtures = _solver_fixtures()
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "coarse_shapes": len(shapes),
            "color_normalized_layouts": len(normalized), "backgrounds": len(backgrounds),
            "object_counts": sorted(counts), "run_widths": sorted(widths), "stack_depths": sorted(depths),
            "quantities": sorted(quantities), "directions": sorted(directions), "ray_modes": sorted(modes),
            "hand_painted_solver_fixtures": fixtures}
