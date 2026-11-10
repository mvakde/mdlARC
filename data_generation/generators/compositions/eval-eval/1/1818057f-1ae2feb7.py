"""Resonant crosses: two run-length clocks meet in a two-dimensional field.

No puzzle metadata enters solve. Background is the strict input majority;
every other color is a signal. Detection and timing use the original input.
"""
from __future__ import annotations

import itertools
import numpy as np


CARDINALS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
RIVALS = ("all_marks", "fixed_three", "either_axis", "component_box",
          "endpoint_phase", "consume_centers", "shortest", "longest",
          "farthest", "lowest_color", "highest_color", "no_cancellation")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _background(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= a.size:
        raise ValueError("background must occupy strictly more than half the input")
    return bg


def _record(a, r, c):
    color = int(a[r, c])
    left = right = c
    top = bottom = r
    while left > 0 and a[r, left - 1] == color:
        left -= 1
    while right + 1 < a.shape[1] and a[r, right + 1] == color:
        right += 1
    while top > 0 and a[top - 1, c] == color:
        top -= 1
    while bottom + 1 < a.shape[0] and a[bottom + 1, c] == color:
        bottom += 1
    return (r, c, color, right - left + 1, bottom - top + 1, top, left)


def _sources(a, bg, all_marks=False, consume=False):
    sources = []
    remaining = a.copy() if consume else a
    for color in np.unique(a):
        if color == bg:
            continue
        if all_marks:
            coordinates = np.argwhere(a == color)
        else:
            mask = a == color
            centers = np.zeros(a.shape, dtype=bool)
            centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                         & mask[2:, 1:-1] & mask[1:-1, :-2]
                                         & mask[1:-1, 2:])
            coordinates = np.argwhere(centers)
        for row, col in coordinates:
            r, c = int(row), int(col)
            if consume and not all(remaining[r + dr, c + dc] == color
                                   for dr, dc in CARDINALS):
                continue
            sources.append(_record(a, r, c))
            if consume:
                for dr, dc in CARDINALS:
                    remaining[r + dr, c + dc] = bg
    return sources


def _component_dimensions(a, r, c):
    color = a[r, c]
    pending, seen = [(r, c)], {(r, c)}
    while pending:
        x, y = pending.pop()
        for dx, dy in CARDINALS[1:]:
            q = x + dx, y + dy
            if (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                    and q not in seen and a[q] == color):
                seen.add(q)
                pending.append(q)
    return (max(c for r, c in seen) - min(c for r, c in seen) + 1,
            max(r for r, c in seen) - min(r for r, c in seen) + 1)


def _answer(a, sources, bg, rival=None):
    """Vectorized arbitration; equal-distance different colors cancel."""
    rr, cc = np.indices(a.shape)
    empty = a == bg
    best = np.full(a.shape, 100000, dtype=int)
    chosen = np.full(a.shape, bg, dtype=np.int8)
    conflict = np.zeros(a.shape, dtype=bool)
    for r, c, color, horizontal, vertical, top, left in sources:
        if rival == "fixed_three":
            horizontal = vertical = 3
        elif rival == "component_box":
            horizontal, vertical = _component_dimensions(a, r, c)
        phase_r, phase_c = (top, left) if rival == "endpoint_phase" else (r, c)
        row_beat = (rr - phase_r) % vertical == 0
        col_beat = (cc - phase_c) % horizontal == 0
        firing = empty & ((row_beat | col_beat) if rival == "either_axis"
                          else (row_beat & col_beat))
        distance = abs(rr - r) + abs(cc - c)
        if rival == "shortest":
            score = horizontal * vertical * 100 + distance
        elif rival == "longest":
            score = (900 - horizontal * vertical) * 100 + distance
        elif rival == "lowest_color":
            score = color * 100 + distance
        elif rival == "highest_color":
            score = (9 - color) * 100 + distance
        elif rival == "farthest":
            score = 100 - distance
        else:
            score = distance
        closer = firing & (score < best)
        tied = firing & (score == best) & (chosen != color)
        chosen[closer] = color
        best[closer] = score[closer]
        conflict[closer] = False
        if rival != "no_cancellation":
            conflict[tied] = True
    out = a.copy()
    painted = empty & (best < 100000) & ~conflict
    out[painted] = chosen[painted]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every admissible field using only its visible cells."""
    a = _grid(input_grid)
    bg = _background(a)
    return _answer(a, _sources(a, bg), bg)


def _alternative(a, name):
    bg = _background(a)
    sources = _sources(a, bg, all_marks=name == "all_marks",
                       consume=name == "consume_centers")
    return _answer(a, sources, bg, name)


def _key(a):
    return a.shape, a.tobytes()


def _cross(a, r, c, horizontal, vertical, color, rng):
    # A center can be asymmetrically placed inside either maximal run.
    left = int(rng.integers(1, horizontal - 1))
    up = int(rng.integers(1, vertical - 1))
    points = ([(r, j) for j in range(c - left, c - left + horizontal)]
              + [(i, c) for i in range(r - up, r - up + vertical)])
    if any(not (0 <= x < a.shape[0] and 0 <= y < a.shape[1]) for x, y in points):
        return None
    for q in points:
        a[q] = color
    return left, up


def _free_patch(a, points, bg, forbidden=()):
    ban = set(forbidden)
    for r, c in points:
        if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) or (r, c) in ban:
            return False
        # One blank cell of separation prevents accidental merging of stamps.
        patch = a[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        if np.any(patch != bg):
            return False
    return True


def _nonclock_clutter(a, bg, colors, rng, count, forbidden=()):
    offsets = (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
        tuple((r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    )
    for _ in range(count * 12):
        if count == 0:
            break
        stamp = offsets[int(rng.integers(len(offsets)))]
        r, c = int(rng.integers(a.shape[0])), int(rng.integers(a.shape[1]))
        points = [(r + x, c + y) for x, y in stamp]
        if _free_patch(a, points, bg, forbidden):
            color = int(rng.choice(colors))
            for q in points:
                a[q] = color
            count -= 1


def _calibration(rng, shape, bg, colors):
    a = np.full(shape, bg, dtype=np.int8)
    h, w = shape
    options = [(x, y) for x in range(3, min(7, (w - 3) // 2) + 1)
               for y in range(3, min(7, (h - 3) // 2) + 1) if x != y]
    horizontal, vertical = options[int(rng.integers(len(options)))]
    r = int(rng.integers(vertical + 1, h - vertical))
    c = int(rng.integers(horizontal + 1, w - horizontal))
    left, up = _cross(a, r, c, horizontal, vertical, colors[0], rng)
    # A bent, connected tail enlarges the component box without changing the
    # center's runs. It defeats component-size-as-period, with only one center.
    endpoint = c - left
    extra = int(rng.integers(1, min(4, r - up) + 1))
    a[r - up - extra:r + 1, endpoint] = colors[0]
    forbidden = [(x, y) for x in range(h) for y in range(w)
                 if (x - r) % vertical == 0 and (y - c) % horizontal == 0]
    _nonclock_clutter(a, bg, colors, rng, int(rng.integers(2, 6)), forbidden)
    assert len(_sources(a, bg)) == 1
    return a


def _interactions(a, out=None):
    """Visible collision evidence; no planted center metadata is retained."""
    bg = _background(a)
    sources = _sources(a, bg)
    edges, ties = set(), 0
    for row, col in np.argwhere(a == bg):
        r, c = int(row), int(col)
        hits = [(abs(r - x) + abs(c - y), color)
                for x, y, color, horizontal, vertical, _, _ in sources
                if (r - x) % vertical == 0 and (c - y) % horizontal == 0]
        if not hits:
            continue
        distance = min(d for d, color in hits)
        winners = {color for d, color in hits if d == distance}
        if len(winners) > 1:
            ties += 1
            if out is not None:
                assert out[r, c] == bg
        else:
            winner = next(iter(winners))
            for d, color in hits:
                if d > distance and color != winner:
                    edges.add((winner, color))
    cycle = any((b, a) in edges for a, b in edges)
    cells = {(r + dr, c + dc) for r, c, *_ in sources for dr, dc in CARDINALS}
    overlap = len(cells) < 5 * len(sources)
    return {"color_cycle": cycle, "ties": ties, "overlap": overlap,
            "centers": len(sources), "periods": {v for s in sources for v in s[3:5]}}


def _interaction_lesson(rng, shape, bg, colors):
    h, w = shape
    for _ in range(100):
        a = np.full(shape, bg, dtype=np.int8)
        horizontal = int(rng.integers(3, min(5, (w - 1) // 4) + 1))
        v1, v2 = map(int, rng.choice(range(3, 8), 2, replace=False))
        r = h // 2 + int(rng.integers(-1, 2))
        c1 = int(rng.integers(horizontal, w - 3 * horizontal))
        c2 = c1 + 2 * horizontal
        _cross(a, r, c1, horizontal, v1, colors[0], rng)
        _cross(a, r, c2, horizontal, v2, colors[1], rng)
        # Two adjacent centers in a single 3x4 block expose simultaneous
        # recognition and separate center phases, even inside one component.
        top = int(rng.choice((0, h - 3)))
        col = int(rng.integers(w - 3))
        points = [(top + x, col + y) for x in range(3) for y in range(4)]
        if not _free_patch(a, points, bg):
            continue
        for q in points:
            a[q] = colors[2]
        _nonclock_clutter(a, bg, colors, rng, int(rng.integers(0, 5)))
        out = solve(a)
        evidence = _interactions(a, out)
        if not (evidence["color_cycle"] and evidence["ties"] and evidence["overlap"]):
            continue
        if any(np.array_equal(out, _alternative(a, name)) for name in
               ("consume_centers", "shortest", "longest", "farthest",
                "lowest_color", "highest_color", "no_cancellation")):
            continue
        return a
    # This finite geometric fallback has hand-checked blank collision witnesses:
    # row 10, cols 3 and 15 reverse precedence; col 9 cancels. The edge block
    # supplies two phases without competing at those three coordinates.
    a = np.full((23, 23), bg, dtype=np.int8)
    a[10, 5:8] = colors[0]
    a[9:12, 6] = colors[0]
    a[10, 11:14] = colors[1]
    a[8:13, 12] = colors[1]
    a[0:3, 0:4] = colors[2]
    return a


def _random_scene(rng, shape, bg, colors, max_period=7, crowded=False):
    a = np.full(shape, bg, dtype=np.int8)
    h, w = shape
    count = int(rng.integers(4, 9) if crowded else rng.integers(1, 6))
    for _ in range(count * 25):
        if count == 0:
            break
        color = int(rng.choice(colors))
        kind = int(rng.integers(4))
        if kind == 0:
            bh = int(rng.integers(3, min(6, h - 1) + 1))
            bw = int(rng.integers(3, min(7, w - 1) + 1))
            top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
            points = [(top + x, left + y) for x in range(bh) for y in range(bw)]
        else:
            horizontal = int(rng.integers(3, min(max_period, w - 1) + 1))
            vertical = int(rng.integers(3, min(max_period, h - 1) + 1))
            temp = np.full(shape, bg, dtype=np.int8)
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            if _cross(temp, r, c, horizontal, vertical, color, rng) is None:
                continue
            if kind == 2:
                # Shared-arm or adjacent-center clusters, with fresh runs
                # measured after their union is formed.
                delta = int(rng.integers(1, 3))
                _cross(temp, r, c + delta, 3, 3, color, rng)
            if kind == 3:
                # Diagonal attachments do not invalidate a local plus.
                for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    if rng.random() < .55:
                        temp[r + dx, c + dy] = color
            points = [tuple(map(int, q)) for q in np.argwhere(temp != bg)]
        if _free_patch(a, points, bg) and len(points) + np.count_nonzero(a != bg) < a.size * .43:
            for q in points:
                a[q] = color
            count -= 1
    _nonclock_clutter(a, bg, colors, rng, int(rng.integers(0, 7)))
    # A rare uninformative packing draw is repaired with an interior cross.
    if not _sources(a, bg):
        a[:] = bg
        _cross(a, h // 2, w // 2, 3, 4, colors[0], rng)
    return a


def _legal_integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _legal_integer(num_train, "num_train", 2, 10)
    num_test = _legal_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = rng.choice(10, 6, replace=False).astype(int)
    stable_palette = bool(rng.random() < .35)
    shift = str(rng.choice(("extend", "clip")))
    puzzle, seen = {"train": [], "test": []}, set()
    base_rotation = int(rng.integers(4))
    mixed_orientation = bool(rng.random() < .65)

    def add(split, kind, max_period=7):
        roles = palette if stable_palette else rng.choice(10, 6, replace=False).astype(int)
        bg, colors = int(roles[0]), list(map(int, roles[1:]))
        if split == "train":
            low, high = (19, 26) if shift == "extend" else (21, 27)
            shape = tuple(map(int, rng.integers(low, high + 1, 2)))
        elif shift == "extend":
            shape = (int(rng.integers(27, 31)), int(rng.integers(18, 31)))
        else:
            shape = tuple(map(int, rng.integers(10, 18, 2)))
        if kind == "calibration":
            a = _calibration(rng, shape, bg, colors)
        elif kind == "interaction":
            a = _interaction_lesson(rng, shape, bg, colors)
        else:
            for _ in range(60):
                a = _random_scene(rng, shape, bg, colors, max_period,
                                  crowded=split == "test" and rng.random() < .55)
                sources = _sources(a, bg)
                # Clusters can merge into longer runs; keep generated test
                # periods within the numerical range exposed by training.
                if split == "test" and any(max(s[3:5]) > max_period for s in sources):
                    continue
                if (np.any(solve(a) != a)
                        and not np.array_equal(solve(a), _alternative(a, "fixed_three"))):
                    break
            else:
                a = np.full(shape, bg, dtype=np.int8)
                _cross(a, shape[0] // 2, shape[1] // 2, 3, 4, colors[0], rng)
        rotation = int(rng.integers(4)) if mixed_orientation else base_rotation
        a = np.rot90(a, rotation).copy()
        # Any palette injection is rule-preserving. The first 15 distinct
        # assignments exceed the maximum of 13 previously occupied pair slots.
        if _key(a) in seen:
            used = list(map(int, np.unique(a)))
            for values in itertools.islice(itertools.permutations(range(10), len(used)), 15):
                mapping = np.arange(10, dtype=np.int8)
                mapping[used] = values
                candidate = mapping[a]
                if _key(candidate) not in seen:
                    a = candidate
                    break
            else:
                raise AssertionError("finite uniqueness bound exceeded")
        seen.add(_key(a))
        puzzle[split].append({"input": a, "output": solve(a)})

    add("train", "calibration")
    add("train", "interaction")
    for _ in range(num_train - 2):
        add("train", "random")
    max_period = max(max(s[3:5]) for pair in puzzle["train"]
                     for s in _sources(pair["input"], _background(pair["input"])))
    for _ in range(num_test):
        add("test", "random", max_period)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check labels, preserved signal, resonance witnesses and curriculum."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            bg = _background(a)
            assert np.array_equal(a[a != bg], out[a != bg]), "signal is protected"
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(a)))
            assert np.array_equal(solve(a), out), "wrong field phase, winner or cancellation"
            assert np.any(a != out), "generated examples must visibly draw a field"
            sources = _sources(a, bg)
            assert sources and all(min(s[3:5]) >= 3 for s in sources)
            # Direct cell invariants check the vectorized arbitration, including
            # non-firing nearer clocks and ties with three or more contenders.
            for row, col in np.argwhere(a == bg):
                r, c = int(row), int(col)
                hits = [(abs(r - x) + abs(c - y), color)
                        for x, y, color, horizontal, vertical, _, _ in sources
                        if (r - x) % vertical == 0 and (c - y) % horizontal == 0]
                if not hits:
                    assert out[r, c] == bg
                else:
                    distance = min(d for d, color in hits)
                    winners = {color for d, color in hits if d == distance}
                    if len(winners) == 1:
                        assert out[r, c] == next(iter(winners))
                    else:
                        assert out[r, c] == bg
            if split == "train":
                evidence.append(_interactions(a, out))
    assert any(e["color_cycle"] and e["ties"] and e["overlap"] for e in evidence)
    # The minimum two demonstrations already refute all these plausible rivals.
    for name in RIVALS:
        assert any(not np.array_equal(_alternative(_grid(p["input"]), name), p["output"])
                   for p in puzzle["train"]), f"training admits shortcut: {name}"
    single_clock = False
    for pair in puzzle["train"]:
        a, out = _grid(pair["input"]), _grid(pair["output"])
        bg = _background(a)
        sources = _sources(a, bg)
        if len(sources) != 1:
            continue
        r, c, color, horizontal, vertical, *_ = sources[0]
        points = ((r, c - horizontal), (r, c + horizontal),
                  (r - vertical, c), (r + vertical, c),
                  (r - vertical, c - horizontal), (r + vertical, c + horizontal))
        if horizontal != vertical and all(0 <= x < a.shape[0] and 0 <= y < a.shape[1]
                                         and a[x, y] == bg and out[x, y] == color for x, y in points):
            single_clock = True
    assert single_clock, "need independent two-axis recurrence in training"
    taught_max = max(v for e in evidence for v in e["periods"])
    for pair in puzzle["test"]:
        a = _grid(pair["input"])
        assert all(max(s[3:5]) <= taught_max for s in _sources(a, _background(a)))
        assert not np.array_equal(pair["output"], _alternative(a, "fixed_three"))


def _puzzle_key(puzzle):
    return tuple((_key(_grid(pair["input"])), _key(_grid(pair["output"])))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded diagnostics, metamorphic checks, and 42-seed variation checks."""
    # Exact manually specified periods and expected lattice, including negative
    # offsets. The attached tail changes the box but not the run clocks.
    a = np.zeros((13, 17), dtype=np.int8)
    a[6, 7:11] = 2
    a[4:9, 8] = 2
    a[1:7, 7] = 2
    expected = a.copy()
    for r in (1, 6, 11):
        for c in (0, 4, 8, 12, 16):
            if a[r, c] == 0:
                expected[r, c] = 2
    assert np.array_equal(solve(a), expected)
    # Overlap: a 3x4 block has two clocks, not one component clock.
    b = np.zeros((11, 14), dtype=np.int8)
    b[3:6, 4:8] = 7
    expected_b = b.copy()
    for r in (1, 4, 7, 10):
        for c in (1, 2, 5, 6, 9, 10, 13):
            if b[r, c] == 0:
                expected_b[r, c] = 7
    assert len(_sources(b, 0)) == 2
    assert np.array_equal(solve(b), expected_b)
    # Three fixed coordinates test both precedence orders and cancellation.
    c = np.zeros((23, 23), dtype=np.int8)
    c[10, 5:8] = 2
    c[9:12, 6] = 2
    c[10, 11:14] = 4
    c[8:13, 12] = 4
    out_c = solve(c)
    assert tuple(out_c[10, [3, 9, 15]]) == (2, 0, 4)
    # Same-color ties agree; a boundary-centered clipped cross never emits.
    same = c.copy()
    same[same == 4] = 2
    assert solve(same)[10, 9] == 2
    clipped = np.zeros((9, 10), dtype=np.int8)
    clipped[0, 3:6] = 5
    clipped[1:4, 4] = 5
    assert np.array_equal(solve(clipped), clipped)
    # Detector truth table: all 512 local binary neighborhoods. Outside the
    # center, boundary cells cannot become complete plus centers.
    for bits in range(512):
        mask = np.array([(bits >> k) & 1 for k in range(9)], dtype=np.int8).reshape(3, 3)
        records = _sources(mask, 0)
        expected_center = all(mask[1 + dr, 1 + dc] == 1 for dr, dc in CARDINALS)
        assert [(s[0], s[1]) for s in records] == ([(1, 1)] if expected_center else [])
    for original in (a, b, c, clipped):
        answer = solve(original)
        for turns in range(4):
            assert np.array_equal(solve(np.rot90(original, turns)), np.rot90(answer, turns))
        assert np.array_equal(solve(original[:, ::-1]), answer[:, ::-1])
        mapping = np.array([9, 3, 0, 1, 8, 2, 4, 6, 7, 5], dtype=np.int8)
        assert np.array_equal(solve(mapping[original]), mapping[answer])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    shapes, palettes, layouts, counts, periods = set(), set(), set(), set(), set()
    edge, overlap, ties, growth, clipping = 0, 0, 0, 0, 0
    for puzzle in puzzles:
        train_areas = [np.asarray(p["input"]).size for p in puzzle["train"]]
        test_area = np.asarray(puzzle["test"][0]["input"]).size
        growth += test_area > max(train_areas)
        clipping += test_area < min(train_areas)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg = _background(inp)
                sources = _sources(inp, bg)
                e = _interactions(inp)
                shapes.add(inp.shape)
                palettes.add((bg, tuple(map(int, np.unique(inp)))))
                layouts.add(tuple((s[0], s[1], s[3], s[4]) for s in sources))
                counts.add(len(sources))
                periods.update(e["periods"])
                overlap += e["overlap"]
                ties += e["ties"] > 0
                edge += any(r in (1, inp.shape[0] - 2) or col in (1, inp.shape[1] - 2)
                            for r, col, *_ in sources)
    assert len(shapes) >= 60 and any(30 in shape for shape in shapes)
    assert len(palettes) >= 60 and len(layouts) >= 150
    assert len(counts) >= 10 and {3, 4, 5, 6, 7} <= periods
    assert min(overlap, ties, edge) >= 42 and min(growth, clipping) >= 8
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(73, train, test)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4),
                        (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {split: [{role: grid.copy() for role, grid in pair.items()}
                   for pair in puzzles[0][split]] for split in puzzles[0]}
    q = tuple(np.argwhere(bad["test"][0]["input"] != _background(bad["test"][0]["input"]))[0])
    bad["test"][0]["output"][q] = (int(bad["test"][0]["output"][q]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted changed source ink")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "shapes": len(shapes), "palettes": len(palettes), "layouts": len(layouts),
            "center_counts": sorted(counts), "periods": sorted(periods),
            "overlap_pairs": overlap, "tie_pairs": ties, "edge_pairs": edge,
            "larger_tests": growth, "clipped_tests": clipping}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
