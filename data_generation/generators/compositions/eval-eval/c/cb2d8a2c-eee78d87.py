"""Detour loom: transport a periodic three-cell ribbon around marked bars.

Only standard library and NumPy are used. No files, parent modules, labels,
random state, or cached generated examples are consulted by solve().
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _full_box(mask):
    return (mask.shape == (3, 3) and 2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> j) & 1 for j in range(9)],
                                          dtype=bool).reshape(3, 3)))
_PERMS = tuple(itertools.permutations(range(3)))
_ORDERS = tuple((t, r, c) for t in (False, True) for r in _PERMS for c in _PERMS)
_IDENTITY = (False, (0, 1, 2), (0, 1, 2))


def _reorder(mask, order):
    transposed, rows, columns = order
    return (mask.T if transposed else mask)[np.ix_(rows, columns)]


def _agreements(mask):
    return {o for o in _ORDERS if np.array_equal(mask, _reorder(mask, o))}


def _check_integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"{name} must be an integer in the allowed range")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _parse(input_grid):
    """Find the open frame, normalize orientation, and read visible commands."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 4 and counts.max() > grid.size // 2
    bg = int(colors[counts.argmax()])
    candidates = []
    for rotation in range(4):
        a = np.rot90(grid, rotation)
        h, w = a.shape
        if h < 14 or w < 9:
            continue
        for left in range(w - 4):
            box = a[:5, left:left + 5]
            bar, ink = int(box[0, 0]), int(box[4, 2])
            if len({bg, bar, ink}) != 3:
                continue
            border = np.ones((5, 5), dtype=bool)
            border[1:4, 1:4] = False
            border[4, 2] = False
            if not np.all(box[border] == bar):
                continue
            inner = box[1:4, 1:4]
            if not np.all((inner == bg) | (inner == ink)):
                continue
            motif = inner == ink
            if not _full_box(motif):
                continue
            rest_colors = set(map(int, colors)) - {bg, bar, ink}
            if len(rest_colors) != 1:
                continue
            marker = rest_colors.pop()
            source = np.zeros(a.shape, dtype=bool)
            source[:5, left:left + 5] = True
            if np.any((a == ink) & ~source):
                continue
            if np.any((a[:5] != bg) & ~source[:5]):
                continue
            if np.any((a == marker) & source):
                continue
            candidates.append((a.copy(), rotation, (bg, bar, marker, ink),
                               motif.copy(), left, source))
    assert len(candidates) == 1, "input must have exactly one readable source frame"
    a, rotation, roles, motif, left, source = candidates[0]
    bg, bar, marker, ink = roles
    h, w = a.shape
    outside = ((a == bar) | (a == marker)) & ~source
    rows = list(map(int, np.flatnonzero(outside.any(axis=1))))
    assert 1 <= len(rows) <= 4
    current, previous = left + 2, None
    bars = []
    for row in rows:
        assert 0 < row <= h - 3
        occupied = np.flatnonzero(outside[row])
        lo, hi = int(occupied[0]), int(occupied[-1])
        assert len(occupied) == hi - lo + 1
        markers = tuple(map(int, np.flatnonzero(a[row] == marker)))
        n = len(markers)
        assert 1 <= n <= 4 and hi - lo + 1 >= 2 * n
        low = markers == tuple(lo + 2 * j for j in range(n))
        high = markers == tuple(hi - 2 * j for j in reversed(range(n)))
        assert low != high, "the marker-bearing endpoint must be unambiguous"
        side = 1 if low else -1
        d = n + 1
        target = hi + d if side == 1 else lo - d
        turn = row - d
        assert lo <= current <= hi, "each bar must intercept the incoming ray"
        assert 1 <= target <= w - 2, "one-cell ribbon margin is required"
        assert turn >= (9 if previous is None else previous + 2)
        bars.append((row, lo, hi, n, side))
        current, previous = target, row
    assert np.all(a[(~source) & (~outside)] == bg)
    return {"a": a, "rotation": rotation, "roles": roles, "motif": motif,
            "left": left, "source": source, "bars": tuple(bars)}


def _segments(info, *, clearance=None, opposite=False, straight=False):
    """Centerline segments in source-facing coordinates, starting after the port."""
    current = (5, info["left"] + 2)
    if straight:
        return [(current, (info["a"].shape[0] - 1, current[1]))]
    segments = []
    for row, lo, hi, n, side in info["bars"]:
        d = n + 1 if clearance is None else clearance
        side = -side if opposite else side
        target = hi + d if side == 1 else lo - d
        bend = (row - d, current[1])
        exit_bend = (row - d, target)
        segments.extend(((current, bend), (bend, exit_bend)))
        current = exit_bend
    segments.append((current, (info["a"].shape[0] - 1, current[1])))
    return [(p, q) for p, q in segments if p != q]


def _paint(info, segments=None, *, solid=False, reset=False, global_tile=False,
           rotate_motif=True, order=_IDENTITY, phase=1, strict=False):
    """Union occupied transverse slices; empty motif cells never erase ink."""
    a = info["a"]
    bg, bar, marker, ink = info["roles"]
    motif = _reorder(info["motif"], order)
    segments = _segments(info) if segments is None else segments
    occupancy = np.zeros(a.shape, dtype=bool)
    footprint = np.zeros(a.shape, dtype=bool)
    elapsed = 0
    origin_column = info["left"] + 2
    for p, q in segments:
        length = abs(q[0] - p[0]) + abs(q[1] - p[1])
        dr, dc = (q[0] - p[0]) // length, (q[1] - p[1]) // length
        nr, nc = (-dc, dr) if rotate_motif else (0, 1)
        for step in range(length + 1):
            r, c = p[0] + step * dr, p[1] + step * dc
            s = step if reset else elapsed + step
            for offset in (-1, 0, 1):
                y, x = r + offset * nr, c + offset * nc
                if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]):
                    assert not strict, "ribbon leaves the grid"
                    continue
                footprint[y, x] = True
                on = (motif[(y - 5 + phase) % 3,
                            (x - origin_column + 1) % 3] if global_tile else
                      motif[(s + phase) % 3, offset + 1])
                if solid or on:
                    occupancy[y, x] = True
        elapsed += length
    if strict:
        assert not np.any(footprint & (a != bg)), "ribbon must clear source and bars"
    # Alternatives may intersect bars: preserve structures in that counterfactual.
    occupancy &= a == bg
    result = a.copy()
    result[a == marker] = bar
    result[occupancy] = ink
    return result, occupancy, footprint


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the grid alone; the source port fixes orientation and all roles."""
    info = _parse(input_grid)
    result, _, _ = _paint(info, strict=True)
    return np.rot90(result, -info["rotation"]).copy()


def _walk(segments):
    path = [segments[0][0]]
    for p, q in segments:
        assert path[-1] == p
        length = abs(q[0] - p[0]) + abs(q[1] - p[1])
        dr, dc = (q[0] - p[0]) // length, (q[1] - p[1]) // length
        path.extend((p[0] + k * dr, p[1] + k * dc) for k in range(1, length + 1))
    assert len(path) == len(set(path)), "centerline must not revisit a cell"
    return path


def _reference_occupancy(info):
    """Check the renderer using route vertices and their neighboring directions."""
    path = _walk(_segments(info))
    result = np.zeros(info["a"].shape, dtype=bool)
    for step, (r, c) in enumerate(path):
        tangents = set()
        if step:
            y, x = path[step - 1]
            tangents.add((r - y, c - x))
        if step + 1 < len(path):
            y, x = path[step + 1]
            tangents.add((y - r, x - c))
        for dr, dc in tangents:
            for offset in range(-1, 2):
                if info["motif"][(step + 1) % 3, offset + 1]:
                    y, x = r - offset * dc, c + offset * dr
                    assert 0 <= y < result.shape[0] and 0 <= x < result.shape[1]
                    result[y, x] = True
    assert not np.any(result & (info["a"] != info["roles"][0]))
    return result


_SHORTCUTS = ("straight", "fixed_clearance", "wrong_endpoint", "solid",
              "global_wallpaper", "reset_at_bends", "unturned_motif")


def _alternative(info, name):
    if name == "straight":
        return _paint(info, _segments(info, straight=True))[0]
    if name == "fixed_clearance":
        return _paint(info, _segments(info, clearance=2))[0]
    if name == "wrong_endpoint":
        return _paint(info, _segments(info, opposite=True))[0]
    options = {"solid": {"solid": True}, "global_wallpaper": {"global_tile": True},
               "reset_at_bends": {"reset": True}, "unturned_motif": {"rotate_motif": False}}
    return _paint(info, **options[name])[0]


def _span_plan(rng, width, counts, first_side, repeat, alternating):
    """Bounded memoized search over intercepting spans, never entire puzzles."""
    failed = set()

    def search(index, current, previous):
        if index == len(counts):
            return []
        state = index, current, previous
        if state in failed:
            return None
        n = counts[index]
        sides = (-1, 1)
        if index == 0 and first_side is not None:
            sides = (first_side,)
        elif index == 1 and repeat:
            sides = (previous,)
        elif index and alternating:
            sides = (-previous,)
        options = []
        for side in sides:
            for lo in range(current + 1):
                for hi in range(max(current, lo + 2 * n - 1), width):
                    target = hi + n + 1 if side == 1 else lo - n - 1
                    if 1 <= target <= width - 2:
                        options.append((lo, hi, side, target))
        for index_option in rng.permutation(len(options)):
            lo, hi, side, target = options[int(index_option)]
            tail = search(index + 1, target, side)
            if tail is not None:
                return [(lo, hi, side)] + tail
        failed.add(state)
        return None

    for initial in rng.permutation(np.arange(2, width - 2)):
        result = search(0, int(initial), 0)
        if result is not None:
            return int(initial), result
    return None


def _make_pair(rng, roles, motif, counts, rotation, first_side=None,
               repeat=False, alternating=False):
    bg, bar, marker, ink = roles
    minimum_width = max(9, 3 * max(counts) + 3)
    chosen = int(rng.integers(minimum_width, 31))
    for width in dict.fromkeys((chosen, min(30, chosen + 5), 30)):
        plan = _span_plan(rng, width, counts, first_side, repeat, alternating)
        if plan is not None:
            break
    else:
        raise AssertionError("no feasible buffered span sequence")
    initial, spans = plan
    minimum_height = 10 + sum(counts) + 3 * len(counts)
    assert minimum_height <= 30
    height = int(rng.integers(minimum_height, 31))
    slack = rng.multinomial(height - minimum_height,
                            np.full(len(counts) + 1, 1 / (len(counts) + 1)))
    rows, previous = [], None
    for i, n in enumerate(counts):
        turn = (9 if previous is None else previous + 2) + int(slack[i])
        previous = turn + n + 1
        rows.append(previous)
    assert height - 1 - rows[-1] == 2 + int(slack[-1])
    a = np.full((height, width), bg, dtype=np.int8)
    left = initial - 2
    a[:5, left:left + 5] = bar
    a[1:4, left + 1:left + 4] = np.where(motif, ink, bg)
    a[4, initial] = ink
    bars = []
    for row, n, (lo, hi, side) in zip(rows, counts, spans):
        a[row, lo:hi + 1] = bar
        for j in range(n):
            a[row, lo + 2 * j if side == 1 else hi - 2 * j] = marker
        bars.append((row, lo, hi, n, side))
    info = {"a": a, "roles": roles, "motif": motif, "left": left, "bars": bars}
    b = _paint(info, strict=True)[0]
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(b, rotation).copy()}


def _marker_counts(rng, number, maximum, required):
    counts = list(map(int, rng.integers(1, maximum + 1, size=number)))
    for slot, value in required.items():
        counts[slot] = value
    while 10 + sum(counts) + 3 * number > 30:
        candidates = [j for j, n in enumerate(counts) if n > 1 and j not in required]
        assert candidates
        counts[int(rng.choice(candidates))] -= 1
    return counts


def _motifs(rng, num_train, num_test, regime):
    train_pool = list(range(len(_MASKS)))
    test_pool = list(train_pool)
    if regime == "density":
        dense = bool(rng.integers(2))
        train_pool = [i for i, m in enumerate(_MASKS) if (int(m.sum()) >= 6) == dense]
        test_pool = [i for i, m in enumerate(_MASKS) if (int(m.sum()) >= 6) != dense]
    first = int(rng.choice(train_pool))
    alternatives = _agreements(_MASKS[first]) - {_IDENTITY}
    second_pool = [i for i in train_pool if _MASKS[i].sum() != _MASKS[first].sum()
                   and not any(np.array_equal(_MASKS[i], _reorder(_MASKS[i], o))
                               for o in alternatives)]
    assert second_pool
    selected = [first, int(rng.choice(second_pool))]
    selected.extend(map(int, rng.choice([i for i in train_pool if i not in selected],
                                       size=num_train - 2, replace=False)))
    tests = list(map(int, rng.choice([i for i in test_pool if i not in selected],
                                    size=num_test, replace=False)))
    return [_MASKS[i].copy() for i in selected], [_MASKS[i].copy() for i in tests]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _check_integer(seed, "seed", 0)
    num_train = _check_integer(num_train, "num_train", 2, 10)
    num_test = _check_integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    maximum = int(rng.integers(2, 5))
    regime = str(rng.choice(("rotation", "length", "density")))
    directions = list(map(int, rng.permutation(4)))
    maximum_train_bars = 2 if regime == "length" else 3
    for _ in range(48):
        train_masks, test_masks = _motifs(rng, num_train, num_test, regime)
        numbers = list(map(int, rng.integers(1, maximum_train_bars + 1, size=num_train)))
        repeat_pair = int(rng.integers(num_train))
        numbers[repeat_pair] = max(2, numbers[repeat_pair])
        while sum(numbers) < maximum:
            eligible = [i for i, number in enumerate(numbers) if number < maximum_train_bars]
            i = int(rng.choice(eligible))
            numbers[i] += 1
        slots = [(i, j) for i, number in enumerate(numbers) for j in range(number)]
        requirements = [{} for _ in numbers]
        for n, slot in zip(rng.permutation(np.arange(1, maximum + 1)), rng.permutation(len(slots))):
            i, j = slots[int(slot)]
            requirements[i][j] = int(n)
        puzzle, seen = {"train": [], "test": []}, set()
        for split, masks in (("train", train_masks), ("test", test_masks)):
            for i, motif in enumerate(masks):
                if split == "train":
                    counts = _marker_counts(rng, numbers[i], maximum, requirements[i])
                    rotation = directions[i % 3] if regime == "rotation" else int(rng.integers(4))
                    first_sign = 1 if directions[0] % 2 else -1
                    side = first_sign if i == 0 else -first_sign if i == 1 else None
                else:
                    number = int(rng.integers(3, 5) if regime == "length" else rng.integers(1, 5))
                    counts = _marker_counts(rng, number, maximum, {})
                    rotation = directions[3] if regime == "rotation" else int(rng.integers(4))
                    side = None
                pair = _make_pair(rng, roles, motif, counts, rotation, first_side=side,
                                  repeat=split == "train" and i == repeat_pair)
                assert _key(pair["input"]) not in seen  # All motifs are distinct.
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
        for split in puzzle:
            puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
        # Make the promised evidence concrete even at the two-training minimum.
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise AssertionError("bounded demonstration-witness search exhausted")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, palettes, analyses = set(), set(), {"train": [], "test": []}
    witnesses = {name: False for name in _SHORTCUTS}
    orders = set(_ORDERS)
    masks = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            expected, occupied, footprint = _paint(info, strict=True)
            out = np.rot90(b, info["rotation"])
            bg, bar, marker, ink = info["roles"]
            assert np.array_equal(out, expected)
            assert np.array_equal(occupied, _reference_occupancy(info))
            assert np.all(out[info["source"]] == info["a"][info["source"]])
            assert np.all(out[info["a"] == marker] == bar)
            assert np.all(out[info["a"] == bar] == bar)
            assert np.array_equal((out == ink) & (info["a"] == bg), occupied)
            assert np.any(footprint & ~occupied), "the ribbon must retain carrier gaps"
            assert np.any(occupied) and not np.array_equal(a, b)
            path = _walk(_segments(info))
            assert len(path) > 9
            # The first three free slices expose a complete cyclically shifted motif.
            x = info["left"] + 2
            assert np.array_equal(out[5:8, x - 1:x + 2] == ink,
                                  info["motif"][[1, 2, 0], :])
            palettes.add(info["roles"])
            analyses[split].append(info)
            mk = info["motif"].tobytes()
            assert mk not in masks[split]
            masks[split].add(mk)
            if split == "train":
                orders &= _agreements(info["motif"])
                for name in _SHORTCUTS:
                    if not witnesses[name]:
                        witnesses[name] = not np.array_equal(out, _alternative(info, name))
    assert len(palettes) == 1
    assert masks["train"].isdisjoint(masks["test"])
    assert orders == {_IDENTITY}, "training must resolve tile orientation"
    assert all(witnesses.values()), f"unexposed alternatives: {witnesses}"
    train = analyses["train"]
    train_counts = {bar[3] for info in train for bar in info["bars"]}
    assert {1, 2} <= train_counts
    assert {bar[4] for info in train for bar in info["bars"]} == {-1, 1}
    assert any(any(a[4] == b[4] for a, b in zip(info["bars"], info["bars"][1:])) for info in train)
    assert len({int(info["motif"].sum()) for info in train}) >= 2
    assert {bar[3] for info in analyses["test"] for bar in info["bars"]} <= train_counts
    # At least one explicitly constructed distribution shift must be visible.
    unseen_direction = {i["rotation"] for i in analyses["test"]} - {i["rotation"] for i in train}
    longer = min(len(i["bars"]) for i in analyses["test"]) > max(len(i["bars"]) for i in train)
    densities_train = [int(i["motif"].sum()) for i in train]
    densities_test = [int(i["motif"].sum()) for i in analyses["test"]]
    density_shift = (max(densities_train) < min(densities_test)
                     or max(densities_test) < min(densities_train))
    assert unseen_direction or longer or density_shift


def self_test():
    """42 puzzles, one at a time; no persistent examples or diagnostic logs."""
    motifs, shapes, palettes, normalized, fingerprints = set(), set(), set(), set(), set()
    directions, densities, bar_numbers, unary, transitions = set(), set(), set(), set(), set()
    pairs = overlap = phase_sensitive = rectangular = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a.copy()), b)
                info = _parse(a)
                shapes.add(a.shape)
                palettes.add(info["roles"])
                motifs.add(info["motif"].tobytes())
                directions.add(info["rotation"])
                densities.add(int(info["motif"].sum()))
                bar_numbers.add(len(info["bars"]))
                unary.update(x[3] for x in info["bars"])
                transitions.update((x[4], y[4]) for x, y in zip(info["bars"], info["bars"][1:]))
                codes = np.empty_like(info["a"])
                for role, color in enumerate(info["roles"]):
                    codes[info["a"] == color] = role
                normalized.add((codes.shape, codes.tobytes()))
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                pairs += 1
                rectangular += a.shape[0] != a.shape[1]
                phase_sensitive += not np.array_equal(_paint(info)[0], _alternative(info, "reset_at_bends"))
                # A turn's two transverse slices share the same path index.
                segments = _segments(info)
                overlap += len(_walk(segments)) < sum(abs(q[0] - p[0]) + abs(q[1] - p[1]) + 1
                                                     for p, q in segments)
        fingerprints.add(digest.hexdigest())
        if seed == 0:
            first = tuple((_key(p["input"]), _key(p["output"]))
                          for s in ("train", "test") for p in puzzle[s])
            again = generate_puzzle(seed)
            assert first == tuple((_key(p["input"]), _key(p["output"]))
                                  for s in ("train", "test") for p in again[s])
    assert len(fingerprints) == 42 and len(motifs) >= 70 and len(shapes) >= 45
    assert len(normalized) >= 180 and len(palettes) >= 30
    assert directions == set(range(4)) and densities == set(range(2, 9))
    assert bar_numbers == {1, 2, 3, 4} and unary == {1, 2, 3, 4}
    assert transitions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    assert phase_sensitive > 100 and rectangular > 100 and overlap == pairs
    # All legal counts, including the minimal demonstrations and maximal return.
    explicit = 0
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            puzzle = generate_puzzle(1000 + explicit, ntrain, ntest)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
            explicit += 1
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                          (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Concrete corner overlap: two headings at the same vertex, no phase advance.
    info = {"a": np.zeros((15, 15), dtype=np.int8), "roles": (0, 1, 2, 3),
            "motif": np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool), "left": 4}
    segments = [((5, 6), (8, 6)), ((8, 6), (8, 10)), ((8, 10), (14, 10))]
    painted = _paint(info, segments, strict=True)[1]
    # s=3 at first bend: phase row 1, just its center. At second bend,
    # s=7: phase row 2, occupied offsets -1 and 0 in both headings.
    assert painted[8, 6] and not painted[8, 5] and not painted[9, 6]
    assert painted[8, 10] and painted[9, 10] and painted[8, 9]
    # Output corruption must be rejected, including filling a genuine carrier gap.
    damaged = generate_puzzle(91, 2, 1)
    pair = damaged["test"][0]
    info = _parse(pair["input"])
    _, occupied, footprint = _paint(info)
    r, c = map(int, np.argwhere(footprint & ~occupied)[0])
    out = np.rot90(pair["output"], info["rotation"])
    out[r, c] = info["roles"][3]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("filled carrier gap accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "motifs": len(motifs), "shapes": len(shapes), "palettes": len(palettes),
            "normalized_structures": len(normalized), "densities": sorted(densities),
            "directions": sorted(directions), "bar_counts": sorted(bar_numbers),
            "marker_counts": sorted(unary), "phase_sensitive_pairs": phase_sensitive,
            "explicit_count_combinations": explicit, "semantic_checks": "generator-owned"}
