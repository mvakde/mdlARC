"""Tick-sheared multicolor boundary signal, observed through a parity crop.

Only NumPy and the standard library are used. Generation writes no artifacts.
The solver parses roles from its input; it never consults generation metadata.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


ORIENTATIONS = (
    ("top", "left"), ("top", "right"),
    ("bottom", "left"), ("bottom", "right"),
    ("left", "top"), ("left", "bottom"),
    ("right", "top"), ("right", "bottom"),
)
CHECK_SEEDS = (
    0, 821909736, 673515023, 616426407, 496466710, 570769496,
    452134711, 800134395, 129313056, 868159728, 406484721,
    435350579, 461573107, 260742652, 731325061, 161467896,
    157608729, 924038002, 117464390, 373041684, 752874837,
    43347685, 113859014, 348555636, 501852430, 17912071,
    629086306, 868132161, 787173041, 339010121, 292211803,
    598474101, 303883711, 947578312, 400775045, 400777102,
    38303974, 627823657, 82324671, 441895776, 484796610, 341802196,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, low, high, name):
    if (not isinstance(value, numbers.Integral)
            or isinstance(value, (bool, np.bool_)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _array(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= d <= 30 for d in grid.shape)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _edge(grid, side):
    if side == "top":
        return grid[0, :]
    if side == "bottom":
        return grid[-1, :]
    if side == "left":
        return grid[:, 0]
    return grid[:, -1]


def _parse(input_grid):
    grid = _array(input_grid)
    n = grid.shape[0]
    if grid.shape != (n, n) or n < 4:
        raise ValueError("family inputs must be square with side 4..30")
    background = int(grid[1, 1])
    if np.any(grid[1:-1, 1:-1] != background):
        raise ValueError("interior must be uniform background")
    if any(grid[r, c] != background for r, c in
           ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))):
        raise ValueError("original corners must be background")
    occupied = {}
    for side in ("top", "bottom", "left", "right"):
        edge = _edge(grid, side)
        hues = set(int(x) for x in edge if x != background)
        if hues:
            occupied[side] = hues
    signal_sides = [side for side, hues in occupied.items() if len(hues) >= 2]
    control_sides = [side for side, hues in occupied.items() if len(hues) == 1]
    if len(occupied) != 2 or len(signal_sides) != 1 or len(control_sides) != 1:
        raise ValueError("need one multicolor signal edge and one monochrome tick edge")
    signal, control = signal_sides[0], control_sides[0]
    if (signal, control) not in ORIENTATIONS:
        raise ValueError("occupied edges must be adjacent")
    control_color = next(iter(occupied[control]))
    if control_color in occupied[signal]:
        raise ValueError("control and signal hues must be disjoint")
    vector = _edge(grid, signal).copy()
    coordinates = np.flatnonzero(_edge(grid, control) != background)
    ticks = tuple(sorted(int(n - 1 - x if signal in ("bottom", "right") else x)
                         for x in coordinates))
    delta = 1 if control in ("left", "top") else -1
    return grid, background, signal, control, control_color, vector, ticks, delta


def _translate(vector, amount, background, wrap=False):
    if wrap:
        return np.roll(vector, amount)
    result = np.full_like(vector, background)
    n = len(vector)
    if 0 <= amount < n:
        result[amount:] = vector[:n - amount]
    elif -n < amount < 0:
        result[:n + amount] = vector[-amount:]
    return result


def _field(parsed, mode="normal"):
    """Full field in RAM; modes other than normal are diagnostic ablations."""
    grid, bg, signal, _, _, initial, ranks, delta = parsed
    n = len(initial)
    active = initial.copy()
    result = np.full_like(grid, bg)
    ticks = set(ranks)
    shifted_once = False
    for rank in range(n):
        if mode == "total":
            active = initial if rank == 0 else _translate(initial, len(ranks) * delta, bg)
        if rank in ticks and mode not in ("none", "late", "total"):
            if mode != "once" or not shifted_once:
                active = _translate(active, -delta if mode == "toward" else delta,
                                    bg, wrap=mode == "wrap")
                shifted_once = True
        coordinate = n - 1 - rank if signal in ("bottom", "right") else rank
        if signal in ("top", "bottom"):
            result[coordinate, :] = active
        else:
            result[:, coordinate] = active
        if rank in ticks and mode == "late":
            active = _translate(active, delta, bg)
    # Original markers win over propagated colors; the commands never move.
    original = grid != bg
    result[original] = grid[original]
    return result


def _crop(field, parity=None):
    parity = field.shape[0] % 2 if parity is None else parity
    return field[:2, :2].copy() if parity else field[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Propagate with inclusive ticks, preserve markers, then crop by original n."""
    return _crop(_field(_parse(input_grid)))


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _different(a, b):
    return not np.array_equal(a, b)


def _effects(parsed):
    field = _field(parsed)
    answer = _crop(field)
    result = {
        "corner": _different(answer, _crop(field, 1 - len(field) % 2)),
        "raw": _different(answer, _crop(parsed[0])),
    }
    for mode in ("none", "late", "total", "once", "toward", "wrap"):
        result[mode] = _different(answer, _crop(_field(parsed, mode)))
    return result


def _ranks(rng, n, count, forced=()):
    """Random, evenly spread, burst, early, late, or end-separated commands."""
    mode = int(rng.integers(6))
    available = np.arange(1, n - 1)
    if mode == 0:
        preferred = rng.permutation(available).tolist()
    elif mode == 1:
        preferred = np.rint(np.linspace(1, n - 2, count)).astype(int).tolist()
    elif mode == 2:
        start = int(rng.integers(1, n - count))
        preferred = list(range(start, start + count))
    elif mode == 3:
        preferred = list(range(1, n - 1))
    elif mode == 4:
        preferred = list(range(n - 2, 0, -1))
    else:
        preferred = []
        for i in range(1, n - 1):
            preferred += [i, n - 1 - i]
    selected = set(int(x) for x in forced)
    for rank in preferred + rng.permutation(available).tolist():
        if len(selected) == count:
            break
        selected.add(int(rank))
    return tuple(sorted(selected))


def _positions(rng, n):
    mode = int(rng.integers(4))
    count = int(rng.integers(2, n - 1))
    if mode == 0:
        start = int(rng.integers(1, n - count))
        return set(range(start, start + count))
    if mode == 1 and n >= 6:
        gap = int(rng.integers(2, min(4, n - 3) + 1))
        count = int(rng.integers(2, (n - 3) // gap + 2))
        span = (count - 1) * gap
        start = int(rng.integers(1, n - 1 - span))
        return {start + i * gap for i in range(count)}
    if mode == 2 and n >= 7:
        # Two runs separated by at least one background cell.
        gap = int(rng.integers(2, n - 3))
        left = int(rng.integers(1, gap))
        right = int(rng.integers(gap + 1, n - 1))
        return set(range(left, gap)) | set(range(gap + 1, right + 1))
    return set(int(x) for x in rng.choice(np.arange(1, n - 1), count, replace=False))


def _input(rng, n, profile, orientation=None):
    odd = bool(n % 2)
    forced_positions = ()
    forced_ticks = ()
    if profile == "local":
        # Tick 1 changes row 1, later ticks must not affect this near window.
        orientation = ("top", "right" if odd else "left")
        count = 2
        forced_ticks = (1,)
        forced_positions = (1, 2) if odd else (n - 2, n - 3)
    elif profile == "history":
        # The far crop sees multiple cumulative shifts, including a tick on it.
        orientation = ("right" if odd else "left", "bottom")
        count = int(rng.integers(2, min(4, n - 3) + 1))
        forced_ticks = (n - 2,)
        forced_positions = (count, count + 1)
    elif profile == "boundary":
        # Downward-moving bottom tracks must disappear rather than wrap to top.
        orientation = ("right" if odd else "left", "top")
        count = 3
        forced_ticks = (n - 2,)
        forced_positions = (1, n - 2, n - 3)
    elif profile == "bottom":
        orientation = ("bottom", "right" if odd else "left")
        count = int(rng.integers(2, min(4, n - 3) + 1))
        forced_ticks = (n - 2,)
        forced_positions = ((count, count + 1) if odd else
                            (n - 2 - count, n - 1 - count))
    else:
        if orientation is None:
            orientation = ORIENTATIONS[int(rng.integers(8))]
        count = int(rng.integers(1, min(8, n - 2) + 1))
        # Near windows can reveal a shift only at the first interior section.
        signal, control = orientation
        near = signal == "top" or (signal == "left" and odd) or (signal == "right" and not odd)
        if near:
            forced_ticks = (1,)
        if control == "top":
            forced_positions = (1,)
    ticks = _ranks(rng, n, count, forced_ticks)
    positions = sorted(_positions(rng, n) | set(forced_positions))
    palette = rng.permutation(10)
    bg, control_color = int(palette[0]), int(palette[1])
    hue_count = int(rng.integers(2, min(8, len(positions)) + 1))
    hues = palette[2:2 + hue_count]
    labels = np.arange(len(positions)) % hue_count
    if rng.random() < .5:
        rng.shuffle(labels)
    colors = {position: int(hues[label]) for position, label in zip(positions, labels)}
    # Witness colors differ even when other signal cells repeat either hue.
    anchors = forced_positions[:2] if len(forced_positions) >= 2 else positions[:2]
    colors[anchors[0]], colors[anchors[1]] = int(hues[0]), int(hues[1])
    grid = np.full((n, n), bg, dtype=np.int8)
    signal, control = orientation
    for pos, color in colors.items():
        _edge(grid, signal)[pos] = color
    for rank in ticks:
        pos = n - 1 - rank if signal in ("bottom", "right") else rank
        _edge(grid, control)[pos] = control_color
    return grid


def _unique_pair(rng, n, profile, used, orientation=None):
    required = {"corner", "none"}
    if profile == "local":
        required |= {"late", "total", "toward"}
    elif profile == "history":
        required |= {"late", "once"}
    elif profile == "boundary":
        required |= {"wrap"}
    for _ in range(96):
        grid = _input(rng, n, profile, orientation)
        if _key(grid) in used:
            continue
        parsed = _parse(grid)
        effects = _effects(parsed)
        if not all(effects[key] for key in required):
            continue
        output = _crop(_field(parsed))
        if profile in ("local", "history", "bottom"):
            signal_hues = set(int(x) for x in parsed[5] if x != parsed[1])
            if len(signal_hues & set(int(x) for x in output.flat)) < 2:
                continue
        used.add(_key(grid))
        return {"input": grid, "output": output}
    raise RuntimeError(f"bounded sampling exhausted: {n}/{profile}/{orientation}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible family member; explicit counts are mandatory ints."""
    if (not isinstance(seed, numbers.Integral) or isinstance(seed, (bool, np.bool_))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    a = _integer(num_train, 2, 10, "num_train")
    b = _integer(num_test, 1, 4, "num_test")
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    first_parity = int(rng.integers(2))
    profiles = ("local", "history", "boundary", "bottom")
    used, sizes = set(), {0: set(), 1: set()}
    puzzle = {"train": [], "test": []}
    # Four consecutive lengths interleave the parities and include an even
    # length not divisible by four. A size threshold cannot fit these crops.
    teaching_sides = {0: [], 1: []}
    if a >= 4:
        start = int(rng.integers(6, 12))
        for n in range(start, start + 4):
            teaching_sides[n % 2].append(n)
        for values in teaching_sides.values():
            rng.shuffle(values)
    for index in range(a):
        parity = first_parity if index % 2 == 0 else 1 - first_parity
        if index < 4 and a >= 4:
            n = teaching_sides[parity].pop()
        else:
            choices = [n for n in range(6, 15) if n % 2 == parity and n not in sizes[parity]]
            if not choices:
                choices = [n for n in range(4, 15) if n % 2 == parity]
            n = int(rng.choice(choices))
        sizes[parity].add(n)
        profile = profiles[index] if index < 4 else "random"
        puzzle["train"].append(_unique_pair(rng, n, profile, used))
    rng.shuffle(puzzle["train"])
    train_orientations = {(_parse(p["input"])[2:4]) for p in puzzle["train"]}
    test_parity = int(rng.integers(2))
    for index in range(b):
        parity = (test_parity + index) % 2
        n = int(rng.choice([x for x in range(15, 31) if x % 2 == parity]))
        # Alternate shared, held-out, and unrestricted orientation regimes.
        regime = (int(seed) + index) % 3
        candidates = [o for o in ORIENTATIONS if
                      (o in train_orientations if regime == 0 else
                       o not in train_orientations if regime == 1 else True)]
        if not candidates:
            candidates = list(ORIENTATIONS)
        orientation = candidates[int(rng.integers(len(candidates)))]
        puzzle["test"].append(_unique_pair(rng, n, "random", used, orientation))
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_field(parsed, field):
    """Check track positions directly from inclusive tick-prefix counts."""
    grid, bg, signal, _, control_color, vector, ticks, delta = parsed
    n = len(vector)
    _require(field.shape == grid.shape, "propagation changed the square dimensions")
    mask = grid != bg
    _require(np.array_equal(field[mask], grid[mask]), "original marker lost")
    _require(np.array_equal(field == control_color, grid == control_color),
             "control ticks grew or moved")
    seeds = [(int(p), int(vector[p])) for p in np.flatnonzero(vector != bg)]
    # This is a local consistency check, not an independent semantic solver.
    for rank in range(n):
        shifts = sum(t <= rank for t in ticks)
        coordinate = n - 1 - rank if signal in ("bottom", "right") else rank
        expected = np.full(n, bg, dtype=np.int8)
        for position, color in seeds:
            target = position + delta * shifts
            if 0 <= target < n:
                expected[target] = color
        original = grid[coordinate, :] if signal in ("top", "bottom") else grid[:, coordinate]
        expected[original != bg] = original[original != bg]
        actual = field[coordinate, :] if signal in ("top", "bottom") else field[:, coordinate]
        _require(np.array_equal(actual, expected), "track color, unit shift, or clipping failed")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            parsed = _parse(pair["input"])
            grid, bg, signal, control, _, vector, ticks, delta = parsed
            output = _array(pair["output"])
            _require(output.shape == (2, 2), "output must be 2x2")
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            field = _field(parsed)
            _check_field(parsed, field)
            _require(np.array_equal(output, _crop(field)), "incorrect answer")
            effects = _effects(parsed)
            _require(all(effects[k] for k in ("corner", "none")),
                     "pair hides a required parent contribution")
            if split == "train":
                train.append((len(grid), signal, control, effects))
    _require({n % 2 for n, *_ in train} == {0, 1}, "both train parities needed")
    for parity in (0, 1):
        witnesses = [f for n, _, _, f in train if n % 2 == parity]
        _require(any(f["corner"] and f["none"] for f in witnesses), "missing joint witness")
    _require(any(f["late"] and f["total"] for _, _, _, f in train),
             "need a local timing and persistent-state witness")
    _require(any(f["once"] for _, _, _, f in train), "need multiple accumulated shifts")
    _require(any(f["raw"] for _, _, _, f in train), "raw input crop fits every demonstration")
    if a >= 3:
        _require(any(f["wrap"] for _, _, _, f in train), "need visible clipping witness")
    if a >= 4:
        _require(all(len({n for n, _, _, _ in train if n % 2 == p}) >= 2 for p in (0, 1)),
                 "four demonstrations need two sizes per parity")
        odd = [n for n, _, _, _ in train if n % 2]
        even = [n for n, _, _, _ in train if not n % 2]
        _require(min(odd) < max(even) and min(even) < max(odd), "parities must overlap in scale")
    _require(max(len(np.asarray(p["input"])) for p in puzzle["train"]) <
             min(len(np.asarray(p["input"])) for p in puzzle["test"]), "missing scale transfer")


def _digest(puzzle, normalize=False):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                if normalize:
                    names = {}
                    digest.update(bytes(names.setdefault(int(v), len(names)) for v in grid.flat))
                else:
                    digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, all legal count combinations, bounded memory, no file writes."""
    fingerprints, structures, sizes, palettes = set(), set(), set(), set()
    orientations = {"train": set(), "test": set()}
    tick_counts, seed_counts, hue_counts, clip_counts = set(), set(), set(), set()
    test_parities = set()
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic puzzle")
        fingerprints.add(_digest(puzzle))
        structures.add(_digest(puzzle, normalize=True))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                parsed = _parse(pair["input"])
                grid, bg, signal, control, _, vector, ticks, delta = parsed
                _require(np.array_equal(solve(grid), pair["output"]), "solve disagreement")
                sizes.add(len(grid))
                palettes.add(tuple(int(x) for x in np.unique(grid)))
                orientations[split].add((signal, control))
                positions = np.flatnonzero(vector != bg)
                seed_counts.add(len(positions))
                hue_counts.add(len(set(int(vector[x]) for x in positions)))
                tick_counts.add(len(ticks))
                clip_counts.add(sum(not 0 <= x + len(ticks) * delta < len(grid) for x in positions))
                if split == "test":
                    test_parities.add(len(grid) % 2)
    _require(len(fingerprints) == len(structures) == 42, "42-seed structural diversity")
    _require(len(sizes) >= 20 and min(sizes) <= 6 and max(sizes) >= 28, "size range")
    _require(len(palettes) >= 30 and max(hue_counts) >= 6, "palette role variation")
    _require(orientations["train"] == orientations["test"] == set(ORIENTATIONS), "orientation coverage")
    _require(len(seed_counts) >= 10 and len(tick_counts) >= 6, "count variation")
    _require(0 in clip_counts and max(clip_counts) >= 3, "clipping variation")
    _require(test_parities == {0, 1}, "both test parities")
    edge_rng = np.random.default_rng(4242)
    for n in (4, 5, 30):
        for orientation in ORIENTATIONS:
            grid = _input(edge_rng, n, "random", orientation)
            parsed = _parse(grid)
            _check_field(parsed, _field(parsed))
            _require(solve(grid).shape == (2, 2), "domain boundary crop")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(20261023 + a * 10 + b, a, b)
            validate_generated_puzzle(puzzle)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4.0, 1), (True, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotation/reflection changes the propagation frame, but parity still crops
    # the absolute top corner. Check the full field before taking that crop.
    example = generate_puzzle(271828)["train"][0]["input"]
    field = _field(_parse(example))
    for turns in range(4):
        rotated = np.rot90(example, turns)
        _require(np.array_equal(_field(_parse(rotated)), np.rot90(field, turns)), "rotation")
        reflected = np.fliplr(rotated)
        _require(np.array_equal(_field(_parse(reflected)), np.fliplr(np.rot90(field, turns))), "reflection")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal explicit counts)")
