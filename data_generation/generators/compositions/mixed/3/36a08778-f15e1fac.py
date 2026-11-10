"""Tick-steered colored branching rays. Only NumPy and the standard library.

Canonical coordinates: signal at top, original control ticks at left. Colors
are normalized by visible roles; signal ranks follow first occurrence away
from the control corner. Every tick shifts the *current* active pattern before
collision detection, including branches inherited from earlier layers. A hit
run chooses the earliest-ranked incident color for its crosspiece and flanks;
all competing additions use that same priority. Original pixels are immutable.
No parent code, example labels, seed state, or external files enter solve().
"""
from __future__ import annotations

import hashlib
import numpy as np


SHORTCUTS = ("no_shift", "no_branch", "ignore_barriers", "late_tick", "temporary_shift",
             "reset", "reverse_priority", "one_bar", "wrap", "all_bars")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _view(grid, rotation, mirror):
    a = np.rot90(grid, rotation)
    return np.fliplr(a) if mirror else a


def _unview(grid, rotation, mirror):
    a = np.fliplr(grid) if mirror else grid
    return np.rot90(a, -rotation).copy()


def _recognize(grid):
    """Find the unique signal/control corner from input geometry alone."""
    values, counts = np.unique(grid, return_counts=True)
    if len(values) < 5 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("expected a dominant background and at least five roles/colors")
    background = int(values[counts.argmax()])
    candidates = []
    for rotation in range(4):
        for mirror in (False, True):
            a = _view(grid, rotation, mirror)
            h, w = a.shape
            if h < 4 or w < 4 or a[0, 0] != background or a[0, -1] != background:
                continue
            signals = [int(c) for c in np.unique(a[0]) if c != background]
            if not 2 <= len(signals) <= 4:
                continue
            if any(np.any(a[1:] == color) for color in signals):
                continue
            controls = set(map(int, np.unique(a[1:, 0]))) - {background}
            if len(controls) != 1 or a[-1, 0] != background:
                continue
            control = controls.pop()
            if control in signals or np.any(a[:, 1:] == control):
                continue
            obstacles = set(map(int, values)) - {background, control, *signals}
            if len(obstacles) != 1:
                continue
            barrier = obstacles.pop()
            if not np.any(a[1:-1, 1:-1] == barrier):
                continue
            # The order is visible, applies to colors, and survives branching.
            ordered = list(dict.fromkeys(int(c) for c in a[0] if int(c) in signals))
            palette = np.array([background, barrier, control, *ordered], dtype=np.int8)
            normalized = np.empty(a.shape, dtype=np.int8)
            for i, color in enumerate(palette):
                normalized[a == color] = i
            candidates.append((normalized, palette, rotation, mirror))
    if len(candidates) != 1:
        raise ValueError("signal and control edges must have a unique interpretation")
    return candidates[0]


def _runs(row):
    result = []
    c = 0
    while c < len(row):
        if row[c] != 1:
            c += 1
            continue
        left = c
        while c + 1 < len(row) and row[c + 1] == 1:
            c += 1
        result.append((left, c))
        c += 1
    return result


def _advance(row, active, reverse=False):
    """Simultaneous collisions on immutable transverse runs, then merge flanks."""
    choose = max if reverse else min
    following = {c: color for c, color in active.items() if row[c] != 1}
    hits = []
    for left, right in _runs(row):
        incident = [(c, color) for c, color in active.items() if left <= c <= right]
        if incident:
            owner = choose(color for _, color in incident)
            hits.append((left, right, owner, incident))
    for left, right, owner, _ in hits:
        for c in (left - 1, right + 1):
            if 0 <= c < len(row):
                following[c] = choose(following.get(c, owner), owner)
    return following, hits


def _simulate(a, shortcut=None):
    """Lossless state is kept only in memory; no execution traces are saved.

    Additions form a priority paint layer. Original nonbackground pixels are
    immutable. This makes overlapping simultaneous crosspieces order independent.
    Paint never feeds back into the active-ray state or the original tick mask.
    """
    h, w = a.shape
    initial = {int(c): int(a[0, c]) for c in np.flatnonzero(a[0] >= 3)}
    active = initial.copy()
    ticks = set(map(int, np.flatnonzero(a[:, 0] == 2)))
    reverse = shortcut == "reverse_priority"
    choose = max if reverse else min
    empty = -1 if reverse else 10
    paint = np.full(a.shape, empty, dtype=np.int8)
    stats = dict(events=0, shifted_events=0, simultaneous=0, multi_color_hits=0,
                 clipped=0, two_flanks=0, edge_events=0, interrupted=0,
                 merges=0, inert=0, ticks=len(ticks), signals=len(initial),
                 signal_colors=len(set(initial.values())), depth=h, width=w)
    shift_count = 0
    reached = set()

    def draw(r, c, color):
        if a[r, c] == 0:
            paint[r, c] = choose(int(paint[r, c]), color)

    def shift(state):
        if shortcut == "wrap":
            return {(c + 1) % w: color for c, color in state.items()}
        stats["clipped"] += sum(c + 1 == w for c in state)
        return {c + 1: color for c, color in state.items() if c + 1 < w}

    for r in range(1, h):
        tick = r in ticks
        shift_count += int(tick)
        if shortcut == "reset":
            active = {c + shift_count: color for c, color in initial.items()
                      if c + shift_count < w}
        elif tick and shortcut not in ("no_shift", "late_tick"):
            active = shift(active)
        following, hits = _advance(a[r], active, reverse)
        if shortcut == "ignore_barriers":
            # The B-only alternative: copy the shifted pattern through every
            # section, hiding paint under immutable barriers but never splitting.
            following, hits = active.copy(), []
        elif shortcut == "no_branch":
            following = {c: color for c, color in active.items() if a[r, c] != 1}
            hits = []
        elif shortcut == "one_bar" and hits:
            hits = hits[:1]
            following = {c: color for c, color in active.items() if a[r, c] != 1}
            for left, right, owner, _ in hits:
                for c in (left - 1, right + 1):
                    if 0 <= c < w:
                        following[c] = choose(following.get(c, owner), owner)
        elif shortcut == "all_bars":
            known = {(left, right) for left, right, _, _ in hits}
            for left, right in _runs(a[r]):
                if (left, right) not in known:
                    hits.append((left, right, 3, []))
                    for c in (left - 1, right + 1):
                        if 0 <= c < w:
                            following[c] = choose(following.get(c, 3), 3)
        stats["simultaneous"] += len(hits) >= 2
        for left, right, owner, incident in hits:
            reached.add((r, left, right))
            stats["events"] += 1
            stats["shifted_events"] += shift_count > 0
            stats["multi_color_hits"] += len({color for _, color in incident}) >= 2
            stats["two_flanks"] += left > 0 and right < w - 1
            stats["edge_events"] += right == w - 1 or left == 0
            span = range(max(0, left - 1), min(w, right + 2))
            stats["interrupted"] += any(a[r - 1, c] == 1 for c in span)
            for c in span:
                draw(r - 1, c, owner)
        arrivals = sum(c in following for left, right, _, _ in hits
                       for c in (left - 1, right + 1) if 0 <= c < w)
        stats["merges"] += max(0, len(active) + arrivals - len(following) -
                               sum(len(incident) for _, _, _, incident in hits))
        for c, color in following.items():
            draw(r, c, color)
        active = following
        if tick and shortcut == "late_tick":
            active = shift(active)
        if tick and shortcut == "temporary_shift":
            active = {c - 1: color for c, color in active.items() if c > 0}
    out = a.copy()
    mask = (a == 0) & (paint != empty)
    out[mask] = paint[mask]
    stats["inert"] = sum((r, left, right) not in reached for r in range(h)
                         for left, right in _runs(a[r]))
    stats["changed"] = int(np.count_nonzero(out != a))
    return out, stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles/orientation from this input and sweep its immutable cells."""
    a, palette, rotation, mirror = _recognize(_grid(input_grid))
    out, _ = _simulate(a)
    return _unview(palette[out], rotation, mirror)


def _differences(a, expected):
    return {name for name in SHORTCUTS
            if not np.array_equal(_simulate(a, name)[0], expected)}


def _place_run(rng, row, pivot, length=None):
    """Sample a transverse interval containing pivot, leaving distinct runs apart."""
    w = len(row)
    if not 1 <= pivot < w or row[pivot] != 0:
        return
    lo = hi = pivot
    while lo > 1 and row[lo - 1] == 0 and (lo < 2 or row[lo - 2] != 1):
        lo -= 1
    while hi + 1 < w and row[hi + 1] == 0 and (hi + 2 == w or row[hi + 2] != 1):
        hi += 1
    available = hi - lo + 1
    if length is None:
        mode = int(rng.integers(4))
        length = (1 if mode == 0 else min(available, int(rng.integers(2, 6)))
                  if mode == 1 else max(1, available - int(rng.integers(3)))
                  if mode == 2 else int(rng.integers(1, available + 1)))
    length = min(length, available)
    left = int(rng.integers(max(lo, pivot - length + 1), min(pivot, hi - length + 1) + 1))
    row[left:left + length] = 1


def _positions(rng, w, count, style):
    available = w - 2
    if style == 0:
        start = int(rng.integers(1, w - count))
        return list(range(start, start + count))
    if style == 1:
        gap = int(rng.integers(1, (available - 1) // (count - 1) + 1))
        start = int(rng.integers(1, w - 1 - gap * (count - 1)))
        return [start + i * gap for i in range(count)]
    if style == 2 and count >= 3:
        n = int(rng.integers(1, count))
        gap = int(rng.integers(2, min(6, available - count + 2)))
        total = count + gap - 1
        start = int(rng.integers(1, w - total))
        return list(range(start, start + n)) + list(range(start + n + gap - 1, start + total))
    return sorted(map(int, rng.choice(np.arange(1, w - 1), count, replace=False)))


def _construct(rng, kind, large=False):
    """Compose a causal prefix witness and a fresh event-driven suffix.

    Witnesses fix relations (separate hits or shared hit), never whole grids.
    General examples use runs, combs, groups, or irregular boundary patterns.
    """
    # Scale-transfer queries are strictly deeper than every training example,
    # whose canonical forward length is at most 24, regardless of orientation.
    h = int(rng.integers(25, 31) if large else rng.integers(13, 25))
    w = int(rng.integers(21, 31) if large else rng.integers(14, 29))
    a = np.zeros((h, w), dtype=np.int8)
    first = int(rng.integers(2, 5))
    second = first + int(rng.integers(2, 5))
    ticks = {first, second}
    extra = int(rng.integers(0, min(5, h - second - 1)))
    if extra:
        ticks.update(map(int, rng.choice(np.arange(second + 1, h - 1), extra, replace=False)))
    a[sorted(ticks), 0] = 2
    start = int(rng.integers(2, min(6, w - 8)))
    if kind == "split":
        gap = int(rng.integers(2, min(7, w - start - 4)))
        positions = [start, start + gap, w - 2]
        ncolors = int(rng.integers(2, 4))
    elif kind == "merge":
        gap = int(rng.integers(2, min(8, w - start - 4)))
        positions = [start, start + gap]
        extras = int(rng.integers(0, 4))
        free = [c for c in range(1, w - 1) if c not in positions]
        positions += list(map(int, rng.choice(free, extras, replace=False)))
        positions.sort()
        ncolors = int(rng.integers(2, min(4, len(positions)) + 1))
    else:
        count = int(rng.integers(2, min(9, w - 3)))
        positions = _positions(rng, w, count, int(rng.integers(4)))
        ncolors = int(rng.integers(2, min(4, count) + 1))
    # First occurrence order equals normalized priority, with repeated colors.
    colors = list(range(3, 3 + ncolors))
    colors += list(map(int, rng.integers(3, 3 + ncolors, len(positions) - ncolors)))
    a[0, positions] = colors
    event = first + int(rng.integers(0, 2))
    if kind == "split":
        for c in positions[:2]:
            a[event, c + 1] = 1
        # A same-color distractor can interrupt the upstream crosspiece.
        a[event - 1, start + 1] = 1
    elif kind == "merge":
        # Span two visibly different incident colors after the first shift.
        left, right = positions[0] + 1, positions[max(1, ncolors - 1)] + 1
        a[event, left:right + 1] = 1
        a[event - 1, right] = 1
    active = {c: int(a[0, c]) for c in positions}
    suffix_start = second + 1 if kind != "general" else first
    density = float(rng.uniform(.16, .42))
    for r in range(1, h):
        if r in ticks:
            active = {c + 1: color for c, color in active.items() if c + 1 < w}
        if r >= suffix_start and active and rng.random() < density:
            pivots = [c for c in active if c > 0]
            rng.shuffle(pivots)
            for c in pivots[:int(rng.integers(1, min(4, len(pivots)) + 1))] if pivots else []:
                _place_run(rng, a[r], c)
        # Unreached bars share the causal length sampler; geometry alone does
        # not decide whether a barrier matters. One-cell stems can connect rows.
        if r > event + 1 and rng.random() < .25:
            free = [c for c in range(1, w) if c not in active and a[r, c] == 0]
            if free:
                c = int(rng.choice(free))
                trial = a[r].copy()
                _place_run(rng, trial, c)
                if all(trial[x] == a[r, x] for x in active):
                    a[r] = trial
        active, _ = _advance(a[r], active)
    return a


def _key(a):
    return a.shape, a.tobytes()


def _make_pair(rng, kind, orientation, palette_order, seen, allowed=None, large=False):
    # No files, worker pools, unbounded retries, or retained candidate grids.
    for _ in range(160):
        a = _construct(rng, kind, large)
        expected, stats = _simulate(a)
        if stats["shifted_events"] == 0 or stats["two_flanks"] == 0:
            continue
        if kind == "split" and not (stats["simultaneous"] and stats["clipped"]):
            continue
        if kind == "merge" and not stats["multi_color_hits"]:
            continue
        differences = _differences(a, expected)
        if not {"no_shift", "no_branch", "ignore_barriers", "late_tick", "reset"} <= differences:
            continue
        if allowed is not None and not differences <= allowed:
            continue
        color_count = int(a.max()) + 1
        palette = np.array(palette_order[:color_count], dtype=np.int8)
        inp = _unview(palette[a], *orientation)
        if _key(inp) in seen:
            continue
        # Re-recognition, including normalization, is checked before accepting.
        try:
            normalized, _, _, _ = _recognize(inp)
        except ValueError:
            continue
        if not np.array_equal(normalized, a):
            continue
        return {"input": inp, "output": _unview(palette[expected], *orientation)}, differences
    raise RuntimeError("bounded construction exhausted; no silent rule-changing fallback")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen, demonstrated = set(), set()
    base_palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    orientations = [(r, f) for r in range(4) for f in (False, True)]
    rng.shuffle(orientations)
    # Tests either transfer orientation or extend scale/composition. Neither
    # regime changes the rule, priority convention, or marker recognizability.
    transfer = bool(rng.integers(2))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("split" if i == 0 else "merge" if i == 1 else "general") if split == "train" else "general"
            orientation = orientations[i % 2] if split == "train" else orientations[2 + i] if transfer else orientations[i % 2]
            palette = list(map(int, rng.permutation(10))) if vary_palette else base_palette
            pair, differences = _make_pair(rng, kind, orientation, palette, seen,
                                           demonstrated if split == "test" else None,
                                           large=split == "test" and not transfer)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                demonstrated |= differences
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, demonstrated = set(), set()
    training_stats = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output dimensions")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a, palette, rotation, mirror = _recognize(inp)
            expected, stats = _simulate(a)
            _require(np.array_equal(out, _unview(palette[expected], rotation, mirror)), "incorrect output")
            _require(np.all(out[inp != palette[0]] == inp[inp != palette[0]]), "original marker/barrier changed")
            _require(set(map(int, out[out != inp])) <= set(map(int, palette[3:])), "added nonsignal color")
            _require(stats["changed"] > 0 and stats["shifted_events"] > 0, "no causal shift/branch interaction")
            differences = _differences(a, expected)
            _require({"no_shift", "no_branch", "ignore_barriers", "late_tick", "reset"} <= differences,
                     "parent necessity or state not visible")
            if split == "train":
                demonstrated |= differences
                training_stats.append(stats)
            else:
                _require(differences <= demonstrated, "test-changing shortcut still fits all demonstrations")
    _require(any(s["simultaneous"] for s in training_stats), "training needs simultaneous runs")
    _require(any(s["multi_color_hits"] for s in training_stats), "training needs color arbitration")
    _require(any(s["clipped"] for s in training_stats), "training needs shift clipping")
    _require(any(s["inert"] for s in training_stats), "training needs unreached barriers")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _micro_tests():
    # Hand-specified expectations establish shift-before-collision, priority,
    # inherited flanks, and persistence without comparing two copies of solve.
    a = np.zeros((9, 12), dtype=np.int8)
    a[0, 2], a[0, 5] = 3, 4
    a[2, 0] = a[6, 0] = 2
    a[2, 3:7] = 1
    out, stats = _simulate(a)
    _require(np.all(out[1, 2:8] == 3), "crosspiece ownership")
    _require(out[2, 2] == out[2, 7] == 3, "branch flank ownership")
    _require(out[3, 2] == out[5, 7] == 3, "branch persistence")
    _require(out[6, 3] == out[6, 8] == out[8, 8] == 3, "descendant tick shift")
    _require(out[6, 2] == out[6, 7] == 0, "tick is inclusive")
    _require(stats["multi_color_hits"] == 1, "same run handled once")
    # Separate runs share a flank; the earlier border color wins that merge.
    b = np.zeros((7, 12), dtype=np.int8)
    b[0, 2], b[0, 4] = 3, 4
    b[2, 0] = 2
    b[2, 3] = b[2, 5] = 1
    ob, sb = _simulate(b)
    _require(ob[2, 4] == ob[5, 4] == 3 and ob[2, 6] == 4, "shared flank merge")
    _require(sb["simultaneous"] == 1, "simultaneous run processing")
    # Drop, do not wrap or reintroduce, a signal shifted out of bounds.
    c = np.zeros((8, 10), dtype=np.int8)
    c[0, 2], c[0, 8] = 3, 4
    c[2, 0] = c[4, 0] = 2
    c[6, 4] = 1
    oc, sc = _simulate(c)
    _require(oc[2, 9] == oc[3, 9] == 4 and not np.any(oc[4:] == 4), "permanent clipping")
    _require(sc["clipped"] == 1, "clipping count")
    # Outside the sampler's witness ranges: one tick at depth one and an
    # immediate collision whose crosspiece reaches the original signal edge.
    # The original lower-priority signal pixel must survive that crosspiece.
    d = np.zeros((6, 7), dtype=np.int8)
    d[0, 2], d[0, 4], d[1, 0] = 3, 4, 2
    d[1, 3:6] = 1
    expected = np.array([[0, 0, 3, 3, 4, 3, 3],
                         [2, 0, 3, 1, 1, 1, 3],
                         [0, 0, 3, 0, 0, 0, 3],
                         [0, 0, 3, 0, 0, 0, 3],
                         [0, 0, 3, 0, 0, 0, 3],
                         [0, 0, 3, 0, 0, 0, 3]], dtype=np.int8)
    palette = np.array([7, 0, 9, 8, 2], dtype=np.int8)
    for rotation in range(4):
        for mirror in (False, True):
            inp = _unview(palette[d], rotation, mirror)
            want = _unview(palette[expected], rotation, mirror)
            _require(np.array_equal(solve(inp.tolist()), want), "input-only boundary regression")


def self_test():
    """42 puzzles in a streaming loop; no sample corpus or logs written."""
    _micro_tests()
    shapes, layouts, palettes, orientations, signatures = set(), set(), set(), set(), set()
    signal_counts, tick_counts, collision_counts, lengths = set(), set(), set(), set()
    feature_totals = {s: {name: 0 for name in ("simultaneous", "multi_color_hits", "clipped",
                     "edge_events", "interrupted", "inert")} for s in ("train", "test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, palette, rotation, mirror = _recognize(pair["input"])
                out, stats = _simulate(a)
                shapes.add(a.shape)
                layouts.add(hashlib.sha256(a.tobytes() + bytes(a.shape)).hexdigest())
                palettes.add(tuple(map(int, palette)))
                orientations.add((rotation, mirror))
                signal_counts.add(stats["signals"])
                tick_counts.add(stats["ticks"])
                collision_counts.add(stats["events"])
                lengths.update(right - left + 1 for row in a for left, right in _runs(row))
                for name in feature_totals[split]:
                    feature_totals[split][name] += bool(stats[name])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solver disagreement")
        if seed < 4:
            _require(_digest(generate_puzzle(seed)) == _digest(puzzle), "nondeterminism")
    _require(len(signatures) == 42 and len(layouts) >= 160, "structural diversity")
    _require(len(shapes) >= 50 and len(palettes) >= 30, "shape/palette diversity")
    _require(len(orientations) == 8, "dihedral coverage")
    _require(len(signal_counts) >= 5 and len(tick_counts) >= 4 and len(collision_counts) >= 7, "count variation")
    _require(len(lengths) >= 12 and max(lengths) >= 20, "barrier geometry variation")
    for split in feature_totals:
        _require(all(n > 0 for n in feature_totals[split].values()), f"missing {split} feature")
    # All count contracts, tested with a different seed, stay bounded (30 calls).
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(86019, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                normalized_layouts=len(layouts), canonical_shapes=len(shapes),
                palettes=len(palettes), orientations=len(orientations),
                signal_counts=sorted(signal_counts), tick_counts=sorted(tick_counts),
                collision_counts=sorted(collision_counts), barrier_lengths=sorted(lengths),
                features=feature_totals, independent_solver=False)
