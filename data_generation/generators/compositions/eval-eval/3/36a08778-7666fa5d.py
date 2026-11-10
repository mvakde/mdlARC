"""Latent diagonal shutters: interval brackets activated by branching rays.

Only Python's standard library and NumPy are used.  Parsing and solving depend
exclusively on the supplied input.  Generation metadata is never passed to solve.
"""
from __future__ import annotations

import json
from collections import defaultdict
from numbers import Integral

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _cell(t, q):
    return (t - q) // 2, (t + q) // 2


def _inside(shape, t, q):
    if (t - q) % 2:
        return False
    r, c = _cell(t, q)
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _last_t(shape, q):
    return min(2 * (shape[0] - 1) + q, 2 * (shape[1] - 1) - q)


def _runs(a, color):
    """All maximal southeast runs, including disconnected runs on one axis."""
    strokes = []
    for r, c in np.argwhere(a == color):
        r, c = int(r), int(c)
        if r and c and a[r - 1, c - 1] == color:
            continue
        end_r, end_c = r, c
        while end_r + 1 < a.shape[0] and end_c + 1 < a.shape[1] and a[end_r + 1, end_c + 1] == color:
            end_r += 1
            end_c += 1
        strokes.append((c - r, r + c, end_r + end_c))
    return sorted(strokes)


def _parse(grid):
    """Recognize roles and one of four inward diagonal directions geometrically."""
    a = _grid(grid)
    values, counts = np.unique(a, return_counts=True)
    _require(len(values) == 3 and counts.max() > a.size / 2, "three roles and a majority field required")
    bg = int(values[np.argmax(counts)])
    candidates = []
    for rotation in range(4):
        b = np.rot90(a, rotation)
        for guide in map(int, values):
            if guide == bg:
                continue
            stubs = _runs(b, guide)
            # A stub is a proper prefix of a diagonal entering on top/left.
            if not stubs or any(lo != abs(q) or hi - lo < 2 or hi >= _last_t(b.shape, q)
                                for q, lo, hi in stubs):
                continue
            if len({q for q, _, _ in stubs}) != len(stubs):
                continue
            marker = next(int(x) for x in values if x not in (bg, guide))
            strokes = _runs(b, marker)
            if len(strokes) < 2 or any(hi - lo < 2 for _, lo, hi in strokes):
                continue
            if len({q for q, _, _ in strokes}) < 2:
                continue
            # A prefix cannot already be inside an active latent shutter.
            safe = True
            for q, lo, hi in stubs:
                for t in range(lo, hi + 1, 2):
                    active = {x for x, start, end in strokes if start <= t <= end}
                    if len(active) >= 2 and min(active) <= q <= max(active):
                        safe = False
            if safe:
                candidates.append(dict(rotation=rotation, grid=b, background=bg,
                                       guide=guide, marker=marker, stubs=stubs, strokes=strokes))
    _require(len(candidates) == 1, "input roles/direction are ambiguous")
    return candidates[0]


def _intervals(strokes, t, alternative=""):
    """Shutter support, inclusive of its two boundary axes."""
    active = sorted({q for q, lo, hi in strokes if lo <= t <= hi})
    if alternative == "centers":
        active = [q for q in active if (q - t) % 2 == 0]
    if alternative == "bounding":
        if min(lo for _, lo, _ in strokes) <= t <= max(hi for _, _, hi in strokes):
            active = sorted({q for q, _, _ in strokes})
        else:
            active = []
    if alternative == "physical":
        # The wrong rule responds only to actual marker pixels on this slice.
        qs = [q for q in active if (q - t) % 2 == 0]
        regions = []
        for q in qs:
            if regions and q == regions[-1][1] + 2:
                regions[-1] = (regions[-1][0], q)
            else:
                regions.append((q, q))
        return regions
    if alternative == "neighbors":
        axes = sorted({q for q, _, _ in strokes})
        regions = [(a, b) for a, b in zip(axes, axes[1:]) if a in active and b in active]
        merged = []
        for left, right in regions:
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], right)
            else:
                merged.append((left, right))
        return merged
    return [(active[0], active[-1])] if len(active) >= 2 else []


def _flanks(t, left, right):
    # The perpendicular diagonal has every second integer q.  Round OUTWARD.
    a, b = left - 1, right + 1
    a -= (a - t) % 2
    b += (t - b) % 2
    return a, b


def _simulate(info, alternative=""):
    a = info["grid"]
    guide_mask = np.zeros(a.shape, dtype=bool)
    reveal_mask = np.zeros(a.shape, dtype=bool)
    pending = defaultdict(set)
    for q, start, _ in info["stubs"]:
        pending[start].add(q)
    events = []
    merged = 0

    def paint(mask, t, q):
        if _inside(a.shape, t, q):
            mask[_cell(t, q)] = True

    for t in range(sum(a.shape) - 1):
        active = {q for q in pending[t] if _inside(a.shape, t, q)}
        regions = _intervals(info["strokes"], t, alternative)
        hits = defaultdict(list)
        following = set()
        for q in sorted(active):
            region = next(((left, right) for left, right in regions if left <= q <= right), None)
            if region is None:
                paint(guide_mask, t, q)
                following.add(q)
            else:
                hits[region].append(q)
        for (left, right), rays in hits.items():
            # Reveal ONLY this reached cross-section, never the whole panel.
            for q in range(left + 1, right):
                paint(reveal_mask, t, q)
            flank_left, flank_right = _flanks(t, left, right)
            flanks = []
            if alternative != "stop":
                for q in range(flank_left, flank_right + 1, 2):
                    paint(guide_mask, t - 2, q)
                for q in (flank_left, flank_right):
                    if _inside(a.shape, t, q):
                        merged += q in following
                        paint(guide_mask, t, q)
                        following.add(q)
                        flanks.append(q)
            events.append((t, left, right, tuple(rays), tuple(flanks)))
        pending[t + 2].update(following)
        if alternative == "all_slices":
            for left, right in regions:
                for q in range(left + 1, right):
                    paint(reveal_mask, t, q)

    out = a.copy()
    field = a == info["background"]
    out[field & guide_mask] = info["guide"]
    # A revealed shutter takes precedence over a later upstream crosspiece.
    out[field & reveal_mask] = info["marker"]
    stats = dict(events=events, merges=merged, guide_mask=guide_mask,
                 reveal_mask=reveal_mask, collisions=len(events),
                 two_flanks=sum(len(e[4]) == 2 for e in events),
                 clipped=sum(len(e[4]) < 2 for e in events),
                 multi_hit=sum(len(e[3]) > 1 for e in events),
                 added_guide=int(np.count_nonzero(field & guide_mask & ~reveal_mask)),
                 added_marker=int(np.count_nonzero(field & reveal_mask)))
    return np.rot90(out, -info["rotation"]).copy(), stats


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible strokes and boundary-attached stubs only."""
    return _simulate(_parse(input_grid))[0]


_ALTERNATIVES = ("all_slices", "physical", "centers", "neighbors", "stop", "bounding")


def _differences(info, output):
    return {name for name in _ALTERNATIVES
            if not np.array_equal(_simulate(info, name)[0], output)}


def _key(a):
    return a.shape, a.tobytes()


def _draw_stroke(a, q, lo, hi):
    lo += (q - lo) % 2
    hi -= (hi - q) % 2
    if hi - lo < 2 or lo < abs(q) or hi > _last_t(a.shape, q):
        return False
    cells = [_cell(t, q) for t in range(lo, hi + 1, 2)]
    if any(a[p] == 1 for p in cells):
        return False
    for p in cells:
        a[p] = 2
    return True


def _sample_scene(rng, mode, shape=None):
    """Compose stroke schedules and entries; no parent generator is invoked."""
    if shape is None:
        lower = 18 if mode == "test" else 12
        shape = (int(rng.integers(lower, 31)), int(rng.integers(lower, 31)))
    a = np.zeros(shape, dtype=np.int8)
    center = int(rng.integers(-3, 4))
    source_qs = [center]
    count = 1 if mode == "intro" else int(rng.choice([1, 2, 3], p=[.55, .30, .15]))
    for q in rng.permutation(np.arange(-min(9, shape[0] - 5), min(9, shape[1] - 5) + 1)):
        q = int(q)
        if len(source_qs) >= count:
            break
        if all(abs(q - x) >= 2 for x in source_qs):
            source_qs.append(q)
    source_end = 0
    for q in source_qs:
        n = int(rng.integers(2, 5))
        source_end = max(source_end, abs(q) + 2 * (n - 1))
        for t in range(abs(q), abs(q) + 2 * n, 2):
            a[_cell(t, q)] = 1

    left = center - int(rng.integers(3, 7))
    right = center + int(rng.integers(3, 7))
    # Mix endpoint phases; neither literal parity nor hit offset is fixed.
    base = max(source_end + 3, abs(left) + 4, abs(right) + 4) + int(rng.integers(0, 3))
    stages = 1 if mode == "intro" else int(rng.integers(2, 5)) if mode == "test" else int(rng.integers(1, 4))
    for stage in range(stages):
        if stage:
            side = int(rng.integers(3))
            if side != 0:
                left -= int(rng.integers(2, 6))
            if side != 1:
                right += int(rng.integers(2, 6))
            base += int(rng.integers(3, 9))
            base = max(base, abs(left) + 2, abs(right) + 2)
        span = int(rng.integers(4, 13))
        for q in (left, right):
            lo = max(abs(q) + 2, base - int(rng.integers(0, 4)))
            hi = min(_last_t(shape, q), base + span + int(rng.integers(0, 4)))
            _draw_stroke(a, q, lo, hi)
        # An interior stroke may start late, finish early, or span the bracket.
        if rng.random() < .75 or mode == "witness":
            q = int(rng.integers(left + 2, right - 1))
            lo = base + int(rng.integers(3, 8)) if mode == "witness" or rng.random() < .6 else base - 2
            hi = min(_last_t(shape, q), lo + 2 * int(rng.integers(2, 5)))
            _draw_stroke(a, q, max(abs(q) + 2, lo), hi)
    # Freely placed strokes introduce inactive singleton intervals, distractor
    # brackets, asymmetric endpoint changes and temporal groups.
    for _ in range(int(rng.integers(0, 3 if mode == "intro" else 6))):
        q = int(rng.integers(-shape[0] + 3, shape[1] - 2))
        low, high = abs(q) + 2, _last_t(shape, q)
        if high - low < 4:
            continue
        lo = int(rng.integers(low, high - 2))
        hi = min(high, lo + 2 * int(rng.integers(2, 8)))
        _draw_stroke(a, q, lo, hi)
    return a


def _make_pair(rng, mode, palette, rotation, seen, required=frozenset(), min_hits=0, shape=None):
    for _ in range(360):
        canonical = _sample_scene(rng, mode, shape)
        a = np.rot90(np.asarray(palette, dtype=np.int8)[canonical], -rotation).copy()
        if _key(a) in seen:
            continue
        try:
            info = _parse(a)
        except AssertionError:
            continue
        o, stats = _simulate(info)
        if stats["collisions"] < min_hits or not stats["added_marker"] or not stats["added_guide"]:
            continue
        if required and not required <= _differences(info, o):
            continue
        return dict(input=a, output=o)
    # Deterministic recovery under the same rule. Even the smallest sampler
    # canvas (12x12) has 24 translated/prefix-length variants, versus at most
    # 13 preceding pairs. Visible endpoints are always used by the solver.
    canvas = shape or ((24, 24) if mode == "test" else (16, 16))
    pixels = ((4, 1), (5, 2), (1, 5), (2, 6), (4, 5), (5, 6),
              (7, 2), (8, 3), (2, 8), (3, 9))
    for dr in range(canvas[0] - 8):
        for dc in range(canvas[1] - 9):
            for length in (2, 3):
                canonical = np.zeros(canvas, dtype=np.int8)
                q = dc - dr
                for t in range(abs(q), abs(q) + 2 * length, 2):
                    canonical[_cell(t, q)] = 1
                for r, c in pixels:
                    canonical[r + dr, c + dc] = 2
                a = np.rot90(np.asarray(palette, dtype=np.int8)[canonical], -rotation).copy()
                if _key(a) in seen:
                    continue
                info = _parse(a)
                o, stats = _simulate(info)
                if stats["collisions"] >= min_hits and required <= _differences(info, o):
                    return dict(input=a, output=o)
    raise AssertionError("no unique witness remains within legal puzzle size")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name in ((seed, "seed"), (num_train, "num_train"), (num_test, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    rotation = int(rng.integers(4))
    mixed_palette = bool(rng.random() < .5)
    mixed_rotation = bool(rng.random() < .5)
    scale_shift = bool(rng.random() < .35)
    puzzle = {"train": [], "test": []}
    seen = set()
    # Two witnesses collectively teach all listed simpler alternatives even
    # when only two demonstrations are requested.
    for i in range(num_train):
        roles = tuple(map(int, rng.choice(10, 3, replace=False))) if mixed_palette and i else palette
        orient = ((rotation + i % 2) % 4 if not scale_shift else
                  (rotation + i) % 4 if mixed_rotation else rotation)
        required = frozenset(("physical", "centers", "stop")) if i == 0 else frozenset(("all_slices", "neighbors", "bounding")) if i == 1 else frozenset()
        shape = (int(rng.integers(16, 23)), int(rng.integers(16, 23))) if scale_shift else None
        if i == 0 and shape is None:
            shape = (int(rng.integers(12, 23)), int(rng.integers(12, 23)))
        pair = _make_pair(rng, "intro" if i == 0 else "witness" if i == 1 else "train", roles, orient, seen,
                          required, min_hits=2 if i == 1 else 1, shape=shape)
        puzzle["train"].append(pair)
        seen.add(_key(pair["input"]))
    for i in range(num_test):
        roles = tuple(map(int, rng.choice(10, 3, replace=False))) if mixed_palette else palette
        orient = ((rotation + 2 + i % 2) % 4 if not scale_shift else
                  int(rng.integers(4)) if mixed_rotation else rotation)
        shape = (int(rng.integers(24, 31)), int(rng.integers(24, 31))) if scale_shift else None
        pair = _make_pair(rng, "test", roles, orient, seen, min_hits=2, shape=shape)
        puzzle["test"].append(pair)
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned grids and actual teaching witnesses, not sampling labels."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, taught = set(), set()
    train_stats = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, o = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == o.shape and _key(a) not in seen, "shape mismatch or duplicate")
            seen.add(_key(a))
            info = _parse(a)
            expected, stats = _simulate(info)
            _require(np.array_equal(expected, o), "incorrect shutter routing")
            bg = info["background"]
            _require(bool(np.all(o[a != bg] == a[a != bg])), "original pixels must be preserved")
            _require(set(map(int, np.unique(o))) <= set(map(int, np.unique(a))), "new literal color")
            _require(stats["added_guide"] > 0 and stats["added_marker"] > 0, "missing interaction")
            differences = _differences(info, o)
            if split == "train":
                taught.update(differences)
                train_stats.append(stats)
            else:
                _require(differences <= taught, "test uses an untaught alternative-rule distinction")
    _require(set(_ALTERNATIVES) <= taught, "training does not expose both concepts")
    _require(any(s["two_flanks"] for s in train_stats), "training lacks a visible split")
    _require(any(s["collisions"] >= 2 for s in train_stats), "training lacks repeated activation")


def self_test():
    """Bounded 42-seed audit plus hand-computed lattice and corruption checks."""
    global _sample_scene
    shapes, structures, palettes, directions, stroke_counts, hit_counts = set(), set(), set(), set(), set(), set()
    fingerprints = set()
    observed = dict(clipped=0, multi_hit=0, merges=0, scale_shifts=0, direction_shifts=0,
                    varying_strokes=0, varying_palettes=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        encoded = {s: [{k: p[k].tolist() for k in p} for p in puzzle[s]] for s in puzzle}
        fingerprints.add(json.dumps(encoded, separators=(",", ":")))
        if seed in (0, 17, 41):
            repeat = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeat[s]) for k in p), "seed not reproducible")
        parsed = {s: [_parse(p["input"]) for p in puzzle[s]] for s in puzzle}
        train_directions = {p["rotation"] for p in parsed["train"]}
        test_directions = {p["rotation"] for p in parsed["test"]}
        direction_shift = train_directions.isdisjoint(test_directions)
        scale_shift = all(min(p["grid"].shape) > max(q["grid"].shape)
                          for p in parsed["test"] for q in parsed["train"])
        _require(direction_shift or scale_shift, "missing meaningful train/test shift")
        observed["direction_shifts"] += direction_shift
        observed["scale_shifts"] += scale_shift
        observed["varying_strokes"] += len({len(p["strokes"]) for s in parsed for p in parsed[s]}) > 1
        observed["varying_palettes"] += len({(p["background"], p["guide"], p["marker"]) for s in parsed for p in parsed[s]}) > 1
        for split in puzzle:
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                a = info["grid"]
                normalized = np.zeros(a.shape, dtype=np.int8)
                normalized[a == info["guide"]] = 1
                normalized[a == info["marker"]] = 2
                shapes.add(a.shape)
                structures.add(_key(normalized))
                palettes.add((info["background"], info["guide"], info["marker"]))
                directions.add(info["rotation"])
                stroke_counts.add(len(info["strokes"]))
                stats = _simulate(info)[1]
                hit_counts.add(stats["collisions"])
                for feature in ("clipped", "multi_hit", "merges"):
                    observed[feature] += bool(stats[feature])
    _require(len(fingerprints) == 42 and len(structures) >= 150, "insufficient structural variation")
    _require(len(shapes) >= 40 and len(palettes) >= 25 and len(stroke_counts) >= 5 and len(hit_counts) >= 4, "variation axes collapsed")
    _require(directions == set(range(4)), "missing ray direction")
    _require(all(observed[x] > 0 for x in observed), "missing structural interactions")
    # Inclusive continuous endpoints and outward rounding have no hidden phase.
    strokes = [(-3, 7, 15), (4, 8, 16), (0, 12, 14)]
    _require(_intervals(strokes, 8) == [(-3, 4)], "continuous interval failure")
    _require(_intervals(strokes, 8, "centers") == [], "center witness failure")
    _require(_intervals(strokes, 8, "neighbors") == [], "inactive middle witness failure")
    _require(_flanks(8, -3, 4) == (-4, 6), "wrong lattice flanks")
    _require(_flanks(9, -3, 4) == (-5, 5), "wrong odd lattice flanks")
    # A complete hand-calculated fixture: one latent hit at t=6, with an
    # inactive middle stroke and an outer boundary between pixel centers.
    a = np.zeros((9, 11), dtype=np.int8)
    a[0, 0] = a[1, 1] = 1
    for r, c in ((4, 1), (5, 2), (6, 3), (7, 4),
                 (1, 5), (2, 6), (3, 7), (4, 8), (4, 5), (5, 6)):
        a[r, c] = 2
    expected = a.copy()
    for r, c in ((4, 2), (3, 3), (2, 4)):
        expected[r, c] = 2
    for r, c in ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
                 (5, 1), (6, 2), (7, 3), (8, 4),
                 (0, 6), (1, 7), (2, 8), (3, 9), (4, 10)):
        expected[r, c] = 1
    for rotation in range(4):
        _require(np.array_equal(solve(np.rot90(a, rotation)), np.rot90(expected, rotation)),
                 "hand-calculated fixture or rotation failed")
    _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(expected)), "reflection failed")
    # Two collisions two depth units apart: the second upstream connector
    # crosses all three previously revealed cells, which must stay solid.
    overlap = a.copy()
    for r, c in ((6, 1), (7, 2), (1, 7), (2, 8)):
        overlap[r, c] = 2
    overlap_out = solve(overlap)
    _require(all(overlap_out[r, c] == 2 for r, c in ((4, 2), (3, 3), (2, 4))),
             "a later crosspiece erased a revealed slice")
    # The rule also supports members outside the sampler's changed-pair
    # regime: disjoint intervals produce no shutter, even on a physical hit.
    transparent = np.zeros((9, 11), dtype=np.int8)
    transparent[0, 0] = transparent[1, 1] = 1
    for r, c in ((3, 3), (4, 4), (4, 8), (5, 9)):
        transparent[r, c] = 2
    transparent_out = transparent.copy()
    for r in (2, 5, 6, 7, 8):
        transparent_out[r, r] = 1
    _require(np.array_equal(solve(transparent), transparent_out), "lone stroke became a wall")
    # A terminal endpoint clips one flank; its upstream connector crosses a
    # preserved input marker at (6,3).  These facts are read off the coordinates.
    clipped = np.zeros((8, 11), dtype=np.int8)
    clipped[0, 1] = clipped[1, 2] = 1
    for r, c in ((4, 1), (5, 2), (6, 3), (7, 4), (3, 7), (4, 8)):
        clipped[r, c] = 2
    expected = clipped.copy()
    for r, c in ((6, 5), (5, 6), (4, 7)):
        expected[r, c] = 2
    for r, c in ((2, 3), (3, 4), (4, 5),
                 (7, 2), (5, 4), (3, 6), (2, 7),
                 (3, 8), (4, 9), (5, 10)):
        expected[r, c] = 1
    _require(np.array_equal(solve(clipped), expected), "hand-calculated clipping failed")
    # Exercise deterministic recovery directly: fourteen unique members even
    # on the smallest sampler canvas, each distinguishing every shortcut.
    original_sampler = _sample_scene
    try:
        _sample_scene = lambda rng, mode, shape=None: np.zeros(shape or (12, 12), dtype=np.int8)
        seen = set()
        recovery_rng = np.random.default_rng(1907)
        for _ in range(14):
            pair = _make_pair(recovery_rng, "witness", (0, 1, 2), 0, seen,
                              frozenset(_ALTERNATIVES), min_hits=2, shape=(12, 12))
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "recovery rule differs")
            seen.add(_key(pair["input"]))
    finally:
        _sample_scene = original_sampler
    # Correctly labeled demonstrations without a middle stroke fail to teach
    # global bracketing across an inactive axis. The validator must notice.
    no_middle = a.copy()
    no_middle[4, 5] = no_middle[5, 6] = 0
    weak = {"train": [{"input": no_middle, "output": solve(no_middle)},
                      {"input": np.rot90(no_middle).copy(), "output": solve(np.rot90(no_middle))}],
            "test": [{"input": a.copy(), "output": solve(a)}]}
    try:
        validate_generated_puzzle(weak)
    except AssertionError as exc:
        _require("untaught" in str(exc), "insufficient-teaching check failed for an unrelated reason")
    else:
        raise AssertionError("insufficient teaching accepted")
    for counts in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(83, *counts)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    changed = np.argwhere(bad["test"][0]["input"] != bad["test"][0]["output"])[0]
    p = tuple(changed)
    bad["test"][0]["output"][p] = bad["test"][0]["input"][p]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return dict(seeds=42, structures=len(structures), shapes=len(shapes),
                palettes=len(palettes), directions=sorted(directions),
                stroke_counts=sorted(stroke_counts), collision_counts=sorted(hit_counts),
                interactions=observed)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
