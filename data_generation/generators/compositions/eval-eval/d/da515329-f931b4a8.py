"""Pinwheel loom: route a two-scale, transparent-overlaid tape along four arms.

Only NumPy and the standard library are used. Nothing is saved by this module.
The three-row header is part of the input; solve needs no puzzle context.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_EVIDENCE = frozenset((
    "straight_rays", "opposite_hand", "native_carrier", "ignore_stencil",
    "flat_carrier", "scaled_stencil", "corner_reset", "image_phase",
    "clip_phase",
))


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= side <= 30 for side in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    assert a.shape[0] >= 8 and a.shape[1] >= 5
    bg, marker = int(a[0, -1]), int(a[2, 0])
    assert bg != marker and np.all(a[2] == marker)
    occupied = np.flatnonzero(a[0] != bg)
    k = len(occupied)
    assert 2 <= k <= 7 and k < a.shape[1]
    assert np.array_equal(occupied, np.arange(k))
    carrier, stencil = a[0, :k].copy(), a[1, :k].copy()
    assert not np.any(carrier == marker) and not np.any(stencil == marker)
    assert len(np.unique(carrier)) >= 2
    assert np.any(stencil == bg), "stencil must retain transparency"
    assert np.all(a[1, k:] == bg)
    board = a[3:]
    assert np.all((board == bg) | (board == marker))
    mask = board == marker
    count = int(mask.sum())
    assert count % 4 == 0
    length = count // 4
    assert 1 <= length <= 5
    ys = np.flatnonzero(mask.sum(axis=1) == 2 * length)
    xs = np.flatnonzero(mask.sum(axis=0) == 2 * length)
    assert len(ys) == len(xs) == 1
    cy, cx = int(ys[0]), int(xs[0])
    h, w = board.shape
    assert length <= cy < h - length and length <= cx < w - length
    expected = np.zeros(board.shape, dtype=bool)
    for d in range(1, length + 1):
        expected[cy-d, cx] = expected[cy+d, cx] = True
        expected[cy, cx-d] = expected[cy, cx+d] = True
    assert np.array_equal(mask, expected), "not one equal-spoke hollow cross"
    return bg, marker, carrier, stencil, length, board.shape, (cy, cx)


def _walk_steps(shape, center, length):
    """Infinite-plane segment walk, recording intrinsic distance only in bounds.

    Extra fields are diagnostic counterfactuals: distance since an actual bend,
    and distance counting only visible pixels. Vertices are visited once.
    A full four-stage cycle advances eight cells outward. The bound retains
    later re-entry, even after an arm first exits the canvas.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    steps = np.full(shape, -1, dtype=np.int32)
    bends = np.full_like(steps, -1)
    visible = np.full_like(steps, -1)
    distance = max(cy, cx, h-1-cy, w-1-cx)
    stages = 4 * (distance // 8 + 3)
    for radial in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        y, x, total, seen, leg = cy, cx, 0, 0, 0
        previous = None

        def travel(dy, dx, count):
            nonlocal y, x, total, seen, leg, previous
            if count == 0:
                return
            if previous != (dy, dx):
                leg = 0
            distances = np.arange(1, count+1)
            yy, xx = y + dy*distances, x + dx*distances
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            rows, cols, ds = yy[ok], xx[ok], distances[ok]
            assert np.all(steps[rows, cols] == -1), "winding paths intersect"
            steps[rows, cols] = total + ds - 1
            bends[rows, cols] = leg + ds - 1
            visible[rows, cols] = seen + np.arange(len(ds))
            seen += len(ds)
            total += count
            leg += count
            y, x = y + dy*count, x + dx*count
            previous = (dy, dx)

        dy, dx = radial
        travel(dy, dx, radius)
        dy, dx = dx, -dy
        for stage in range(stages):
            travel(dy, dx, 4*stage+2)
            travel(dx, -dy, radius-2)
            travel(dy, dx, radius)
            dy, dx = dx, -dy
    return steps, bends, visible


def _analytic_steps(shape, center, length):
    """Closed-form interval check; no accumulated path endpoints or walking."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    result = np.full(shape, -1, dtype=np.int32)
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))

    def assign(mask, times):
        assert np.all(result[mask] == -1), "analytic paths intersect"
        result[mask] = np.broadcast_to(times, shape)[mask]

    for _ in range(4):
        assign((xx == 0) & (yy >= -radius) & (yy <= -1), -yy-1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                n = 4*q+s
                first = radius + 2*n*(n+radius-1)
                run = 4*n+2
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                assign((across == 0) & (along >= 1) & (along <= run),
                       first+along-1)
                assign((along == run) & (across >= 1) & (across <= radius-2),
                       first+run+across-1)
                assign((across == radius-2) & (along >= run+1) & (along <= run+radius),
                       first+along+radius-3)
        yy, xx = xx, -yy
    return result


def _dye(steps, carrier, stencil, length, bg):
    result = np.full(steps.shape, bg, dtype=np.int8)
    on = steps >= 0
    t = steps[on]
    fine = stencil[t % len(stencil)]
    coarse = carrier[(t // length) % len(carrier)]
    result[on] = np.where(fine == bg, coarse, fine)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    bg, _, carrier, stencil, length, shape, center = _decode(input_grid)
    steps = _walk_steps(shape, center, length)[0]
    return _dye(steps, carrier, stencil, length, bg)


def _reference(input_grid):
    bg, _, carrier, stencil, length, shape, center = _decode(input_grid)
    steps = _analytic_steps(shape, center, length)
    # Independently construct one full tape period by replication and overlay.
    tape = np.repeat(carrier, length)
    fine = np.tile(stencil, length)
    tape[fine != bg] = fine[fine != bg]
    result = np.full(shape, bg, dtype=np.int8)
    on = steps >= 0
    result[on] = tape[steps[on] % len(tape)]
    return result


def _evidence(input_grid, output_grid):
    bg, _, carrier, stencil, length, shape, center = _decode(input_grid)
    steps, bends, visible = _walk_steps(shape, center, length)
    k = len(carrier)
    on = steps >= 0
    t = steps[on]
    coarse, fine = carrier[(t//length) % k], stencil[t % k]
    alternatives = {}

    def on_path(values):
        a = np.full(shape, bg, dtype=np.int8)
        a[on] = values
        return a

    alternatives["native_carrier"] = on_path(np.where(fine == bg, carrier[t % k], fine))
    alternatives["ignore_stencil"] = on_path(coarse)
    alternatives["flat_carrier"] = on_path(np.where(fine == bg, carrier[0], fine))
    stretched = stencil[(t//length) % k]
    alternatives["scaled_stencil"] = on_path(np.where(stretched == bg, coarse, stretched))
    alternatives["corner_reset"] = _dye(bends, carrier, stencil, length, bg)
    alternatives["clip_phase"] = _dye(visible, carrier, stencil, length, bg)
    yy, xx = np.indices(shape)
    radial = abs(yy-center[0]) + abs(xx-center[1]) - 1
    alternatives["image_phase"] = _dye(np.where(on, radial, -1), carrier, stencil, length, bg)
    rays = (yy == center[0]) | (xx == center[1])
    alternatives["straight_rays"] = _dye(np.where(rays, radial, -1), carrier, stencil, length, bg)
    opposite = np.fliplr(_walk_steps(shape, (center[0], shape[1]-1-center[1]), length)[0])
    alternatives["opposite_hand"] = _dye(opposite, carrier, stencil, length, bg)
    return {name for name, answer in alternatives.items()
            if not np.array_equal(answer, output_grid)}


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _normalized(values):
    roles = {}
    return tuple(roles.setdefault(int(c), len(roles)) for c in np.asarray(values).flat)


def _assemble(shape, center, length, carrier, stencil, bg, marker):
    h, w = shape
    a = np.full((h+3, w), bg, dtype=np.int8)
    a[0, :len(carrier)], a[1, :len(stencil)], a[2] = carrier, stencil, marker
    cy, cx = center
    for d in range(1, length+1):
        a[3+cy-d, cx] = a[3+cy+d, cx] = marker
        a[3+cy, cx-d] = a[3+cy, cx+d] = marker
    return {"input": a, "output": solve(a)}


def _sample_pair(rng, length, k_range, teaching, boundary=False):
    palette = rng.permutation(10)
    bg, marker = int(palette[0]), int(palette[1])
    k = int(rng.integers(k_range[0], k_range[1]+1))
    # Arbitrary short strips, including repeated colors and unequal runs.
    inks = palette[2:2+int(rng.integers(2, min(4, k)+1))]
    if rng.random() < .4:
        cuts = sorted(rng.choice(np.arange(1, k), len(inks)-1, replace=False))
        carrier = inks[np.searchsorted(cuts, np.arange(k), side="right")]
    else:
        labels = np.r_[np.arange(len(inks)), rng.integers(len(inks), size=k-len(inks))]
        carrier = inks[rng.permutation(labels)]
    stencil = np.full(k, bg, dtype=np.int8)
    count = int(rng.integers(1, k)) if teaching or rng.random() >= .12 else 0
    positions = rng.choice(k, count, replace=False)
    if rng.random() < .5:
        positions = (np.arange(count)+int(rng.integers(k))) % k
    stencil_inks = rng.choice(palette[2:], size=int(rng.integers(1, min(3, max(1, count))+1)), replace=False)
    stencil[positions] = rng.choice(stencil_inks, count)
    # All teaching views expose several turns, not just their seed.
    low = max(2*max(2, length)+7, k+1, 13) if teaching else max(2*length+1, k+1, 7)
    h = int(rng.integers(low, 28))
    w = int(rng.integers(low, 31))
    if rng.random() < .25:
        h = min(w, 27)
    margin = 2 if teaching else 0
    cy = int(rng.integers(length+margin, h-length-margin))
    cx = int(rng.integers(length+margin, w-length-margin))
    if rng.random() < .2 and not boundary:
        cy, cx = (h-1)//2, (w-1)//2
    if boundary:
        axis = int(rng.integers(2))
        if axis == 0:
            cy = int(rng.choice([length, h-1-length]))
        else:
            cx = int(rng.choice([length, w-1-length]))
    return _assemble((h, w), (cy, cx), length, carrier, stencil, bg, marker)


def _fallback_teaching(rng, lengths):
    # Finite constructive witnesses prevent stochastic exhaustion. This rare
    # fallback still uses the same visible rule and is shuffled with all pairs.
    colors = [int(c) for c in rng.permutation(10)]
    bg, marker, a, b, c, d = colors[:6]
    pairs = [_assemble((23, 25), (8+i, 10+2*i), length,
                       np.array([a, b, c]), np.array([bg, d, bg]), bg, marker)
             for i, length in enumerate(lengths)]
    covered = set().union(*(_evidence(p["input"], p["output"]) for p in pairs))
    assert _EVIDENCE <= covered, "constructive teaching witnesses failed"
    return pairs


def _teaching_pairs(rng, lengths, k_range):
    # Select on observable evidence, keeping only two candidates at a time.
    # Each attempt uses fresh geometry, palette, strips and opacity. No saved
    # sample or source answer participates in construction or inference.
    for _ in range(64):
        pairs = [_sample_pair(rng, length, k_range, True) for length in lengths]
        covered = set().union(*(_evidence(p["input"], p["output"]) for p in pairs))
        if _EVIDENCE <= covered and len({_key(p["input"]) for p in pairs}) == 2:
            return pairs
    return _fallback_teaching(rng, lengths)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain = _integer(num_train, 2, 10, "num_train")
    ntest = _integer(num_test, 1, 4, "num_test")
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))  # sampling choices, never hidden solve state
    lengths = (2, 3) if rng.random() < .55 else (3, 4)
    if regime != 0 and rng.random() < .25:
        lengths = (2, 4)
    k_range = (2, 4) if regime == 1 else (2, 7)
    train = _teaching_pairs(rng, lengths, k_range)
    seen = {_key(p["input"]) for p in train}
    puzzle = {"train": train, "test": []}
    for split, count in (("train", ntrain), ("test", ntest)):
        while len(puzzle[split]) < count:
            is_train = split == "train"
            train_lengths = sorted({_decode(p["input"])[4] for p in puzzle["train"]})
            length = (max(lengths)+1 if regime == 0 and not is_train else
                      int(rng.choice(train_lengths if not is_train else [1, *lengths])))
            kr = (5, 7) if regime == 1 and not is_train else k_range
            # A maximum of 14 retained pairs. Recoloring a counterfactual new
            # candidate is unnecessary; geometric/strip space is enormous.
            for _ in range(64):
                pair = _sample_pair(rng, length, kr, is_train,
                                    boundary=(regime == 2 and not is_train))
                if _key(pair["input"]) not in seen:
                    break
            else:
                # Guaranteed distinct by enumerating 90 ordered marker roles
                # on a legal fixed view, far more than 14 retained inputs.
                bg, marker, carrier, stencil, _, shape, center = _decode(pair["input"])
                for new_bg in range(10):
                    for new_marker in range(10):
                        if new_bg == new_marker:
                            continue
                        rest = [c for c in range(10) if c not in (new_bg, new_marker)]
                        old = [c for c in range(10) if c not in (bg, marker)]
                        mapping = dict(zip(old, rest)) | {bg: new_bg, marker: new_marker}
                        trial = np.array([mapping[int(c)] for c in pair["input"].flat], dtype=np.int8).reshape(pair["input"].shape)
                        if _key(trial) not in seen:
                            pair = {"input": trial, "output": solve(trial)}
                            break
                    else:
                        continue
                    break
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), set()
    info = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            bg, marker, carrier, stencil, length, shape, center = _decode(a)
            assert b.shape == shape and np.array_equal(b, _reference(a)), "incorrect woven winding"
            assert not np.any(b == marker), "construction marker must be erased"
            assert set(np.unique(b)) <= {bg, *carrier, *stencil}
            analytic = _analytic_steps(shape, center, length)
            assert np.array_equal(b != bg, analytic >= 0), "thread support/corridors changed"
            assert b[center] == bg
            assert np.all(b[a[3:] == marker] != bg), "seed lost its routed thread"
            assert int(np.sum(b != bg)) > 4*length, "no continuation exposed"
            info[split].append((length, len(carrier), shape, center))
            if split == "train":
                evidence |= _evidence(a, b)
    assert _EVIDENCE <= evidence, "training misses alternatives: " + str(_EVIDENCE-evidence)
    train_lengths = {x[0] for x in info["train"]}
    assert len(train_lengths & {2, 3, 4}) >= 2
    for length in train_lengths:
        assert any(l == length and min(y-l, x-l, h-1-y-l, w-1-x-l) >= 2
                   for l, _, (h, w), (y, x) in info["train"])
    # The sampler always supplies at least one observable generalization axis.
    new_length = all(x[0] == max(train_lengths)+1 for x in info["test"])
    new_period = min(x[1] for x in info["test"]) > max(x[1] for x in info["train"])
    edge_shift = all(min(y-l, x-l, h-1-y-l, w-1-x-l) == 0
                     for l, _, (h, w), (y, x) in info["test"])
    assert new_length or new_period or edge_shift, "no meaningful test shift"


def self_test():
    """Bounded streaming tests: 42 seeds, geometry properties and mutations."""
    metrics = Counter()
    sets = {name: set() for name in ("shapes", "centers", "lengths", "periods",
                                    "carrier_structures", "stencil_masks", "palettes", "answers")}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                digest.update(a.tobytes())
                digest.update(b.tobytes())
                assert np.array_equal(solve(a.copy()), b)
                bg, marker, carrier, stencil, length, shape, center = _decode(a)
                sets["shapes"].add(shape)
                sets["centers"].add((length, shape, center))
                sets["lengths"].add(length)
                sets["periods"].add(len(carrier))
                sets["carrier_structures"].add(_normalized(carrier))
                sets["stencil_masks"].add(tuple(stencil != bg))
                sets["palettes"].add((bg, marker))
                sets["answers"].add((shape, _normalized(b)))
                metrics[split+"_pairs"] += 1
                metrics["empty_stencils"] += int(np.all(stencil == bg))
                metrics["overlapping_inks"] += int(bool(set(carrier) & set(stencil[stencil != bg])))
                metrics["tall"] += int(shape[0] > shape[1])
                metrics["wide"] += int(shape[1] > shape[0])
                steps, _, visible = _walk_steps(shape, center, length)
                metrics["reentry_phase_witnesses"] += int(np.any((steps >= 0) & (steps != visible)))
        signatures.add(digest.digest())
        if seed == 0:
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("corrupted label accepted")
    assert len(signatures) == 42
    assert sets["lengths"] == set(range(1, 6)) and sets["periods"] == set(range(2, 8))
    assert len(sets["shapes"]) >= 60 and len(sets["centers"]) >= 150
    assert len(sets["carrier_structures"]) >= 20 and len(sets["stencil_masks"]) >= 25
    assert len(sets["palettes"]) >= 50 and len(sets["answers"]) >= 180
    assert all(metrics[k] > 0 for k in ("empty_stencils", "overlapping_inks", "tall", "wide", "reentry_phase_witnesses"))
    for length in range(1, 6):
        for shape, center in (((27, 30), (length, length)),
                              ((21, 25), (20-length, 24-length)),
                              ((23, 23), (11, 11))):
            small = _walk_steps(shape, center, length)[0]
            assert np.array_equal(small, _analytic_steps(shape, center, length))
            support = small >= 0
            assert not np.any(support[:-1, :-1] & support[1:, :-1] & support[:-1, 1:] & support[1:, 1:])
            rotated = (shape[1]-1-center[1], center[0])
            assert np.array_equal(np.rot90(small), _walk_steps(shape[::-1], rotated, length)[0])
            big = _analytic_steps((81, 81), (40, 40), length)
            top, left = 40-center[0], 40-center[1]
            assert np.array_equal(small, big[top:top+shape[0], left:left+shape[1]])
    # Direct checks of the closed-form start/distance convention and dye rule.
    steps = _walk_steps((23, 23), (11, 11), 3)[0]
    assert steps[10, 11] == 0 and steps[8, 11] == 2
    assert steps[8, 13] == 4 and steps[9, 13] == 5 and steps[9, 16] == 8
    c, s = np.array([1, 2, 3]), np.array([0, 4, 0])
    assert _dye(np.arange(9)[None], c, s, 3, 0).tolist() == [[1, 4, 1, 2, 4, 2, 3, 4, 3]]
    # Short spokes share A's geometry but retain different B stretch factors.
    short = _walk_steps((23, 23), (11, 11), 1)[0]
    assert np.array_equal(short, _walk_steps((23, 23), (11, 11), 2)[0])
    assert not np.array_equal(_dye(short, c, s, 1, 0), _dye(short, c, s, 2, 0))
    rng = np.random.default_rng(5678)
    for lengths in ((2, 3), (3, 4), (2, 4)):
        _fallback_teaching(rng, lengths)
    # Tight, unsampled family members and arbitrary palette reassignment.
    for length in range(1, 6):
        side = max(5, 2*length+1)
        pair = _assemble((side, side), (side//2, side//2), length, c, s, 0, 9)
        assert np.array_equal(pair["output"], _reference(pair["input"]))
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    for counts in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(723, *counts), generate_puzzle(723, *counts)
        assert all(_key(p[k]) == _key(q[k]) for split in a
                   for p, q in zip(a[split], b[split]) for k in p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": dict(metrics), "diversity": {k: len(v) for k, v in sets.items()},
            "verification": "generator-owned checks; no independent semantic approval"}
