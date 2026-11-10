"""Periodic threads transported along four growing rectilinear windings.

No parent module, saved example, seed, or generation state enters solve().
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PERIOD_CAP = 5


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _primitive(word):
    word = tuple(int(x) for x in word)
    for p in range(1, len(word) + 1):
        if len(word) % p == 0 and all(word[i] == word[i % p]
                                    for i in range(len(word))):
            return word[:p]
    raise AssertionError("unreachable")


def _routes(radius, shape, center, hand):
    """Visible (infinite-plane step index, row, column) for each arm.

    Index zero is the first cell after a spoke tip. Every unit step advances
    the clock, including steps outside the rectangle. Segment corners are
    visited once. A four-stage cycle moves the bounding envelope out by 8.
    """
    cy, cx = center
    h, w = shape
    distance = max(cy, cx, h - 1 - cy, w - 1 - cx)
    stages = 4 * (distance // 8 + 3)
    paths = []
    for sy, sx in DIRECTIONS:
        y, x = cy + radius * sy, cx + radius * sx
        dy, dx = hand * sx, -hand * sy
        t = 0
        path = []
        for k in range(stages):
            for vy, vx, length in ((dy, dx, 4 * k + 2),
                                  (hand * dx, -hand * dy, radius - 2),
                                  (dy, dx, radius)):
                for _ in range(length):
                    y, x = y + vy, x + vx
                    if 0 <= y < h and 0 <= x < w:
                        path.append((t, y, x))
                    t += 1
            dy, dx = hand * dx, -hand * dy
        paths.append(path)
    return paths


def _seed_mask(shape, center, radius):
    mask = np.zeros(shape, dtype=bool)
    for dy, dx in DIRECTIONS:
        for d in range(1, radius + 1):
            mask[center[0] + d * dy, center[1] + d * dx] = True
    return mask


def _origin(a):
    values, counts = np.unique(a, return_counts=True)
    singletons = values[counts == 1]
    if len(singletons) != 1:
        raise ValueError("the origin must be the only singleton color")
    marker = int(singletons[0])
    cy, cx = map(int, np.argwhere(a == marker)[0])
    h, w = a.shape
    if not (1 <= cy < h - 1 and 1 <= cx < w - 1):
        raise ValueError("origin has no surrounding cross")
    background = int(a[cy - 1, cx - 1])
    if any(a[cy + dy, cx + dx] != background
           for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))):
        raise ValueError("origin diagonals must be background")
    base = int(a[cy - 1, cx])
    if len({marker, background, base}) != 3:
        raise ValueError("origin, corridor and spoke colors must differ")
    lengths = []
    for dy, dx in DIRECTIONS:
        d = 1
        while (0 <= cy + d * dy < h and 0 <= cx + d * dx < w
               and a[cy + d * dy, cx + d * dx] == base):
            d += 1
        lengths.append(d - 1)
    if len(set(lengths)) != 1 or not 2 <= lengths[0] <= 5:
        raise ValueError("expected four equal intact spokes of length 2..5")
    radius = lengths[0]
    possible = []
    for hand in (1, -1):
        first = [(cy + radius * dy + hand * dx,
                  cx + radius * dx - hand * dy) for dy, dx in DIRECTIONS]
        if all(a[y, x] not in (background, marker) for y, x in first):
            possible.append(hand)
    if len(possible) != 1:
        raise ValueError("visible first turns must encode one handedness")
    return background, base, marker, (cy, cx), radius, possible[0]


def _recover(observations, base, accent):
    """All sparse-error periodic explanations must give one infinite word."""
    n = len(observations)
    limit = max(1, n // 12)
    candidates = {}
    for p in range(2, PERIOD_CAP + 1):
        word, errors, supported = [], 0, True
        for phase in range(p):
            vals = [v for t, v in observations if t % p == phase]
            if len(vals) < 3:
                supported = False
                break
            b = vals.count(base)
            c = vals.count(accent)
            if b == c:
                supported = False
                break
            word.append(base if b > c else accent)
            errors += min(b, c)
        if supported and errors <= limit and len(set(word)) == 2:
            candidates[_primitive(word)] = errors
    if len(candidates) != 1:
        raise ValueError("thread has no unique periodic continuation")
    word, errors = next(iter(candidates.items()))
    return word, errors


def _decode(a):
    bg, base, marker, center, radius, hand = _origin(a)
    paths = _routes(radius, a.shape, center, hand)
    permitted = _seed_mask(a.shape, center, radius)
    permitted[center] = True
    threads = []
    accents = []
    for path in paths:
        active = [a[y, x] != bg for _, y, x in path]
        n = sum(active)
        if n == 0 or active != [True] * n + [False] * (len(path) - n):
            raise ValueError("each observed thread must be a visible prefix")
        observations = [(t, int(a[y, x])) for t, y, x in path[:n]]
        colors = {v for _, v in observations}
        if base not in colors or len(colors) != 2:
            raise ValueError("a thread must use its accent and the spoke color")
        accent, = colors - {base}
        if accent in (bg, marker):
            raise ValueError("illegal thread accent")
        word, errors = _recover(observations, base, accent)
        for _, y, x in path:
            if permitted[y, x]:
                raise ValueError("winding paths may not intersect")
            permitted[y, x] = True
        threads.append({"path": path, "observations": observations,
                        "n": n, "word": word, "period": len(word),
                        "accent": accent, "errors": errors})
        accents.append(accent)
    if len(set(accents)) != 4 or set(accents) & {bg, base, marker}:
        raise ValueError("four distinct arm accents are required")
    if np.any(a[~permitted] != bg):
        raise ValueError("unexpected cells outside the cross and windings")
    return {"background": bg, "base": base, "marker": marker,
            "center": center, "radius": radius, "hand": hand,
            "threads": threads, "permitted": permitted}


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _decode(a)
    answer = a.copy()
    for thread in info["threads"]:
        word = thread["word"]
        for t, y, x in thread["path"]:
            answer[y, x] = word[t % len(word)]
    return answer


def _shortcut_errors(info):
    """Counterfactuals: copy first phase observations; skip off-canvas time."""
    copied, visible_clock = 0, 0
    for th in info["threads"]:
        word, first = th["word"], {}
        p = len(word)
        for t, color in th["observations"]:
            first.setdefault(t % p, color)
        for i, (t, _, _) in enumerate(th["path"]):
            copied += word[t % p] != first[t % p]
            visible_clock += word[t % p] != word[i % p]
    return int(copied), int(visible_clock)


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _prefix_length(path, period, target):
    counts = [0] * period
    for n, (t, _, _) in enumerate(path, 1):
        counts[t % period] += 1
        if n >= target and min(counts) >= 4:
            return n
    return len(path)


def _thread(rng, path, period, base, accent, flipped_bit, early):
    target = int(rng.integers(4, 7)) * period + int(rng.integers(period))
    n = _prefix_length(path, period, target)
    if len(path) - n < 8:
        raise ValueError("not enough room for unseen continuation")
    indices = [t for t, _, _ in path[:n]]
    early_positions = [i for i, t in enumerate(indices) if t < period]
    error_index = int(rng.choice(early_positions)) if early else int(rng.integers(n))
    error_phase = indices[error_index] % period
    words = [tuple((bits >> k) & 1 for k in range(period))
             for bits in range(1, (1 << period) - 1)
             if (((bits >> error_phase) & 1) == flipped_bit
                 and len(_primitive(tuple((bits >> k) & 1
                                          for k in range(period)))) == period)]
    # Random proposals followed by a finite exhaustive one-flip fallback.
    proposals = [(words[int(rng.integers(len(words)))], True)
                 for _ in range(48)] + [(w, False) for w in words]
    for bits, random_noise in proposals:
        word = tuple(accent if b else base for b in bits)
        clean = [word[t % period] for t in indices]
        noisy = clean.copy()
        # One randomly chosen arm has an error in its first cycle. Other
        # arms may have their errors anywhere, including later repetitions.
        errors = [error_index]
        if random_noise:
            cap = min(3, period, max(1, n // 12))
            more = int(rng.integers(1, cap + 1)) - 1
            phases = [k for k in range(period) if k != error_phase]
            for phase in rng.choice(phases, size=more, replace=False):
                choices = [i for i, t in enumerate(indices) if t % period == phase]
                errors.append(int(rng.choice(choices)))
        for i in errors:
            noisy[i] = accent if noisy[i] == base else base
        observations = list(zip(indices, noisy))
        try:
            recovered, _ = _recover(observations, base, accent)
        except ValueError:
            continue
        if recovered == word:
            return n, word, noisy
    raise ValueError("periodic evidence is ambiguous in this crop")


def _shape_center(rng, radius, regime, is_test, teach, attempt):
    if regime == "grow":
        low, high = (26, 30) if is_test else (22, 25)
    elif regime == "compress":
        low, high = (19, 24) if is_test else (26, 30)
    else:
        low, high = (21, 30) if is_test else (22, 29)
    if attempt == 63:
        shape = (high, high)
        return shape, tuple((v - 1) // 2 for v in shape)
    shape = tuple(int(v) for v in rng.integers(low, high + 1, size=2))
    if rng.random() < .18:
        shape = (high, high)
    if teach or regime == "compress":
        center = tuple((s - 1) // 2 + int(rng.integers(-1, 2)) for s in shape)
    elif is_test and regime == "reentry":
        center = tuple(int(rng.choice([radius, s - 1 - radius]))
                       + (int(rng.integers(0, 2)) if rng.random() < .3 else 0)
                       for s in shape)
    elif rng.random() < .3:
        center = tuple((s - 1) // 2 for s in shape)
    else:
        center = tuple(int(rng.integers(radius, s - radius)) for s in shape)
    center = tuple(max(radius, min(s - 1 - radius, c))
                   for s, c in zip(shape, center))
    return shape, center


def _make_pair(rng, radius, hand, roles, regime, is_test, teach):
    bg, base, marker, *accents = roles
    periods = [2, 3] + [int(rng.integers(2, 6)) for _ in range(2)]
    rng.shuffle(periods)
    initial = [0, 1, int(rng.integers(2)), int(rng.integers(2))]
    rng.shuffle(initial)
    early_arm = int(rng.integers(4))
    for attempt in range(64):
        shape, center = _shape_center(rng, radius, regime, is_test, teach, attempt)
        paths = _routes(radius, shape, center, hand)
        try:
            plans = [_thread(rng, path, p, base, accent, first, arm == early_arm)
                     for arm, (path, p, accent, first)
                     in enumerate(zip(paths, periods, accents, initial))]
        except ValueError:
            continue
        if teach and any([t for t, _, _ in path[:n]] != list(range(n))
                         for path, (n, _, _) in zip(paths, plans)):
            continue
        if teach and radius >= 3 and not any(
                word[t % len(word)] != word[i % len(word)]
                for path, (_, word, _) in zip(paths, plans)
                for i, (t, _, _) in enumerate(path)):
            # The label must demonstrate that invisible steps consume phase.
            continue
        a = np.full(shape, bg, dtype=np.int8)
        a[_seed_mask(shape, center, radius)] = base
        a[center] = marker
        b = a.copy()
        for path, (n, word, noisy) in zip(paths, plans):
            for (_, y, x), color in zip(path[:n], noisy):
                a[y, x] = color
            for t, y, x in path:
                b[y, x] = word[t % len(word)]
        return {"input": a, "output": b}
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["grow", "reentry", "compress"]))
    hand = int(rng.choice([1, -1]))
    cap = int(rng.integers(3, 5)) if ntrain >= 3 else 3
    radii = [2, 3] + [int(rng.integers(2, cap + 1)) for _ in range(ntrain - 2)]
    if ntrain >= 3:
        radii[-1] = cap
    if rng.random() < .5:
        radii[0], radii[1] = radii[1], radii[0]
    if regime == "grow":
        radii += [max(radii) + 1] * ntest
    else:
        radii += [int(rng.choice(sorted(set(radii)))) for _ in range(ntest)]
    # Distinct ordered background/base roles guarantee unique inputs even
    # if geometry or motif proposals happen to coincide.
    assignments = [(bg, base) for bg in range(10) for base in range(10) if bg != base]
    order = rng.permutation(len(assignments))[:ntrain + ntest]
    puzzle = {"train": [], "test": []}
    for serial, radius in enumerate(radii):
        bg, base = assignments[int(order[serial])]
        other = rng.choice([c for c in range(10) if c not in (bg, base)],
                           size=5, replace=False)
        roles = (bg, base, *(int(c) for c in other))
        split = "train" if serial < ntrain else "test"
        puzzle[split].append(_make_pair(rng, radius, hand, roles, regime,
                                       split == "test", serial < 2))
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, descriptions = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            descriptions[split].append(info)
            bg, base, marker = info["background"], info["base"], info["marker"]
            assert len(np.unique(a)) == len(np.unique(b)) == 7
            assert set(np.unique(a)) == set(np.unique(b))
            seed = _seed_mask(a.shape, info["center"], info["radius"])
            assert np.all(a[seed] == base) and np.all(b[seed] == base)
            assert a[info["center"]] == b[info["center"]] == marker
            assert np.all(b[~info["permitted"]] == bg)
            flip_directions = set()
            early_error = False
            for thread in info["threads"]:
                p, word = thread["period"], thread["word"]
                observations, n = thread["observations"], thread["n"]
                assert min(Counter(t % p for t, _ in observations).values()) >= 4
                assert len({t % p for t, _ in observations}) == p
                assert 1 <= thread["errors"] <= max(1, n // 12)
                assert len(thread["path"]) - n >= 8
                errors_by_phase = Counter()
                for t, v in observations:
                    if v != word[t % p]:
                        errors_by_phase[t % p] += 1
                        flip_directions.add((v == base, word[t % p] == base))
                        early_error |= t < p
                assert max(errors_by_phase.values()) == 1
                for t, y, x in thread["path"]:
                    assert b[y, x] == word[t % p], "wrong winding phase or color"
                # An arm-wide majority loses the minority phases in the answer.
                assert len(set(word)) == 2
            assert flip_directions == {(True, False), (False, True)}
            assert early_error, "a copied initial unit must be contradicted"
            assert len({t["period"] for t in info["threads"]}) >= 2
            assert _shortcut_errors(info)[0] > 0
            assert not np.array_equal(a, b)
    train = descriptions["train"]
    assert {2, 3} <= {d["radius"] for d in train}
    assert len({d["hand"] for s in descriptions.values() for d in s}) == 1
    assert all([t for t, _ in th["observations"]] == list(range(th["n"]))
               for d in train[:2] for th in d["threads"])
    assert any(_shortcut_errors(d)[1] > 0 for d in train[:2])
    for d in descriptions["test"]:
        assert d["radius"] <= max(t["radius"] for t in train) + 1


def _measure(puzzles):
    counts = Counter()
    radii, periods, hands, shapes, centers, prefixes, words, roles = (set() for _ in range(8))
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = np.asarray(pair["input"]), np.asarray(pair["output"])
                d = _decode(a)
                radii.add(d["radius"])
                hands.add(d["hand"])
                shapes.add(a.shape)
                centers.add((a.shape, d["center"], d["radius"]))
                roles.add((d["background"], d["base"]))
                counts["pairs"] += 1
                counts["square_30"] += int(a.shape == (30, 30))
                counts["touching_seed"] += int(min(d["center"][0] - d["radius"],
                    d["center"][1] - d["radius"], a.shape[0] - 1 - d["center"][0] - d["radius"],
                    a.shape[1] - 1 - d["center"][1] - d["radius"]) == 0)
                counts["repairs"] += int(np.count_nonzero((a != b) & (a != d["background"])))
                counts["growth"] += int(np.count_nonzero((a == d["background"]) & (b != a)))
                copied, clock = _shortcut_errors(d)
                counts["copy_shortcut_wrong_pairs"] += int(copied > 0)
                counts["visible_clock_wrong_pairs"] += int(clock > 0)
                if split == "train":
                    counts["train_visible_clock_wrong_pairs"] += int(clock > 0)
                for th in d["threads"]:
                    periods.add(th["period"])
                    prefixes.add(th["n"])
                    words.add(tuple(int(v == th["accent"]) for v in th["word"]))
                    path = th["path"]
                    counts["threads"] += 1
                    counts["clipped_observed_prefix"] += int(any(
                        path[i][0] != i for i in range(th["n"])))
                    counts["reentry_threads"] += int(any(
                        q[0] != p[0] + 1 for p, q in zip(path, path[1:])))
                    counts["partial_observed_cycle"] += int(th["n"] % th["period"] != 0)
    return {**dict(counts), "radii": sorted(radii), "periods": sorted(periods),
            "hands": sorted(hands), "shapes": len(shapes), "layouts": len(centers),
            "prefix_lengths": len(prefixes), "binary_words": len(words),
            "ordered_background_base_roles": len(roles)}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = _measure(puzzles)
    assert metrics["radii"] == [2, 3, 4, 5]
    assert metrics["periods"] == [2, 3, 4, 5]
    assert metrics["hands"] == [-1, 1]
    assert metrics["shapes"] >= 65 and metrics["layouts"] >= 120
    assert metrics["binary_words"] >= 20 and metrics["prefix_lengths"] >= 20
    assert metrics["ordered_background_base_roles"] >= 60
    for name in ("square_30", "touching_seed", "clipped_observed_prefix",
                 "reentry_threads", "partial_observed_cycle", "repairs", "growth"):
        assert metrics[name] > 0, name
    for puzzle in puzzles:
        for pair in puzzle["train"] + puzzle["test"]:
            assert np.array_equal(solve(pair["input"]), pair["output"])
    assert all(_key(a[k]) == _key(b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s])
               for k in ("input", "output"))
    # Counts include every legal combination, with no expensive seed cross-product.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            generate_puzzle(6000 + 10 * ntrain + ntest, ntrain, ntest)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Rotation and reflection preserve distance clocks; reflection flips the hand.
    pair = puzzles[0]["train"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    # Explicit crop equivalence: tracing never stops on its first exit.
    for radius in range(2, 6):
        for hand in (1, -1):
            small = _routes(radius, (21, 29), (radius, 28 - radius), hand)
            large = _routes(radius, (81, 81), (40, 40), hand)
            for sp, lp in zip(small, large):
                cropped = [(t, y - 40 + radius, x - 40 + 28 - radius)
                           for t, y, x in lp
                           if 0 <= y - 40 + radius < 21 and 0 <= x - 40 + 28 - radius < 29]
                assert sp == cropped
    # A label mutation must fail geometric/color checks, not only solve equality.
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
           for s in ("train", "test")}
    b = bad["test"][0]["output"]
    d = _decode(bad["test"][0]["input"])
    _, y, x = d["threads"][0]["path"][-1]
    b[y, x] = d["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    return metrics


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
