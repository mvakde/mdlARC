"""Decode an expanded square, then turn its color layers using detached bars.

All conventions are input-visible or fixed by this family. No parent generator,
saved puzzle, filesystem resource, or generation state is used by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


SELF_TEST_SEEDS = (0, 751104139, 236391903, 292278387, 257208783, 886404345,
                  196359066, 781044165, 301599736, 797072586, 888869054,
                  414683195, 351992572, 324819508, 865589540, 325333822,
                  351413919, 856719606, 262538872, 574827815, 449598740,
                  58506039, 873768091, 723235608, 604186689, 324263059,
                  766484343, 932722976, 736838439, 223828735, 205212329,
                  680878486, 557867843, 466472682, 685302755, 202036987,
                  229033901, 302405321, 271520750, 172284523, 693247162,
                  725396135)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(a, bg):
    """Eight-connected occupied components; bars have a blank diagonal halo."""
    seen = a == bg
    h, w = a.shape
    result = []
    for rr, cc in np.argwhere(~seen):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        pending, cells = [(r, c)], []
        while pending:
            y, x = pending.pop()
            cells.append((y, x))
            for v in range(max(0, y - 1), min(h, y + 2)):
                for u in range(max(0, x - 1), min(w, x + 2)):
                    if not seen[v, u]:
                        seen[v, u] = True
                        pending.append((v, u))
        result.append(np.asarray(cells, dtype=int))
    return result


def _is_bar(a, cells):
    lo, hi = cells.min(axis=0), cells.max(axis=0)
    return (len(np.unique(a[cells[:, 0], cells[:, 1]])) == 1
            and (lo[0] == hi[0] or lo[1] == hi[1])
            and int(np.prod(hi - lo + 1)) == len(cells))


def _bbox(cells):
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _compress(a):
    """Only adjacent complete vectors are compared: no global deduplication."""
    rr = np.r_[0, 1 + np.flatnonzero(np.any(a[1:] != a[:-1], axis=1))]
    cc = np.r_[0, 1 + np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0))]
    rows = np.diff(np.r_[rr, a.shape[0]])
    cols = np.diff(np.r_[cc, a.shape[1]])
    return a[np.ix_(rr, cc)].copy(), rows, cols


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("The exterior border must identify a uniform background")
    components = _components(a, bg)
    complex_parts = [p for p in components if not _is_bar(a, p)]
    if not complex_parts:
        raise ValueError("Missing non-bar mosaic")
    bounds = _bbox(np.concatenate(complex_parts))
    top, left, bottom, right = bounds
    crop = a[top:bottom, left:right]
    # A single non-bar component establishes the mosaic's complete extent.
    if not any(_bbox(p) == bounds for p in complex_parts):
        raise ValueError("No spanning mosaic component")
    source, rows, cols = _compress(crop)
    if source.shape[0] != source.shape[1] or source.shape[0] < 3:
        raise ValueError("The decoded domain must be a square of side at least 3")
    if not np.array_equal(np.repeat(np.repeat(source, rows, axis=0), cols, axis=1), crop):
        raise ValueError("Not a row/column replication mosaic")
    palette = tuple(int(c) for c in np.unique(source) if c != bg)
    if len(palette) < 2:
        raise ValueError("At least two content layers are required")
    markers = {}
    for part in components:
        y0, x0, y1, x1 = _bbox(part)
        if top <= y0 and left <= x0 and y1 <= bottom and x1 <= right:
            continue
        if not _is_bar(a, part):
            raise ValueError("Every exterior component must be a monochrome bar")
        # No exterior cell can enter even the one-pixel halo of the entire crop.
        if np.any((part[:, 0] >= top - 1) & (part[:, 0] <= bottom)
                  & (part[:, 1] >= left - 1) & (part[:, 1] <= right)):
            raise ValueError("Counter touches the mosaic's halo")
        color = int(a[tuple(part[0])])
        if color not in palette or color in markers:
            raise ValueError("Exactly one bar per controlled content color")
        markers[color] = {"length": len(part), "cells": part,
                          "orientation": ("single" if len(part) == 1 else
                                          "horizontal" if y1 - y0 == 1 else "vertical")}
    if not markers:
        raise ValueError("At least one detached control is required")
    return {"background": bg, "source": source, "crop": crop, "bounds": bounds,
            "rows": rows, "cols": cols, "palette": palette, "markers": markers}


def _turn(r, c, n, k):
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _coordinate_answer(source, bg, counts):
    n = len(source)
    output = np.full(source.shape, bg, dtype=np.int8)
    for r, c in np.argwhere(source != bg):
        color = int(source[r, c])
        y, x = _turn(int(r), int(c), n, counts.get(color, 0))
        if output[y, x] != bg:
            raise ValueError("Rotated masks conflict; no overwrite priority exists")
        output[y, x] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    counts = {c: m["length"] for c, m in info["markers"].items()}
    return _coordinate_answer(info["source"], info["background"], counts)


def _array_answer(source, bg, counts):
    """Construction/checking uses whole-mask rotations instead of coordinates."""
    answer = np.full(source.shape, bg, dtype=np.int8)
    occupied = np.zeros(source.shape, dtype=bool)
    for color in np.unique(source):
        if color == bg:
            continue
        mask = np.rot90(source == color, -counts.get(int(color), 0) % 4)
        if np.any(occupied & mask):
            raise ValueError("Conflicting masks")
        answer[mask] = color
        occupied |= mask
    return answer


def _orbits(n):
    seen, orbits = set(), []
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                orbit = list(dict.fromkeys(_turn(r, c, n, k) for k in range(4)))
                seen.update(orbit)
                orbits.append(orbit)
    return orbits


def _boundaries(a):
    return (np.all(np.any(a[1:] != a[:-1], axis=1))
            and np.all(np.any(a[:, 1:] != a[:, :-1], axis=0)))


def _seam(a):
    return bool(np.any(np.count_nonzero(a[1:] != a[:-1], axis=1) == 1)
                or np.any(np.count_nonzero(a[:, 1:] != a[:, :-1], axis=0) == 1))


def _separated_repeat(a):
    return (any(np.array_equal(a[i], a[j]) for i in range(len(a))
                for j in range(i + 2, len(a)))
            or any(np.array_equal(a[:, i], a[:, j]) for i in range(a.shape[1])
                   for j in range(i + 2, a.shape[1])))


def _asymmetric(mask):
    return not np.array_equal(mask, np.rot90(mask, 2))


def _make_source(rng, n, bg, colors, witness, required_asym=()):
    """Disjoint rotational territories prevent conflicts, even for new counts.

    Territories are a construction device, not an extra rule for the solver.
    Their partial occupancy produces moving layers, holes, and multiple motifs.
    """
    orbits = _orbits(n)
    full = [i for i, o in enumerate(orbits) if len(o) == 4]
    border = [i for i in full if any(min(r, c, n - 1 - r, n - 1 - c) == 0
                                    for r, c in orbits[i])]
    k = len(colors)
    for _ in range(384):
        owners = rng.integers(k, size=len(orbits))
        # Mix random territories with radially grouped territories.
        if rng.random() < .45:
            order = sorted(full, key=lambda i: (min(min(r, c, n - 1 - r, n - 1 - c)
                                                    for r, c in orbits[i]), rng.random()))
            offset = int(rng.integers(k))
            for j, i in enumerate(order):
                owners[i] = (j * k // len(order) + offset) % k
        extent = int(rng.choice(border))
        if witness == 1:
            # A repeated-vector teaching case, not a compulsory family locator.
            for i in border:
                owners[i] = 1
            available = [i for i in full if i not in border]
            forced_full = set(border)
        else:
            available = [i for i in full if i != extent]
            forced_full = {extent}
        reserved = [int(i) for i in rng.choice(available, k, replace=False)]
        for j, i in enumerate(reserved):
            owners[i] = j
        symmetry = ["asymmetric"] * k if witness >= 0 else list(
            rng.choice(("asymmetric", "half", "quarter"), k, p=(.65, .20, .15)))
        densities = rng.uniform(.45, .94, size=k)
        motif = int(rng.integers(3))
        anchor = rng.uniform(0, n, size=2)
        source = np.full((n, n), bg, dtype=np.int8)
        for i, orbit in enumerate(orbits):
            owner = int(owners[i])
            if i in forced_full:
                selected = orbit
            elif len(orbit) == 1:
                selected = orbit if rng.random() < .7 else []
            elif symmetry[owner] == "quarter":
                selected = orbit if rng.random() < densities[owner] or i == reserved[owner] else []
            elif symmetry[owner] == "half":
                phase = int(rng.integers(2))
                selected = orbit[phase::2]
                if i != reserved[owner] and rng.random() < densities[owner]:
                    selected = orbit
            else:
                amount = int(rng.binomial(4, densities[owner]))
                if i == reserved[owner]:
                    amount = int(rng.choice((1, 3)))
                scores = rng.random(4)
                if motif == 1:
                    scores += np.array([(r - anchor[0]) ** 2 + (c - anchor[1]) ** 2
                                        for r, c in orbit]) / max(1, n)
                elif motif == 2:
                    scores += np.array([(r + int(anchor[0])) % 3 for r, c in orbit])
                selected = [orbit[int(j)] for j in np.argsort(scores)[:amount]]
            for r, c in selected:
                source[r, c] = colors[owner]
        if not _boundaries(source):
            continue
        parts = _components(source, bg)
        if not any(not _is_bar(source, p) and _bbox(p) == (0, 0, n, n) for p in parts):
            continue
        if witness == 1 and not _seam(source):
            continue
        if witness >= 0 and not all(_asymmetric(source == color) for color in colors):
            continue
        if not all(_asymmetric(source == colors[j]) for j in required_asym):
            continue
        return source
    raise RuntimeError("Bounded source construction exhausted")


def _scales(rng, n, teach, strong, modest):
    for _ in range(128):
        mode = 3 if teach else int(rng.integers(4))
        upper = 4 if modest else 5
        rows, cols = rng.integers(1, upper, size=(2, n))
        if mode in (0, 1):
            rows[:] = int(rng.integers(2, min(4, 22 // n + 1)))
        if mode in (0, 2):
            cols[:] = int(rng.integers(2, min(4, 22 // n + 1)))
        if strong:
            rows, cols = rng.integers(1, 3, size=(2, n))
            axis = rows if rng.random() < .5 else cols
            axis[int(rng.integers(n))] = int(rng.integers(6, 10))
        if max(rows.sum(), cols.sum()) > 22 or max(rows) == 1 or max(cols) == 1:
            continue
        if teach and (len(set(rows)) == 1 or len(set(cols)) == 1):
            continue
        return rows, cols
    raise RuntimeError("Bounded scale construction exhausted")


def _bar_groups(blocked, length, required):
    h, w = blocked.shape
    groups = []
    for orientation in ("horizontal", "vertical"):
        if required is not None and required != orientation:
            continue
        bh, bw = (1, length) if orientation == "horizontal" else (length, 1)
        rr, cc = np.indices((h - bh + 1, w - bw + 1))
        clear = np.ones(rr.shape, dtype=bool)
        for i in range(length):
            clear &= ~blocked[rr + (i if bh > 1 else 0), cc + (i if bw > 1 else 0)]
        starts = np.argwhere(clear)
        if len(starts):
            groups.append((orientation, starts))
    return groups


def _embed(rng, expanded, bg, counts, forced):
    mh, mw = expanded.shape
    for attempt in range(40):
        if attempt >= 16:
            h = w = 30
            top = left = 1
        else:
            h = int(rng.integers(max(mh + 4, 14), 31))
            w = int(rng.integers(max(mw + 4, 14), 31))
            top = int(rng.integers(1, h - mh))
            left = int(rng.integers(1, w - mw))
        a = np.full((h, w), bg, dtype=np.int8)
        a[top:top + mh, left:left + mw] = expanded
        blocked = np.zeros((h, w), dtype=bool)
        blocked[0] = blocked[-1] = True
        blocked[:, 0] = blocked[:, -1] = True
        blocked[top - 1:top + mh + 1, left - 1:left + mw + 1] = True
        good = True
        for color in sorted(counts, key=lambda c: -counts[c]):
            length = counts[color]
            groups = _bar_groups(blocked, length, forced.get(color))
            if not groups:
                good = False
                break
            orientation, starts = groups[int(rng.integers(len(groups)))]
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            for i in range(length):
                r, c = (y, x + i) if orientation == "horizontal" else (y + i, x)
                a[r, c] = color
                blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
        if good:
            return a
    raise RuntimeError("Bounded counter packing exhausted")


def _make_pair(rng, split, mode, witness, previous_size=None, previous_bg=None):
    if mode == 2:
        sizes = list(range(5, 8)) if split == "train" else [8, 9]
    else:
        sizes = list(range(4, 9))
    if witness == 0:
        sizes = [n for n in sizes if n >= 5]
    if witness == 1:
        sizes = [n for n in sizes if n >= 6 and n != previous_size]
    n = int(rng.choice(sizes))
    maximum = min(5, (n * n) // 4 - 1)
    if witness == 1:
        maximum = min(maximum, ((n - 2) ** 2) // 4)
    minimum = 4 if witness == 0 else 3 if witness == 1 else 2
    k = int(rng.integers(minimum, maximum + 1))
    bg = int(rng.choice([c for c in range(10) if c != previous_bg]))
    colors = [int(c) for c in rng.choice([c for c in range(10) if c != bg], k, replace=False)]
    cap = 5 if mode == 0 and split == "train" else 9
    lengths = [int(rng.integers(cap + 1)) for _ in colors]
    forced = {}
    if witness == 0:
        lengths[:4] = [1, 2, 2, 3]
        orientations = ("horizontal", "vertical") if rng.random() < .5 else ("vertical", "horizontal")
        forced = {colors[1]: orientations[0], colors[2]: orientations[1]}
    elif witness == 1:
        lengths[:3] = [5, 4, 0]
    if not any(lengths):
        lengths[0] = 1
    if split == "test" and mode == 0:
        lengths[0] = int(rng.choice((6, 7, 9)))
    counts = {c: length for c, length in zip(colors, lengths) if length}
    # Witness-free pairs must still have a visible action; symmetric layers
    # remain allowed alongside whichever layer establishes that action.
    for _ in range(32):
        required = (0,) if split == "test" and mode == 0 else ()
        source = _make_source(rng, n, bg, colors, witness, required)
        output = _array_answer(source, bg, counts)
        if not np.array_equal(output, source):
            break
        if witness < 0:
            counts[colors[0]] = 1 if split != "test" or mode != 0 else 9
    else:
        raise RuntimeError("Could not sample a moving layer")
    rows, cols = _scales(rng, n, witness >= 0, split == "test" and mode == 1, mode == 1)
    expanded = np.repeat(np.repeat(source, rows, axis=0), cols, axis=1)
    input_grid = _embed(rng, expanded, bg, counts, forced)
    parsed = _parse(input_grid)
    assert np.array_equal(parsed["source"], source)
    return {"input": input_grid, "output": output}, n, bg


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    used, previous_size, previous_bg = set(), None, None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            witness = i if split == "train" and i < 2 else -1
            for _ in range(16):
                pair, n, bg = _make_pair(rng, split, mode, witness, previous_size, previous_bg)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    previous_size, previous_bg = n, bg
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(infos):
    lengths, evidence = set(), set()
    for info in infos:
        source, bg = info["source"], info["background"]
        orientations = {}
        residues = set()
        for color in info["palette"]:
            if not _asymmetric(source == color):
                continue
            if color not in info["markers"]:
                evidence.add("unmarked")
                continue
            marker = info["markers"][color]
            length = marker["length"]
            lengths.add(length)
            residues.add(length % 4)
            if length % 2:
                evidence.add("clockwise")
            orientations.setdefault(length, set()).add(marker["orientation"])
        if len(residues) >= 2:
            evidence.add("independent")
        if any({"horizontal", "vertical"} <= values for values in orientations.values()):
            evidence.add("orientation")
        uneven = len(set(info["rows"])) > 1 and len(set(info["cols"])) > 1
        if uneven:
            evidence.add("nonuniform")
        if uneven and _seam(source):
            evidence.add("hidden_seam")
        if _separated_repeat(source):
            evidence.add("separated_repeat")
    return lengths, evidence


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, answer = _grid(pair["input"]), _grid(pair["output"])
            key = (a.shape, a.tobytes())
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _parse(a)
            source, bg = info["source"], info["background"]
            counts = {c: m["length"] for c, m in info["markers"].items()}
            assert _boundaries(source), "An adjacent logical boundary is invisible"
            assert max(info["rows"]) > 1 and max(info["cols"]) > 1
            assert all(1 <= v <= 9 for v in counts.values())
            assert answer.shape == source.shape
            assert np.array_equal(answer, _array_answer(source, bg, counts))
            assert np.array_equal(answer, solve(a)), "Coordinate and mask calculations disagree"
            assert not np.array_equal(source, answer), "Every sampled pair must expose rotation"
            assert info["crop"].shape != answer.shape, "Every sampled pair must expose decoding"
            for color in (bg,) + info["palette"]:
                assert np.count_nonzero(source == color) == np.count_nonzero(answer == color)
            if split == "train":
                training.append(info)
            else:
                testing.append(info)
    lengths, evidence = _training_evidence(training)
    assert {1, 2, 3, 4, 5} <= lengths, "Missing informative unary/modulo demonstrations"
    assert evidence == {"unmarked", "clockwise", "independent", "orientation",
                        "nonuniform", "hidden_seam", "separated_repeat"}
    train_count_max = max(m["length"] for info in training for m in info["markers"].values())
    train_scale_max = max(int(max(max(info["rows"]), max(info["cols"]))) for info in training)
    train_size_max = max(len(info["source"]) for info in training)
    for info in testing:
        long_action = any(m["length"] >= 6 and _asymmetric(info["source"] == c)
                          and m["length"] % 4 for c, m in info["markers"].items())
        scale_max = int(max(max(info["rows"]), max(info["cols"])))
        assert ((train_count_max <= 5 and long_action)
                or (train_scale_max <= 3 and scale_max >= 6)
                or (train_size_max <= 7 and len(info["source"]) >= 8)), "Missing visible train/test shift"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming evidence over the 42 requested seeds; no disk writes."""
    repeated = np.array([[1, 1, 1, 1], [2, 0, 3, 2],
                         [1, 1, 1, 1], [3, 2, 0, 3]], dtype=np.int8)
    expanded = np.repeat(np.repeat(repeated, [2, 1, 4, 2], axis=0), [1, 3, 2, 4], axis=1)
    assert np.array_equal(_compress(expanded)[0], repeated)
    assert _separated_repeat(repeated)
    # Hand-specified small member, including a non-frame source and an explicit
    # expected answer. It also probes a size below the sampler's 4..9 range.
    compact = np.array([[1, 2, 1], [2, 0, 2], [1, 0, 0]], dtype=np.int8)
    pixel = np.repeat(np.repeat(compact, [2, 1, 3], axis=0), [1, 3, 2], axis=1)
    small = np.zeros((12, 14), dtype=np.int8)
    small[2:8, 2:8] = pixel
    small[10, 2] = 1
    small[10, 6:9] = 2
    expected = np.array([[1, 2, 1], [2, 0, 0], [0, 2, 1]], dtype=np.int8)
    assert np.array_equal(solve(small), expected)
    try:
        wrong_order = _compress(_array_answer(pixel, 0, {1: 1, 2: 3}))[0]
    except ValueError:
        pass  # Pixel-space turns can even conflict despite valid compact turns.
    else:
        assert not np.array_equal(wrong_order, expected), "Decode-before-turn witness failed"
    # Explicit collision: rotating (0,0) once reaches stationary (0,2).
    collision = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
    try:
        _coordinate_answer(collision, 0, {1: 1})
    except ValueError:
        pass
    else:
        raise AssertionError("Mask conflict was accepted")
    fingerprints, shapes, sizes, backgrounds, layers, profiles = set(), set(), set(), set(), set(), set()
    symmetries, lengths, orientations = set(), set(), set()
    changes = {"decoding": 0, "rotation": 0, "repeat": 0, "seam": 0}
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"].copy()
                info = _parse(a)
                source, bg = info["source"], info["background"]
                assert np.array_equal(solve(a), pair["output"])
                assert np.array_equal(a, pair["input"]), "Solver mutated its input"
                shapes.add(a.shape)
                sizes.add(len(source))
                backgrounds.add(bg)
                layers.add(len(info["palette"]))
                profiles.add((tuple(info["rows"]), tuple(info["cols"])))
                for color in info["palette"]:
                    mask = source == color
                    symmetries.add("quarter" if np.array_equal(mask, np.rot90(mask)) else
                                   "half" if np.array_equal(mask, np.rot90(mask, 2)) else "asymmetric")
                for marker in info["markers"].values():
                    lengths.add(marker["length"])
                    orientations.add(marker["orientation"])
                changes["decoding"] += info["crop"].shape != pair["output"].shape
                changes["rotation"] += not np.array_equal(source, pair["output"])
                changes["repeat"] += _separated_repeat(source)
                changes["seam"] += _seam(source)
    assert len(fingerprints) == 42 and len(shapes) >= 30 and len(profiles) >= 100
    assert len(sizes) >= 5 and backgrounds == set(range(10)) and layers == {2, 3, 4, 5}
    assert symmetries == {"quarter", "half", "asymmetric"} and lengths == set(range(1, 10))
    assert orientations == {"single", "horizontal", "vertical"}
    assert changes["decoding"] == changes["rotation"] == 210
    assert changes["repeat"] >= 42 and changes["seam"] >= 42
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generate_puzzle(12345 + 10 * train + test, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    damaged = generate_puzzle(871, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    damaged = generate_puzzle(872, 2, 1)
    pair = damaged["test"][0]
    info = _parse(pair["input"])
    marker = next(iter(info["markers"].values()))
    r, c = map(int, marker["cells"][0])
    absent = next(c for c in range(10) if c != info["background"] and c not in info["palette"])
    pair["input"][r, c] = absent
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Unmatched counter color accepted")
    return {"seeds": 42, "pairs": 210, "input_shapes": len(shapes),
            "compact_sizes": sorted(sizes), "layer_counts": sorted(layers),
            "expansion_profiles": len(profiles), "ablation_witnesses": changes}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
