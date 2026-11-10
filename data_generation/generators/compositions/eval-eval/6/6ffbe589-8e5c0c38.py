"""Counter-aligned mirror mosaics. Standard library and NumPy only.

The solver reads one input; no generation parameters or puzzle state are used.
Generation constructs labeled symmetric occupancy, adds unmatched cells, and
scrambles the color layers. Prediction instead rotates input coordinates and
tests visible partners. See task_spec.md for the complete family grammar.
"""
from __future__ import annotations

import copy
import json
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _locate(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= a.size / 2:
        raise ValueError("the background must occupy more than half the input")
    candidates = []
    for anchor in range(10):
        if anchor == bg or counts[anchor] != 4:
            continue
        points = set(map(tuple, np.argwhere(a == anchor).tolist()))
        rr, cc = zip(*points)
        top, bottom, left, right = min(rr), max(rr), min(cc), max(cc)
        if bottom - top != right - left or bottom - top < 4:
            continue
        if points != {(top, left), (top, right), (bottom, left), (bottom, right)}:
            continue
        # Only the corner marks may occupy the bounding square's perimeter.
        perimeter = np.zeros(a.shape, bool)
        perimeter[top, left:right+1] = perimeter[bottom, left:right+1] = True
        perimeter[top:bottom+1, left] = perimeter[top:bottom+1, right] = True
        if np.any(perimeter & (a != bg) & (a != anchor)):
            continue
        source = a[top+1:bottom, left+1:right].copy()
        colors = tuple(int(x) for x in np.unique(source) if x != bg)
        if not 2 <= len(colors) <= 6 or anchor in colors:
            continue
        exterior = (a != bg).copy()
        exterior[top:bottom+1, left:right+1] = False
        bars = {}
        good = True
        for color in np.unique(a[exterior]):
            color = int(color)
            p = np.argwhere(exterior & (a == color))
            rows, cols = p.T
            horizontal = np.ptp(rows) == 0 and np.ptp(cols) + 1 == len(p)
            vertical = np.ptp(cols) == 0 and np.ptp(rows) + 1 == len(p)
            if color not in colors or not 1 <= len(p) <= 9 or not (horizontal or vertical):
                good = False
                break
            if any(top-1 <= r <= bottom+1 and left-1 <= c <= right+1 for r, c in p):
                good = False
                break
            bars[color] = {"length": len(p), "orientation": "single" if len(p) == 1
                           else "horizontal" if horizontal else "vertical",
                           "cells": tuple(map(tuple, p.tolist()))}
        if not good:
            continue
        items = list(bars.items())
        if any(max(abs(r-y), abs(c-x)) <= 1
               for i, (_, one) in enumerate(items) for _, two in items[i+1:]
               for r, c in one["cells"] for y, x in two["cells"]):
            continue
        candidates.append({"grid": a, "background": bg, "anchor": anchor,
                           "top": top, "left": left, "size": len(source),
                           "source": source, "colors": colors, "bars": bars})
    if len(candidates) != 1:
        raise ValueError("there must be one unambiguous four-corner working square")
    return candidates[0]


def _rotate_coordinates(source, bg, counts, direction=1, absent=0):
    """Input-coordinate implementation, also checking collision/boundary rules."""
    n = len(source)
    aligned = np.full_like(source, bg)
    for r, c in np.argwhere(source != bg):
        r, c = int(r), int(c)
        color = int(source[r, c])
        turns = direction * counts.get(color, absent) % 4
        y, x = r, c
        for _ in range(turns):
            y, x = x, n - 1 - y
        if aligned[y, x] != bg:
            raise ValueError("commanded layers collide")
        aligned[y, x] = color
    return aligned


def _supported(aligned, bg, per_color=False):
    reflected = aligned[:, ::-1]
    keep = ((aligned != bg) & (reflected == aligned) if per_color
            else (aligned != bg) & (reflected != bg))
    return np.where(keep, aligned, bg).astype(np.int8)


def _prediction(info, direction=1, absent=0, per_color=False):
    counts = {c: b["length"] for c, b in info["bars"].items()}
    aligned = _rotate_coordinates(info["source"], info["background"], counts, direction, absent)
    return _supported(aligned, info["background"], per_color), aligned


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Align color layers by unary counters, then retain joint mirror support."""
    return _prediction(_locate(input_grid))[0]


def _orbits(n):
    unseen = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while unseen:
        r, c = min(unseen)
        orbit = []
        for _ in range(4):
            if (r, c) not in orbit:
                orbit.append((r, c))
            r, c = c, n-1-r
        unseen.difference_update(orbit)
        result.append(orbit)
    return result


def _owners(rng, n, colors):
    """Invisible collision prevention; reflection can cross orbit ownership."""
    orbits = _orbits(n)
    style = int(rng.integers(3))
    if style == 0:
        order = rng.permutation(len(orbits))
    elif style == 1:
        order = np.argsort([min(r, c, n-1-r, n-1-c) + rng.uniform(0, 2.5)
                            for orbit in orbits for r, c in [orbit[0]]])
    else:
        ar, ac = rng.uniform(0, n-1, 2)
        anchors = [(ar, ac), (ac, n-1-ar), (n-1-ar, n-1-ac), (n-1-ac, ar)]
        order = np.argsort([min((r-y)**2+(c-x)**2 for y, x in anchors) + rng.uniform(0, 2)
                            for orbit in orbits for r, c in [orbit[0]]])
    lengths = np.ones(len(colors), int)
    lengths += rng.multinomial(len(orbits)-len(colors), rng.dirichlet(np.full(len(colors), 3.0)))
    shuffled = rng.permutation(len(colors))
    owners = np.empty((n, n), np.int8)
    cursor = 0
    for index in shuffled:
        for i in order[cursor:cursor+lengths[index]]:
            for p in orbits[int(i)]:
                owners[p] = colors[index]
        cursor += int(lengths[index])
    return owners


def _core_mask(rng, n):
    """Procedural occupancy motifs, always centered left/right but free in y."""
    mask = np.zeros((n, n), bool)
    style = int(rng.integers(6))
    top = int(rng.integers(0, max(1, n//3)))
    bottom = int(rng.integers(max(top+3, n*2//3), n+1))
    radius = n//2
    center = (n-1)/2
    max_radius = int(rng.integers(max(1, radius//2), radius+1))
    if style == 0:  # Contours and nested interrupted contours.
        left = radius-max_radius
        right = n-1-left
        thickness = int(rng.integers(1, max(2, max_radius//2+1)))
        mask[top:bottom, left:left+thickness] = True
        mask[top:bottom, right-thickness+1:right+1] = True
        mask[top:top+thickness, left:right+1] = True
        mask[max(top, bottom-thickness):bottom, left:right+1] = True
        if rng.random() < .5:
            mask[int(rng.integers(top, bottom)), left:right+1] = False
    elif style in (1, 2, 5):
        level = max_radius
        hollow = rng.random() < .45
        phase = rng.uniform(0, 2*np.pi)
        period = rng.uniform(2.5, max(3, bottom-top))
        for r in range(top, bottom):
            if style == 1:  # Uneven lobes, necks, tips.
                level = int(round(max_radius*(.5+.5*np.cos((r-top)*2*np.pi/period+phase))))
            elif style == 2:  # Correlated irregular profiles.
                level = int(np.clip(level+rng.integers(-2, 3), 0, max_radius))
            else:  # Branches, with some empty rows / separated islands.
                level = int(rng.integers(0, max_radius+1))
                if rng.random() < .25:
                    continue
            for c in range((n+1)//2):
                distance = abs(c-center)
                if distance <= level + (0.5 if n % 2 == 0 else 0):
                    if (distance >= max(0, level-1) if hollow else rng.random() < .8):
                        mask[r, c] = mask[r, n-1-c] = True
    elif style == 3:  # Symmetric rectangular islands, possibly axis singletons.
        for _ in range(int(rng.integers(2, 6))):
            r = int(rng.integers(top, bottom))
            c = int(rng.integers((n+1)//2))
            height = int(rng.integers(1, min(4, bottom-r)+1))
            width = int(rng.integers(1, min(3, n-c)+1))
            mask[r:r+height, c:c+width] = True
        mask |= mask[:, ::-1]
    else:  # Patterned bands and paired constellations.
        density = rng.uniform(.18, .52)
        phase = int(rng.integers(3))
        for r in range(top, bottom):
            for c in range((n+1)//2):
                if rng.random() < density + (.25 if (r+c+phase) % 3 == 0 else 0):
                    mask[r, c] = mask[r, n-1-c] = True
    # Empty/full degeneracies do not carry enough examples of support vs noise.
    return mask


def _add_noise(rng, clean, bg, owners, amount):
    n = len(clean)
    aligned = clean.copy()
    core_cells = np.argwhere(clean != bg)
    r0, c0 = core_cells.min(0)
    r1, c1 = core_cells.max(0)
    for _ in range(amount):
        occupied = aligned != bg
        candidates = [(r, c) for r in range(n) for c in range(n)
                      if not occupied[r, c] and not occupied[r, n-1-c] and c != n-1-c]
        if not candidates:
            break
        inside, attached, detached = [], [], []
        for p in candidates:
            r, c = p
            if r0 <= r <= r1 and c0 <= c <= c1:
                inside.append(p)
            near = any(0 <= r+dy < n and 0 <= c+dx < n and clean[r+dy, c+dx] != bg
                       for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)))
            (attached if near else detached).append(p)
        pool = [inside, attached, detached][int(rng.integers(3))] or candidates
        p = pool[int(rng.integers(len(pool)))]
        aligned[p] = owners[p]
    return aligned


def _pack(rng, source, bg, anchor, counts, forced):
    n = len(source)
    box = n+2
    colors = sorted(counts, key=lambda c: -counts[c])
    for attempt in range(36):
        if attempt >= 24:
            h = w = 30
            top = left = 0
        else:
            h, w = (int(rng.integers(box+2, 31)) for _ in range(2))
            top = int(rng.integers(h-box+1))
            left = int(rng.integers(w-box+1))
            if rng.random() < .3:
                top = int(rng.choice([0, h-box]))
            if rng.random() < .3:
                left = int(rng.choice([0, w-box]))
        inp = np.full((h, w), bg, np.int8)
        inp[top+1:top+n+1, left+1:left+n+1] = source
        for r, c in ((top,left), (top,left+box-1), (top+box-1,left), (top+box-1,left+box-1)):
            inp[r, c] = anchor
        blocked = np.zeros((h, w), bool)
        blocked[max(0,top-1):min(h,top+box+1), max(0,left-1):min(w,left+box+1)] = True
        success = True
        for color in colors:
            length = counts[color]
            choices = []
            for orientation in ("horizontal", "vertical"):
                if color in forced and forced[color] != orientation:
                    continue
                bh, bw = (1,length) if orientation == "horizontal" else (length,1)
                for r in range(h-bh+1):
                    for c in range(w-bw+1):
                        if not blocked[r:r+bh, c:c+bw].any():
                            choices.append((r,c,bh,bw))
            if not choices:
                success = False
                break
            r, c, bh, bw = choices[int(rng.integers(len(choices)))]
            inp[r:r+bh, c:c+bw] = color
            blocked[max(0,r-1):min(h,r+bh+1), max(0,c-1):min(w,c+bw+1)] = True
        if not success or np.count_nonzero(inp == bg) <= inp.size/2:
            continue
        try:
            info = _locate(inp)
        except ValueError:
            continue
        if info["anchor"] == anchor and info["size"] == n and info["top"] == top and info["left"] == left:
            return inp
    return None


def _interaction_witness(info, expected):
    """A command change must alter support of a color whose command did not move."""
    bg = info["background"]
    counts = {c: b["length"] for c, b in info["bars"].items()}
    for moved in info["colors"]:
        altered = dict(counts)
        altered[moved] = counts.get(moved, 0)+1
        try:
            alternative = _supported(_rotate_coordinates(info["source"], bg, altered), bg)
        except ValueError:
            continue
        if any(not np.array_equal(expected == other, alternative == other)
               for other in info["colors"] if other != moved):
            return True
    return False


def _evidence(info, expected):
    bg = info["background"]
    counts = {c: b["length"] for c, b in info["bars"].items()}
    correct, aligned = _prediction(info)
    raw_clean = _supported(info["source"], bg)
    early = _rotate_coordinates(raw_clean, bg, counts)
    ranked = {color: sorted(set(counts.values())).index(k)+1 for color, k in counts.items()}
    rank_output = _supported(_rotate_coordinates(info["source"], bg, ranked), bg)
    return {
        "deletion": not np.array_equal(aligned, expected),
        "rotation": not np.array_equal(raw_clean, expected),
        "order": not np.array_equal(early, expected),
        "cross_color": not np.array_equal(_supported(aligned, bg, True), expected),
        "clockwise": not np.array_equal(_prediction(info, direction=-1)[0], correct),
        "ranking": not np.array_equal(rank_output, correct),
        "unmarked": bool(set(info["colors"])-set(counts)) and
                    not np.array_equal(_prediction(info, absent=1)[0], correct),
        "interaction": _interaction_witness(info, expected),
    }


def _make_pair(rng, mode, split, index, previous_bg):
    witness = index if split == "train" and index < 2 else -1
    if witness >= 0:
        layer_count = 3
    elif mode == 2:
        layer_count = int(rng.integers(4, 7) if split == "test" else rng.integers(2, 4))
    else:
        layer_count = int(rng.integers(2, 7))
    minimum = max(6, layer_count+3)
    if mode == 1:
        n = int(rng.integers(12, 20) if split == "test" else rng.integers(minimum, 12))
    else:
        n = int(rng.integers(minimum, 20))
    palette = list(map(int, rng.permutation(10)))
    bg = next(c for c in palette if c != previous_bg)
    palette.remove(bg)
    anchor, colors = palette[0], palette[1:layer_count+1]
    cap = 9 if mode == 0 and split == "test" else 4
    lengths = [int(rng.integers(0, cap+1)) for _ in colors]
    forced = {}
    if witness == 0:
        lengths = [1, 2, 2]
        two = ("horizontal", "vertical") if rng.random() < .5 else ("vertical", "horizontal")
        forced = {colors[1]: two[0], colors[2]: two[1]}
    elif witness == 1:
        lengths = [3, 0, 4]
    elif mode == 0 and split == "test":
        lengths[int(rng.integers(layer_count))] = int(rng.integers(5, 10))
    counts = {c: k for c, k in zip(colors, lengths) if k}
    for _ in range(160):
        owners = _owners(rng, n, colors)
        mask = _core_mask(rng, n)
        if not 8 <= int(mask.sum()) <= n*n*.72:
            continue
        clean = np.where(mask, owners, bg).astype(np.int8)
        if any(np.count_nonzero(clean == c) < 2 for c in colors):
            continue
        noise_count = 0 if witness < 0 and rng.random() < .12 else int(rng.integers(2, max(3, min(20, n*n//6))+1))
        aligned = _add_noise(rng, clean, bg, owners, noise_count)
        # Generation uses whole-array rotations, unlike the solver's coordinates.
        source = np.full_like(clean, bg)
        for color in colors:
            raw_mask = np.rot90(aligned == color, counts.get(color, 0) % 4)
            assert not np.any(raw_mask & (source != bg))
            source[raw_mask] = color
        cheap = {"background": bg, "source": source, "colors": tuple(colors),
                 "bars": {c: {"length": k} for c, k in counts.items()}}
        evidence = _evidence(cheap, clean)
        if witness >= 0 and not all(evidence[k] for k in
                ("deletion", "rotation", "order", "cross_color", "clockwise", "interaction")):
            continue
        if witness == 1 and not evidence["unmarked"]:
            continue
        if witness == 1 and not evidence["ranking"]:
            continue
        inp = _pack(rng, source, bg, anchor, counts, forced)
        if inp is not None:
            return {"input": inp, "output": clean}
    raise RuntimeError("bounded mosaic construction exhausted; no ambiguous fallback is used")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be an integer from 1 through 4, below num_train")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    puzzle = {"train": [], "test": []}
    seen, previous_bg = set(), None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, mode, split, index, previous_bg)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            previous_bg = _locate(pair["input"])["background"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _locate(inp)
            expected, aligned = _prediction(info)
            bg, n = info["background"], info["size"]
            assert out.shape == (n, n) and np.array_equal(out, expected), "incorrect alignment or support"
            assert np.array_equal(out != bg, (out != bg)[:, ::-1]), "joint occupancy must be symmetric"
            assert np.all((out == aligned) | (out == bg)), "cleaning only deletes aligned cells"
            assert set(map(int, np.unique(out))) - {bg} == set(info["colors"]), "every layer must survive"
            for color in info["colors"]:
                assert np.count_nonzero(info["source"] == color) == np.count_nonzero(aligned == color)
                assert 2 <= np.count_nonzero(out == color) <= np.count_nonzero(aligned == color)
            assert np.array_equal(solve(inp), out)
            infos[split].append(info)
    evidence = [_evidence(i, _grid(p["output"])) for i, p in zip(infos["train"], puzzle["train"])]
    for feature in ("deletion", "rotation", "order", "cross_color", "clockwise", "interaction", "unmarked", "ranking"):
        assert any(e[feature] for e in evidence), f"missing teaching evidence: {feature}"
    lengths = {b["length"] for i in infos["train"] for b in i["bars"].values()}
    assert {1, 2, 3, 4} <= lengths, "teach short unary counts including a full cycle"
    assert any(any({b["orientation"] for b in i["bars"].values() if b["length"] == k}
                       == {"horizontal", "vertical"} for k in lengths) for i in infos["train"]), "teach count, not bar direction"
    assert len({i["background"] for i in infos["train"]}) >= 2
    training, testing = infos["train"], infos["test"]
    # Observable single-dimension curricula, no hidden sampled-mode reliance.
    train_max = max(b["length"] for i in training for b in i["bars"].values())
    long_shift = train_max <= 4 and all(any(b["length"] >= 5 for b in i["bars"].values()) for i in testing)
    size_shift = max(i["size"] for i in training) < min(i["size"] for i in testing)
    count_shift = max(len(i["colors"]) for i in training) < min(len(i["colors"]) for i in testing)
    assert long_shift or size_shift or count_shift, "missing visible train/test shift"


def _measure(pair):
    info = _locate(pair["input"])
    out = _grid(pair["output"])
    _, aligned = _prediction(info)
    bg = info["background"]
    occupied = out != bg
    cross = occupied & (out[:, ::-1] != out)
    rows, cols = np.nonzero(occupied)
    return {"shape": tuple(info["grid"].shape), "size": info["size"],
            "background": bg, "anchor": info["anchor"], "layers": len(info["colors"]),
            "lengths": tuple(sorted(b["length"] for b in info["bars"].values())),
            "orientations": tuple(sorted(b["orientation"] for b in info["bars"].values())),
            "position": (info["top"], info["left"]),
            "retained": int(occupied.sum()), "noise": int(np.count_nonzero(aligned != bg)-occupied.sum()),
            "cross_cells": int(cross.sum()), "density": float(occupied.mean()),
            "extent": (int(np.ptp(rows)+1), int(np.ptp(cols)+1)),
            "axis_cells": int(occupied[:, info["size"]//2].sum()) if info["size"] % 2 else 0,
            "edge": info["top"] == 0 or info["left"] == 0 or
                    info["top"]+info["size"]+2 == info["grid"].shape[0] or
                    info["left"]+info["size"]+2 == info["grid"].shape[1]}


def self_test():
    """Bounded family checks over 42 seeds; no files or external tools required."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        records.extend(_measure(p) for s in ("train", "test") for p in puzzle[s])
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(signatures) == 42
    assert len({r["shape"] for r in records}) >= 35
    assert len({r["size"] for r in records}) >= 10
    assert {r["background"] for r in records} == set(range(10))
    assert {r["anchor"] for r in records} == set(range(10))
    assert {r["layers"] for r in records} == set(range(2, 7))
    assert {k for r in records for k in r["lengths"]} == set(range(1, 10))
    assert any(r["axis_cells"] for r in records) and any(r["size"] % 2 == 0 for r in records)
    assert any(r["noise"] == 0 for r in records) and any(r["noise"] >= 10 for r in records)
    assert any(r["cross_cells"] >= 20 for r in records)
    assert any(r["density"] < .2 for r in records) and any(r["density"] > .5 for r in records)
    assert any(r["edge"] for r in records) and any(not r["edge"] for r in records)
    # A hand-built member that does NOT partition quarter-turn orbits by color.
    # Cross-color partners, an axis singleton, blank margins and excess are all
    # explicit. Expected coordinates are specified directly, not via solve.
    source = np.zeros((7, 7), np.int8)
    for p in ((4,1),(5,1),(0,3)):
        source[p] = 2
    for p in ((1,4),(1,5),(3,3)):
        source[p] = 5
    expected = np.zeros((7,7), np.int8)
    expected[1,1:3] = 2
    expected[1,4:6] = 5
    expected[3,3] = 5
    for length in (1,5):
        inp = np.zeros((13,13), np.int8)
        inp[1:8,1:8] = source
        for p in ((0,0),(0,8),(8,0),(8,8)):
            inp[p] = 9
        inp[11,:length] = 2
        assert np.array_equal(solve(inp), expected), "non-orbit input or modulo counter failed"
    # Eight turns is a full cycle, allowing all but the two axis cells to vanish.
    inp[11,:] = 0
    inp[11,:8] = 2
    cycle_expected = np.zeros((7,7), np.int8)
    cycle_expected[0,3], cycle_expected[3,3] = 2, 5
    assert np.array_equal(solve(inp), cycle_expected)
    inp[11,:] = 0
    inp[11,:2] = 2
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("commanded collision was not rejected")
    # Remove supported evidence, retain unsupported evidence, and alter a command:
    # each mutation must invalidate the labeled puzzle.
    puzzle = generate_puzzle(123, 2, 1)
    pair = puzzle["train"][0]
    info = _locate(pair["input"])
    _, aligned = _prediction(info)
    bg = info["background"]
    for kind in ("erase_partner", "keep_outlier", "wrong_counter"):
        bad = copy.deepcopy(puzzle)
        target = bad["train"][0]
        if kind == "erase_partner":
            p = tuple(np.argwhere(target["output"] != bg)[0])
            target["output"][p] = bg
        elif kind == "keep_outlier":
            p = tuple(np.argwhere((aligned != bg) & (target["output"] == bg))[0])
            target["output"][p] = aligned[p]
        else:
            p = next(iter(info["bars"].values()))["cells"][0]
            target["input"][p] = bg
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"mutation accepted: {kind}")
    # Count boundaries and type errors are exercised without expensive sampling.
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "input_shapes": len({r["shape"] for r in records}),
            "sizes": sorted({r["size"] for r in records}),
            "layers": sorted({r["layers"] for r in records}),
            "noise_range": [min(r["noise"] for r in records), max(r["noise"] for r in records)],
            "semantic_verification": "generator-owned checks only"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
