"""Torn blueprints: a magnified seam witness resolves ambiguous overlaps.

Only NumPy and the Python standard library are used. No parent module, stored
example, seed, or generation metadata is consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _domino(a, bg):
    """A registration witness is an adjacent pair of unequal foreground colors."""
    return bool(np.any((a[1:] != a[:-1]) & (a[1:] != bg) & (a[:-1] != bg))
                or np.any((a[:, 1:] != a[:, :-1]) & (a[:, 1:] != bg) & (a[:, :-1] != bg)))


def _intersection(a, pa, b, pb):
    r0, c0 = max(pa[0], pb[0]), max(pa[1], pb[1])
    r1 = min(pa[0] + a.shape[0], pb[0] + b.shape[0])
    c1 = min(pa[1] + a.shape[1], pb[1] + b.shape[1])
    if r0 >= r1 or c0 >= c1:
        return None
    return (a[r0-pa[0]:r1-pa[0], c0-pa[1]:c1-pa[1]],
            b[r0-pb[0]:r1-pb[0], c0-pb[1]:c1-pb[1]])


def _translations(a, b, bg):
    """All supported offsets of b relative to a; never maximize overlap."""
    found = []
    for dr in range(1 - b.shape[0], a.shape[0]):
        for dc in range(1 - b.shape[1], a.shape[1]):
            x, y = _intersection(a, (0, 0), b, (dr, dc))
            if np.array_equal(x, y) and _domino(x, bg):
                found.append((dr, dc))
    return found


def _merge(parts, offsets, bg):
    top = min(p[0] for p in offsets)
    left = min(p[1] for p in offsets)
    bottom = max(p[0] + a.shape[0] for a, p in zip(parts, offsets))
    right = max(p[1] + a.shape[1] for a, p in zip(parts, offsets))
    plan = np.full((bottom - top, right - left), bg, dtype=np.int8)
    normalized = [(p[0] - top, p[1] - left) for p in offsets]
    for a, (r, c) in zip(parts, normalized):
        plan[r:r+a.shape[0], c:c+a.shape[1]] = a
    return plan, normalized


def _assemblies(parts, bg, max_shape=(10, 10)):
    """Enumerate connected, conflict-free translated mosaics modulo translation.

    A potential match is optional: unrelated occurrences need not coincide.
    Every actual rectangular intersection, including background cells, agrees.
    Enumeration is finite (at most three patches and a 10-by-10 logical span),
    with no greedy score or truncation of valid candidate answers.
    """
    n = len(parts)
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            ds = _translations(parts[i], parts[j], bg)
            edges[i, j] = ds
            edges[j, i] = [(-r, -c) for r, c in ds]
    offsets = {0: (0, 0)}
    visited, results = set(), {}

    def search():
        state = tuple(offsets.get(i) for i in range(n))
        if state in visited:
            return
        visited.add(state)
        if len(offsets) == n:
            plan, normalized = _merge(parts, [offsets[i] for i in range(n)], bg)
            results.setdefault(_key(plan), (plan, normalized))
            return
        for j in range(n):
            if j in offsets:
                continue
            candidates = set()
            for i, (r, c) in offsets.items():
                candidates.update((r+dr, c+dc) for dr, dc in edges[i, j])
            for position in sorted(candidates):
                compatible = True
                for i, ip in offsets.items():
                    overlap = _intersection(parts[i], ip, parts[j], position)
                    if overlap is not None and not np.array_equal(*overlap):
                        compatible = False
                        break
                if not compatible:
                    continue
                offsets[j] = position
                span = tuple(max(offsets[i][d] + parts[i].shape[d] for i in offsets)
                             - min(offsets[i][d] for i in offsets) for d in (0, 1))
                if all(span[d] <= max_shape[d] for d in (0, 1)):
                    search()
                del offsets[j]

    search()
    return list(results.values())


def _parse(inp):
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    _require(int(counts[bg]) * 2 > inp.size, "background must occupy a majority")
    parts, boxes, squares = [], [], []
    for cells in _components(inp != bg):
        r0 = min(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        r1 = max(p[0] for p in cells) + 1
        c1 = max(p[1] for p in cells) + 1
        a = inp[r0:r1, c0:c1].copy()
        colors = set(int(inp[p]) for p in cells)
        if len(colors) == 1:
            _require(a.shape[0] == a.shape[1] and 2 <= a.shape[0] <= 4
                     and len(cells) == a.size, "invalid enlarged seed square")
            squares.append((r0, c0, a.shape[0], next(iter(colors))))
        else:
            _require(all(2 <= d <= 6 for d in a.shape), "invalid compact patch")
            _require(_domino(a, bg), "patch lacks a colored registration feature")
            parts.append(a)
            boxes.append((r0, c0, r1, c1))
    _require(2 <= len(parts) <= 3, "need two or three compact patches")
    _require(3 <= len(squares) <= 7, "need three to seven seed squares")
    if len({s[0] for s in squares}) == 1:
        axis = 1
    else:
        _require(len({s[1] for s in squares}) == 1, "seed must be a straight run")
        axis = 0
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    pitch = squares[1][axis] - squares[0][axis]
    _require(all(s[2] == side for s in squares), "unequal square sizes")
    _require(1 <= pitch - side <= 3, "gutter must be one to three cells")
    _require(all(squares[k][axis] == squares[0][axis] + k*pitch
                 for k in range(len(squares))), "seed is not consecutive")
    return bg, parts, boxes, squares, axis, side, pitch


def _matches(plan, sequence, axis):
    length = len(sequence)
    for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
            if all(int(plan[r + k*(axis == 0), c + k*(axis == 1)]) == color
                   for k, color in enumerate(sequence)):
                yield r, c


def _overlap_boxes(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _render(inp, plan, base, side, pitch, bg):
    out = inp.copy()
    for (r, c), color in np.ndenumerate(plan):
        if color != bg:
            y, x = base[0] + r*pitch, base[1] + c*pitch
            out[y:y+side, x:x+side] = color
    return out


def _recover(input_grid):
    inp = _grid(input_grid)
    bg, parts, boxes, squares, axis, side, pitch = _parse(inp)
    sequence = tuple(s[3] for s in squares)
    max_shape = tuple(min(10, (d - side)//pitch + 1) for d in inp.shape)
    assemblies = _assemblies(parts, bg, max_shape)
    answers = {}
    for plan, offsets in assemblies:
        for match in _matches(plan, sequence, axis):
            base = (squares[0][0] - match[0]*pitch, squares[0][1] - match[1]*pitch)
            end = tuple(base[d] + (plan.shape[d]-1)*pitch + side for d in (0, 1))
            if any(base[d] < 0 or end[d] > inp.shape[d] for d in (0, 1)):
                continue
            box = (*base, *end)
            if any(_overlap_boxes(box, b) for b in boxes):
                continue
            # The entire future rectangle is empty except for the given run.
            region = inp[base[0]:end[0], base[1]:end[1]].copy()
            for s in squares:
                rr, cc = s[0] - base[0], s[1] - base[1]
                region[rr:rr+side, cc:cc+side] = bg
            if np.any(region != bg):
                continue
            expected = _render(inp, plan, base, side, pitch, bg)
            answers.setdefault(_key(expected), (expected, plan, offsets, match, base))
    _require(len(answers) == 1, "no unique joint overlap-and-sequence completion")
    expected, plan, offsets, match, base = next(iter(answers.values()))
    same_shape = sum(a.shape == plan.shape for a, _ in assemblies)
    seed_sites = {(match[0] + k*(axis == 0), match[1] + k*(axis == 1))
                  for k in range(len(sequence))}
    added = [int(v) for p, v in np.ndenumerate(plan) if p not in seed_sites and v != bg]
    footprints = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            overlap = _intersection(parts[i], offsets[i], parts[j], offsets[j])
            if overlap is not None and _domino(overlap[0], bg):
                footprints.append(overlap[0].shape)
    meta = {"background": bg, "patches": len(parts), "plan_shape": plan.shape,
            "axis": axis, "side": side, "gutter": pitch-side, "length": len(sequence),
            "match": match, "base": base, "holes": int(np.count_nonzero(plan == bg)),
            "assemblies": len(assemblies), "same_shape_assemblies": same_shape,
            "single_patch_matches": sum(len(list(_matches(p, sequence, axis))) for p in parts),
            "plan_cells": int(np.count_nonzero(plan != bg)),
            "largest_patch_cells": max(int(np.count_nonzero(p != bg)) for p in parts),
            "added_colors": tuple(sorted(set(added))), "overlap_shapes": tuple(footprints),
            "palette_size": len(set(int(v) for v in inp.flat)) - 1,
            "interior_line": 0 < match[1-axis] < plan.shape[1-axis]-1}
    return expected, meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete enlarged mosaic from this input alone."""
    return _recover(input_grid)[0]


def _patch_ok(a, bg):
    nonzero = np.argwhere(a != bg)
    return (len(nonzero) > 0 and len(_components(a != bg)) == 1
            and np.all(nonzero.min(axis=0) == 0)
            and np.all(nonzero.max(axis=0) == np.array(a.shape)-1)
            and _domino(a, bg))


def _split_patch(parts, offsets, rng, bg):
    options = []
    for i, a in enumerate(parts):
        for axis in (0, 1):
            for cut in range(1, a.shape[axis]-1):
                for width in (1, 2):
                    if cut + width >= a.shape[axis]:
                        continue
                    x = a[:cut+width, :] if axis == 0 else a[:, :cut+width]
                    y = a[cut:, :] if axis == 0 else a[:, cut:]
                    delta = (cut, 0) if axis == 0 else (0, cut)
                    overlap = _intersection(x, (0, 0), y, delta)
                    if min(*x.shape, *y.shape) >= 2 and _patch_ok(x, bg) and _patch_ok(y, bg) and _domino(overlap[0], bg):
                        options.append((i, x.copy(), y.copy(), delta))
    if not options:
        return None
    i, a, b, delta = options[int(rng.integers(len(options)))]
    origin = offsets[i]
    return (parts[:i] + [a, b] + parts[i+1:],
            offsets[:i] + [origin, (origin[0]+delta[0], origin[1]+delta[1])] + offsets[i+1:])


def _blueprint(rng, bg, palette, axis, three, test):
    """Construct repeated registration strips; select a unique seam witness."""
    for _ in range(180):
        height = int(rng.integers(5 if test else 4, 7 if test else 6))
        width = int(rng.integers(2, 5 if test else 4))
        wh = int(rng.integers(2, min(3, height//2) + 1))
        ww = int(rng.integers(2, 5 if test else 4))
        overlap = int(rng.integers(1, min(2, width-1, ww-1) + 1))
        main = rng.choice(palette, (height, width)).astype(np.int8)
        wing = rng.choice(palette, (wh, ww)).astype(np.int8)
        # Holes are observed empty sites, not transparent cells during joining.
        main[rng.random(main.shape) < .07] = bg
        wing[rng.random(wing.shape) < .07] = bg
        key = rng.choice(palette, (wh, overlap)).astype(np.int8)
        if not _domino(key, bg):
            continue
        starts = [(a, b) for a in range(height-wh+1) for b in range(a+wh, height-wh+1)]
        a, b = starts[int(rng.integers(len(starts)))]
        selected = int(rng.choice([a, b]))
        main[a:a+wh, -overlap:] = key
        main[b:b+wh, -overlap:] = key
        wing[:, :overlap] = key
        parts = [main, wing]
        offsets = [(0, 0), (selected, width-overlap)]
        if not all(_patch_ok(p, bg) for p in parts):
            continue
        if three:
            split = _split_patch(parts, offsets, rng, bg)
            if split is None:
                continue
            parts, offsets = split
        if axis == 0:
            parts = [p.T.copy() for p in parts]
            offsets = [(c, r) for r, c in offsets]
        intended, _ = _merge(parts, offsets, bg)
        assemblies = _assemblies(parts, bg)
        if sum(p.shape == intended.shape for p, _ in assemblies) < 2:
            continue
        intended_key = _key(intended)
        # Find literal sequences that occur in exactly one complete candidate
        # mosaic and nowhere within any individual compact patch.
        occurrences = {}
        min_length = overlap + 2
        max_length = min(7, intended.shape[axis])
        for length in range(min_length, max_length+1):
            if test and length == min_length and max_length > min_length:
                continue
            for plan, _ in assemblies:
                for r in range(plan.shape[0] - (length-1 if axis == 0 else 0)):
                    for c in range(plan.shape[1] - (length-1 if axis == 1 else 0)):
                        seq = tuple(int(plan[r+k*(axis == 0), c+k*(axis == 1)]) for k in range(length))
                        if bg not in seq:
                            occurrences.setdefault(seq, []).append((_key(plan), (r, c)))
        choices = []
        for seq, hits in occurrences.items():
            if len(hits) != 1 or hits[0][0] != intended_key:
                continue
            match = hits[0][1]
            # The witness actually bridges main-only and wing-only material.
            canonical_match = match if axis == 1 else (match[1], match[0])
            rr, cc = canonical_match
            if not (selected <= rr < selected+wh and cc < width-overlap
                    and cc+len(seq) > width):
                continue
            if test and not (0 < match[1-axis] < intended.shape[1-axis]-1):
                continue
            if any(list(_matches(p, seq, axis)) for p in parts):
                continue
            if np.count_nonzero(intended != bg) <= max(np.count_nonzero(p != bg) for p in parts):
                continue
            added = {int(v) for p, v in np.ndenumerate(intended) if v != bg and
                     p not in {(match[0]+k*(axis == 0), match[1]+k*(axis == 1)) for k in range(len(seq))}}
            if len(added) >= 2:
                choices.append((seq, match))
        if choices:
            seq, match = choices[int(rng.integers(len(choices)))]
            return parts, intended, seq, match
    raise RuntimeError("bounded blueprint sampling exhausted")


def _layout_options(plan, parts, avoid_side=None):
    options = []
    for side in range(2, 5):
        if side == avoid_side:
            continue
        for gutter in range(1, 4):
            pitch = side + gutter
            big = tuple((d-1)*pitch + side for d in plan.shape)
            for axis in (0, 1):
                other = 1-axis
                required = list(big)
                required[axis] += max(p.shape[axis] for p in parts) + 1
                required[other] = max(required[other], sum(p.shape[other] for p in parts) + len(parts)-1)
                if max(required) <= 30:
                    options.append((side, pitch, axis, tuple(required)))
    return options


def _pack(rng, parts, plan, bg, avoid_side=None):
    options = _layout_options(plan, parts, avoid_side)
    _require(bool(options), "no feasible enlarged geometry")
    side, pitch, axis, required = options[int(rng.integers(len(options)))]
    shape = tuple(int(rng.integers(d, min(30, d+6)+1)) for d in required)
    big = tuple((d-1)*pitch + side for d in plan.shape)
    order = sorted(range(len(parts)), key=lambda i: -parts[i].size)
    # Try scattering patches around any sides of the future completion.
    for _ in range(12):
        base = tuple(int(rng.integers(shape[d]-big[d]+1)) for d in (0, 1))
        occupied = np.zeros(shape, dtype=bool)
        r, c = base
        occupied[max(0, r-1):min(shape[0], r+big[0]+1), max(0, c-1):min(shape[1], c+big[1]+1)] = True
        placements = {}
        for i in order:
            ph, pw = parts[i].shape
            feasible = [(r, c) for r in range(shape[0]-ph+1) for c in range(shape[1]-pw+1)
                        if not np.any(occupied[r:r+ph, c:c+pw])]
            if not feasible:
                break
            r, c = feasible[int(rng.integers(len(feasible)))]
            placements[i] = (r, c)
            occupied[max(0, r-1):min(shape[0], r+ph+1), max(0, c-1):min(shape[1], c+pw+1)] = True
        if len(placements) == len(parts):
            return shape, base, placements, side, pitch
    # Guaranteed strip fallback uses exactly the same rule and sampled shapes.
    other = 1-axis
    base = [0, 0]
    placements = {}
    strip_size = max(p.shape[axis] for p in parts)
    before = bool(rng.integers(2))
    if before:
        base[axis] = strip_size+1
        strip = 0
    else:
        strip = big[axis]+1
    base[other] = int(rng.integers(shape[other]-big[other]+1))
    cursor = 0
    for i in rng.permutation(len(parts)):
        i = int(i)
        p = [0, 0]
        p[axis] = strip + int(rng.integers(strip_size-parts[i].shape[axis]+1))
        p[other] = cursor
        placements[i] = tuple(p)
        cursor += parts[i].shape[other]+1
    return shape, tuple(base), placements, side, pitch


def _make_pair(rng, axis, three, test, avoid_side=None):
    palette = rng.permutation(10)
    bg = int(palette[0])
    colors = palette[1:int(rng.integers(4, 8))]
    parts, plan, seq, match = _blueprint(rng, bg, colors, axis, three, test)
    # Sample the orientation of the entire latent blueprint. All observations
    # change together; solving still translates each input patch unchanged.
    for dimension in (0, 1):
        if rng.integers(2):
            parts = [np.flip(p, dimension).copy() for p in parts]
            plan = np.flip(plan, dimension).copy()
            match = list(match)
            match[dimension] = plan.shape[dimension] - match[dimension] - (len(seq) if dimension == axis else 1)
            match = tuple(match)
            if dimension == axis:
                seq = seq[::-1]
    shape, base, placements, side, pitch = _pack(rng, parts, plan, bg, avoid_side)
    inp = np.full(shape, bg, dtype=np.int8)
    for i, a in enumerate(parts):
        r, c = placements[i]
        inp[r:r+a.shape[0], c:c+a.shape[1]] = a
    for k, color in enumerate(seq):
        r = base[0] + (match[0]+k*(axis == 0))*pitch
        c = base[1] + (match[1]+k*(axis == 1))*pitch
        inp[r:r+side, c:c+side] = color
    out = _render(inp, plan, base, side, pitch, bg)
    _require(np.array_equal(solve(inp), out), "constructed plan disagrees with input-only solution")
    return {"input": inp, "output": out}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside legal range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    axes = [1] if mode == 0 else [0] if mode == 1 else [0, 1]
    train_axes = [axes[k % len(axes)] for k in range(nt)]
    rng.shuffle(train_axes)
    result, seen = {"train": [], "test": []}, set()
    first_side = None
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            axis = train_axes[index] if split == "train" else int(rng.choice(axes))
            three = index == 1 if split == "train" and index < 2 else bool(rng.random() < .65)
            for _ in range(12):
                pair = _make_pair(rng, axis, three, split == "test",
                                  first_side if split == "train" and index == 1 else None)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct-input sampling exhausted")
            seen.add(key)
            result[split].append(pair)
            if split == "train" and index == 0:
                first_side = _parse(pair["input"])[5]
        rng.shuffle(result[split])
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, m = _recover(inp)
            _require(np.array_equal(out, expected), "incorrect enlarged mosaic or changed preserved pixel")
            _require(m["same_shape_assemblies"] >= 2, "large witness must resolve real assembly ambiguity")
            _require(m["single_patch_matches"] == 0, "witness must bridge a seam")
            _require(m["plan_cells"] > m["largest_patch_cells"], "one compact patch suffices")
            _require(len(m["added_colors"]) >= 2, "need informative literal-color additions")
            _require(not np.array_equal(inp, out), "identity example")
            metas[split].append(m)
    _require({m["axis"] for m in metas["test"]} <= {m["axis"] for m in metas["train"]}, "untaught witness axis")
    _require({m["patches"] for m in metas["train"]} == {2, 3}, "demonstrate both patch counts")
    _require(len({m["side"] for m in metas["train"]}) >= 2, "demonstrate scale inference")


def self_test():
    """Bounded 42-seed semantic, diversity, equivariance, and corruption checks.

    Puzzles are processed one at a time. Only hashes, counters and small sets
    survive each iteration; no trace or full corpus is written to disk.
    """
    features = {s: {k: set() for k in ("plan_shape", "axis", "side", "gutter", "length",
                "patches", "background", "holes", "match", "palette_size", "overlap_shapes")}
                for s in ("train", "test")}
    hashes, shapes = set(), set()
    ambiguous = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(repr(inp.shape).encode() + inp.tobytes() + out.tobytes())
                expected, m = _recover(inp)
                assert np.array_equal(expected, out)
                for k in features[split]:
                    features[split][k].add(m[k])
                shapes.add(inp.shape)
                ambiguous[split] += int(m["same_shape_assemblies"] >= 2)
        hashes.add(digest.hexdigest())
        if seed < 3:
            pair = puzzle["test"][0]
            assert np.array_equal(solve(pair["input"].T), pair["output"].T)
            mapping = np.random.default_rng(1000+seed).permutation(10)
            assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
    assert len(hashes) == 42 and len(shapes) >= 25
    for split in ("train", "test"):
        f = features[split]
        assert f["axis"] == {0, 1} and f["patches"] == {2, 3}
        assert len(f["plan_shape"]) >= 8 and len(f["match"]) >= 8
        assert len(f["background"]) >= 8 and len(f["length"]) >= 2
        assert len(f["side"]) >= 2 and len(f["gutter"]) >= 2
        assert any(h > 0 for h in f["holes"])
        assert len(f["overlap_shapes"]) >= 6
    # Small, explicitly reasoned fixture: the key column 1/2 occurs twice.
    # Both joins overlap two cells and yield a 4-by-3 mosaic, so neither area
    # nor overlap maximization distinguishes them. Only 4,1,8 crosses the
    # correct seam; 6,1,8 crosses the other candidate's seam.
    main = np.array([[4, 1], [5, 2], [6, 1], [7, 2]], dtype=np.int8)
    wing = np.array([[1, 8], [2, 9]], dtype=np.int8)
    fixture = np.zeros((25, 21), dtype=np.int8)
    fixture[1:5, 1:3] = main
    fixture[8:10, 1:3] = wing
    for k, color in enumerate((4, 1, 8)):
        fixture[8:10, 8+3*k:10+3*k] = color
    literal_plan = np.array([[4, 1, 8], [5, 2, 9], [6, 1, 0], [7, 2, 0]], dtype=np.int8)
    literal_expected = fixture.copy()
    for (r, c), color in np.ndenumerate(literal_plan):
        if color:
            literal_expected[8+3*r:10+3*r, 8+3*c:10+3*c] = color
    predicted, fm = _recover(fixture)
    assert np.array_equal(predicted, literal_expected)
    assert fm["same_shape_assemblies"] == 2 and fm["single_patch_matches"] == 0
    damaged_key = fixture.copy()
    damaged_key[9, 1] = 3
    ambiguous_input = fixture.copy()
    ambiguous_input[3, 1] = 4  # now 4,1,8 occurs in both legal mosaics
    for bad_input in (damaged_key, ambiguous_input):
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent or ambiguous input accepted")
    # Empty observed cells cannot be treated as transparent paint.
    x = np.array([[1, 2], [3, 0]], dtype=np.int8)
    y = np.array([[1, 2], [3, 4]], dtype=np.int8)
    assert (0, 0) not in _translations(x, y, 0)
    p = generate_puzzle(234, 2, 1)
    q = generate_puzzle(234, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "input_shapes": len(shapes),
            "pairs_with_assembly_ambiguity": dict(ambiguous),
            "feature_cardinalities": {s: {k: len(v) for k, v in f.items()} for s, f in features.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
