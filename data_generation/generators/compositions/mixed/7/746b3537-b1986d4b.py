"""Rank-nested square inventories followed by consecutive-vector compression.

Only standard-library modules and NumPy are used. Generation has bounded retries,
keeps one puzzle at a time, and writes no files. See task_spec.md for the rule.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _digest(a):
    a = np.asarray(a, dtype=np.uint8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _components(a, background):
    """Same-color four-connected components; differently colored contact is safe."""
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            color = int(a[r, c])
            if seen[r, c] or color == background:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and a[yy, xx] == color:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            ys, xs = zip(*cells)
            y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
            hh, ww = y1 - y0 + 1, x1 - x0 + 1
            result.append((color, hh, ww, len(cells) == hh * ww,
                           (y0, x0, y1, x1), len(cells)))
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    hist = np.bincount(a.ravel(), minlength=10)
    background = int(np.argmax(hist))
    _require(hist[background] > a.size // 2, "background must occupy more than half the input")
    components = _components(a, background)
    classes, boxes = {}, []
    for color, hh, ww, solid, box, _ in components:
        if solid and hh == ww and hh >= 2:
            _require(2 <= hh <= 8, "square side outside family")
            if hh in classes:
                _require(classes[hh][0] == color, "same size has different colors")
                classes[hh][1] += 1
            else:
                classes[hh] = [color, 1]
            boxes.append((hh, color, box))
    _require(2 <= len(classes) <= 5, "need two to five size classes")
    colors = {color for color, count in classes.values()}
    _require(len(colors) == len(classes), "different size classes must have different colors")
    _require(all(1 <= count <= 6 for color, count in classes.values()), "class count outside family")
    remaining = set(int(x) for x in np.unique(a)) - colors - {background}
    _require(len(remaining) <= 1, "at most one distractor color")
    noise = next(iter(remaining), None)
    fragments = 0
    for color, hh, ww, solid, box, _ in components:
        if color in colors and not (solid and hh == ww and hh >= 2):
            side = next(s for s, (cc, _) in classes.items() if cc == color)
            y0, x0, y1, x1 = box
            edge = y0 == 0 or x0 == 0 or y1 == a.shape[0] - 1 or x1 == a.shape[1] - 1
            _require(edge and solid and hh != ww and max(hh, ww) == side
                     and min(hh, ww) < side, "malformed payload fragment")
            fragments += 1
    if noise is not None:
        windows = np.lib.stride_tricks.sliding_window_view(a == noise, (2, 2))
        _require(not windows.all(axis=(-1, -2)).any(), "distractor contains a filled 2x2")
    sides = tuple(sorted(classes))
    width = sum(n for _, n in classes.values()) + max(n for _, n in classes.values())
    _require(width <= 30, "compressed output too wide")
    return {"background": background, "noise": noise, "classes": classes,
            "sides": sides, "boxes": boxes, "fragments": fragments,
            "edge_squares": sum(y0 == 0 or x0 == 0 or y1 == a.shape[0] - 1 or x1 == a.shape[1] - 1
                                for _, _, (y0, x0, y1, x1) in boxes)}


def _rank_strip(facts):
    """B's construction at actual metric sizes, before A's operation."""
    classes = facts["classes"]
    maximum_count = max(n for _, n in classes.values())
    active_sets = [sorted((s for s, (_, n) in classes.items() if n >= r), reverse=True)
                   for r in range(1, maximum_count + 1)]
    width = sum(active[0] + 1 for active in active_sets)
    strip = np.full((max(classes) + 1, width), facts["background"], dtype=np.int8)
    cursor = 0
    for active in active_sets:
        for side in active:
            strip[:side, cursor:cursor + side] = classes[side][0]
        cursor += active[0] + 1
    return strip


def _collapse_vectors(a):
    """Keep one representative per maximal run of equal WHOLE vectors.

    Compute both masks on the original strip, preserving every crossing value.
    Removing duplicate rows cannot change column equality (and conversely), so
    row-first, column-first, and simultaneous collapse give the same result.
    """
    keep_rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    keep_cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return a[np.ix_(keep_rows, keep_cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles and counts from this input alone; never read generator state."""
    return _collapse_vectors(_rank_strip(_analyze(input_grid)))


def _ordinal_invariant(facts):
    """Algebraic crossing check for the composed output, without a metric strip."""
    sides = facts["sides"]
    data = [facts["classes"][s] for s in sides]
    count = max(n for _, n in data)
    out = np.full((len(sides) + 1, sum(n for _, n in data) + count),
                  facts["background"], dtype=np.int8)
    cursor = 0
    for rank in range(1, count + 1):
        active = [i for i, (_, n) in enumerate(data) if n >= rank]
        for row in range(len(sides)):
            for column, index in enumerate(active):
                eligible = [i for i in active if i >= max(row, index)]
                if eligible:
                    out[row, cursor + column] = data[eligible[0]][0]
        cursor += len(active) + 1
    return out


def _reserve(mask, r, c, hh, ww):
    h, w = mask.shape
    mask[max(0, r - 1):min(h, r + hh + 1), max(0, c - 1):min(w, c + ww + 1)] = True


def _pack(rng, h, w, background, classes, want_edge, want_fragment):
    items = [(s, color) for s, (color, n) in classes.items() for _ in range(n)]
    for attempt in range(16):
        # A larger canvas is a bounded packing fallback, never a changed rule.
        if attempt >= 8:
            h = w = 30
        a = np.full((h, w), background, dtype=np.int8)
        reserved = np.zeros((h, w), dtype=bool)
        order = [items[int(i)] for i in rng.permutation(len(items))]
        order.sort(key=lambda item: -item[0])
        failed = False
        for index, (side, color) in enumerate(order):
            windows = np.lib.stride_tricks.sliding_window_view(reserved, (side, side))
            positions = np.argwhere(~windows.any(axis=(-1, -2)))
            if want_edge and index == 0:
                positions = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0)
                                      | (positions[:, 0] == h - side) | (positions[:, 1] == w - side)]
            if not len(positions):
                failed = True
                break
            r, c = (int(x) for x in positions[int(rng.integers(len(positions)))])
            a[r:r + side, c:c + side] = color
            _reserve(reserved, r, c, side, side)
        if failed:
            continue
        if want_fragment:
            candidates = []
            for side, (color, _) in classes.items():
                for depth in range(1, side):
                    for c in range(w - side + 1):
                        candidates.extend(((0, c, depth, side, color), (h - depth, c, depth, side, color)))
                    for r in range(h - side + 1):
                        candidates.extend(((r, 0, side, depth, color), (r, w - depth, side, depth, color)))
            for ii in rng.permutation(len(candidates)):
                r, c, hh, ww, color = candidates[int(ii)]
                if not reserved[r:r + hh, c:c + ww].any():
                    a[r:r + hh, c:c + ww] = color
                    break
            else:
                continue
        return a
    raise RuntimeError("bounded square packing exhausted")


def _add_noise(a, background, color, rng, count):
    cells = np.argwhere(a == background)
    placed = 0
    for ii in rng.permutation(len(cells)):
        r, c = (int(x) for x in cells[int(ii)])
        a[r, c] = color
        if any(np.all(a[y:y + 2, x:x + 2] == color)
               for y in (r - 1, r) for x in (c - 1, c)
               if 0 <= y < a.shape[0] - 1 and 0 <= x < a.shape[1] - 1):
            a[r, c] = background
        else:
            placed += 1
            if placed == count:
                return
    raise RuntimeError("bounded noise placement exhausted")


def _inventory(rng, split, scaffold):
    if scaffold == "small_tail":
        # Unequal gaps expose metric compression, repeated nests expose recurrence.
        options = [s for s in combinations(range(2, 7), 3) if s[1] - s[0] != s[2] - s[1]]
        sides = options[int(rng.integers(len(options)))]
        counts = (int(rng.integers(3, 5)), 2, 2)
    elif scaffold == "other_tail":
        sides = tuple(sorted(int(x) for x in rng.choice(np.arange(2, 7), 3, replace=False)))
        leader = int(rng.integers(1, 3))
        counts = [int(rng.integers(1, 3)) for _ in sides]
        counts[leader] = int(rng.integers(3, 5))
        counts[0] = 1
    else:
        k = int(rng.choice((2, 3, 4, 5) if split == "test" else (2, 3, 4)))
        if split == "test":
            largest = int(rng.integers(7, 9))
            sides = tuple(sorted(int(x) for x in rng.choice(np.arange(2, largest), k - 1, replace=False))) + (largest,)
        else:
            sides = tuple(sorted(int(x) for x in rng.choice(np.arange(2, 7), k, replace=False)))
        counts = [int(x) for x in rng.integers(1, 7, size=k)]
        # Reduce surplus occurrences, never delete a size class. Large-scale
        # tests still have ample placement space and a strict background majority.
        while sum(s * s * n for s, n in zip(sides, counts)) > 230 or sum(counts) + max(counts) > 30:
            candidates = [i for i, n in enumerate(counts) if n > 1]
            i = int(rng.choice(candidates))
            counts[i] -= 1
    return sides, tuple(counts)


def _make_pair(rng, split, scaffold, palette, stable):
    sides, counts = _inventory(rng, split, scaffold)
    p = palette if stable else [int(x) for x in rng.permutation(10)]
    background, noise = p[:2]
    classes = {s: [p[s], n] for s, n in zip(sides, counts)}
    area = sum(s * s * n for s, n in zip(sides, counts))
    shapes = [(h, w) for h in range(17, 31) for w in range(17, 31) if area <= 0.29 * h * w]
    h, w = shapes[int(rng.integers(len(shapes)))]
    fragment = scaffold == "other_tail" or (scaffold is None and rng.random() < 0.35)
    edge = scaffold == "small_tail" or rng.random() < 0.45
    a = _pack(rng, h, w, background, classes, edge, fragment)
    # Noise can touch payload and cluster, but never becomes a competing square.
    if scaffold is not None or rng.random() < 0.8:
        room = int(np.count_nonzero(a == background)) - (a.size // 2 + 1)
        requested = min(room, max(3, int(a.size * rng.uniform(0.015, 0.075))))
        if requested > 0:
            _add_noise(a, background, noise, rng, requested)
    facts = _analyze(a)
    _require(facts["classes"] == classes, "placed inventory disagrees with visible inventory")
    return {"input": a, "output": solve(a)}


def _count_arg(name, value, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count_arg("num_train", num_train, 2, 10)
    num_test = _count_arg("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    stable = bool(rng.random() < 0.25)
    scaffolds = ["small_tail", "other_tail"] + [None] * (num_train - 2)
    rng.shuffle(scaffolds)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            scaffold = scaffolds[index] if split == "train" else None
            for _ in range(24):
                try:
                    pair = _make_pair(rng, split, scaffold, palette, stable)
                except RuntimeError:
                    continue
                key = _digest(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded pair generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pair lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _digest(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts = _analyze(a)
            strip = _rank_strip(facts)
            _require(np.array_equal(out, solve(a)), "incorrect solve output")
            _require(np.array_equal(out, _ordinal_invariant(facts)), "incorrect color crossings")
            _require(out.shape[0] == len(facts["classes"]) + 1, "one global row band per class plus background")
            _require(bool(np.all(out[-1] == facts["background"])) and bool(np.all(out[:, -1] == facts["background"])), "trailing background")
            _require(not np.any(np.all(out[1:] == out[:-1], axis=1)), "uncollapsed rows")
            _require(not np.any(np.all(out[:, 1:] == out[:, :-1], axis=0)), "uncollapsed columns")
            _require(strip.shape[0] > out.shape[0] and strip.shape[1] > out.shape[1], "both compression axes must contribute")
            _require(set(np.unique(out)) == {facts["background"]} | {cc for cc, _ in facts["classes"].values()}, "payload loss or noise leak")
            if split == "train":
                training.append(facts)
    profiles = [tuple(f["classes"][s][1] for s in f["sides"]) for f in training]
    _require(any(len(p) >= 3 and p[0] > max(p[1:]) and min(p) >= 2 for p in profiles), "missing repeated nests and small tail")
    _require(any(len(p) >= 3 and max(p[1:]) > p[0] and 1 in p for p in profiles), "missing different exhaustion order")
    _require(any(f["edge_squares"] for f in training), "edge-square demonstration")
    _require(any(f["fragments"] for f in training), "ignored-fragment demonstration")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                h.update(_digest(pair[role]).encode("ascii"))
    return h.hexdigest()


def self_test():
    """Bounded analytic, mutation, count-contract and 42-seed diversity checks."""
    fixture = {"background": 0, "sides": (2, 3, 5), "classes": {2: [1, 3], 3: [2, 1], 5: [3, 2]}}
    expected = np.array([[1, 2, 3, 0, 1, 3, 0, 1, 0],
                         [2, 2, 3, 0, 3, 3, 0, 0, 0],
                         [3, 3, 3, 0, 3, 3, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    strip = _rank_strip(fixture)
    _require(np.array_equal(_collapse_vectors(strip), expected), "hand-computed rank example")
    _require(np.array_equal(_ordinal_invariant(fixture), expected), "hand-computed crossings")
    transpose_order = _collapse_vectors(strip.T).T
    _require(np.array_equal(transpose_order, expected), "axis order invariance")
    # Nonadjacent identical columns and separators must recur.
    _require(np.array_equal(expected[:, 2], expected[:, 5]), "fixture recurrence")
    _require(not np.array_equal(expected[:, 0], expected[:, 4]), "equal top cells do not imply equal columns")
    crossed = np.array([[1, 1, 2, 1, 1], [1, 1, 2, 1, 1],
                        [3, 3, 4, 3, 3], [1, 1, 2, 1, 1]])
    _require(np.array_equal(_collapse_vectors(crossed), [[1, 2, 1], [3, 4, 3], [1, 2, 1]]), "separated recurrence or crossing destroyed")
    wrong_counts = {**fixture, "classes": {s: [cc, 3] for s, (cc, _) in fixture["classes"].items()}}
    _require(not np.array_equal(_collapse_vectors(_rank_strip(wrong_counts)), expected), "rank threshold ablation")
    _require(strip.shape != expected.shape, "compression ablation")
    # Input-only boundary case with a complete edge square and a nonsquare crop.
    edge = np.zeros((18, 20), dtype=np.int8)
    edge[:2, :2] = 1
    edge[5:7, 4:6] = 1
    edge[8:11, 10:13] = 2
    edge[-1:, 16:19] = 2
    edge[3, 5:8] = 9
    ef = _analyze(edge)
    _require(ef["classes"] == {2: [1, 2], 3: [2, 1]} and ef["fragments"] == 1, "edge and fragment count")
    _require(np.array_equal(solve(edge), [[1, 2, 0, 1, 0], [2, 2, 0, 0, 0], [0, 0, 0, 0, 0]]), "input-only hand example")
    _require(_puzzle_digest(generate_puzzle(991, 5, 2)) == _puzzle_digest(generate_puzzle(991, 5, 2)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(700 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    signatures, shapes, inventories, palettes, output_shapes = (set() for _ in range(5))
    class_counts, leaders, backgrounds = set(), set(), set()
    fragments = clean = edges = ties = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                facts = _analyze(pair["input"])
                sides = facts["sides"]
                counts = tuple(facts["classes"][s][1] for s in sides)
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                inventories.add((sides, counts))
                palettes.add(tuple(facts["classes"][s][0] for s in sides))
                backgrounds.add(facts["background"])
                class_counts.add(len(sides))
                leaders.update(i for i, n in enumerate(counts) if n == max(counts))
                fragments += facts["fragments"] > 0
                clean += facts["fragments"] == 0
                edges += facts["edge_squares"] > 0
                ties += counts.count(max(counts)) > 1
                # Omit A / omit thresholding B: both change every uneven case.
                strip = _rank_strip(facts)
                _require(strip.shape != pair["output"].shape, "A unnecessary")
                if len(set(counts)) > 1:
                    equal = {**facts, "classes": {s: [cc, max(counts)] for s, (cc, _) in facts["classes"].items()}}
                    _require(not np.array_equal(pair["output"], _collapse_vectors(_rank_strip(equal))), "B threshold unnecessary")
                if split == "test":
                    _require(max(sides) >= 7, "missing scale shift")
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 35 and len(inventories) >= 90 and len(output_shapes) >= 20, "structural diversity")
    _require(class_counts == {2, 3, 4, 5} and len(leaders) >= 4, "class or leader diversity")
    _require(len(palettes) >= 70 and backgrounds == set(range(10)), "palette roles")
    _require(min(fragments, clean, edges, ties) >= 15, "boundary, optional-fragment or tie coverage")
