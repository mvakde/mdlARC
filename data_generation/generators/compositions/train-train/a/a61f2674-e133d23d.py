"""Reconstruct two fragmented scans of comb trees and prune to extremal paths.

Only NumPy and the standard library are used.  Outputs are constructed from
sampled geometry; solve independently recovers the geometry from input cells.
"""

from collections import deque

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes() if a.dtype == np.int8 else a.astype(np.int8).tobytes()


def _split_input(value):
    """Find the sole full separator whose color occurs nowhere else."""
    a = _grid(value)
    candidates = []
    for axis in (1, 0):
        if a.shape[axis] % 2 != 1 or a.shape[axis] < 3:
            continue
        mid = a.shape[axis] // 2
        if axis == 1:
            first, line, second = a[:, :mid], a[:, mid], a[:, mid + 1:]
        else:
            first, line, second = a[:mid], a[mid], a[mid + 1:]
        divider = int(line.flat[0])
        if not np.all(line == divider):
            continue
        f, s = set(map(int, np.unique(first))), set(map(int, np.unique(second)))
        common = f & s
        if len(f) != 2 or len(s) != 2 or len(common) != 1 or divider in f | s:
            continue
        bg = next(iter(common))
        c1, c2 = next(iter(f - {bg})), next(iter(s - {bg}))
        candidates.append(dict(first=first != bg, second=second != bg,
                               background=bg, divider=divider,
                               long_color=c1, short_color=c2, axis=axis))
    if len(candidates) != 1:
        raise ValueError("input must have two unambiguously separated binary scans")
    return candidates[0]


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue, cells = deque([start]), [start]
        while queue:
            r, c = queue.popleft()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
                    cells.append(p)
        result.append(cells)
    return result


def _parse_tree(cells):
    """Recover a capped spine and its separated one-sided straight teeth.

    Individual teeth can point to either side of the spine.  The two caps,
    at least three separated teeth, and unique extrema make the axis unique.
    """
    candidates = []
    for transpose in (False, True):
        points = {(int(c), int(r)) if transpose else (int(r), int(c))
                  for r, c in cells}
        r0, r1 = min(r for r, c in points), max(r for r, c in points)
        for spine in range(min(c for r, c in points), max(c for r, c in points) + 1):
            if not all((r, spine) in points for r in range(r0, r1 + 1)):
                continue
            teeth, valid = [], True
            for r in range(r0, r1 + 1):
                cols = sorted(c for rr, c in points if rr == r and c != spine)
                if not cols:
                    continue
                if cols[-1] < spine and cols == list(range(cols[0], spine)):
                    side = -1
                elif cols[0] > spine and cols == list(range(spine + 1, cols[-1] + 1)):
                    side = 1
                else:
                    valid = False
                    break
                teeth.append((r, len(cols), side))
            if not valid or len(teeth) < 3:
                continue
            roots = [t[0] for t in teeth]
            lengths = [t[1] for t in teeth]
            if (roots[0] == r0 or roots[-1] == r1
                    or any(b - a < 2 for a, b in zip(roots, roots[1:]))
                    or lengths.count(min(lengths)) != 1
                    or lengths.count(max(lengths)) != 1
                    or max(lengths) - min(lengths) < 2):
                continue
            def real(p):
                return (p[1], p[0]) if transpose else p
            arms = [tuple(real((r, spine + side * k)) for k in range(1, length + 1))
                    for r, length, side in teeth]
            lo, hi = int(np.argmin(lengths)), int(np.argmax(lengths))
            a, b = sorted((roots[lo], roots[hi]))
            connector = tuple(real((r, spine)) for r in range(a, b + 1))
            candidates.append(dict(teeth=teeth, arms=arms, shortest=lo, longest=hi,
                                   connector=connector, transpose=transpose,
                                   spine=spine, caps=(r0, r1), cells=tuple(cells)))
    if len(candidates) != 1:
        raise ValueError("each union component must be one unambiguous comb tree")
    return candidates[0]


def _analyze_input(value):
    info = _split_input(value)
    info["union"] = info["first"] | info["second"]
    info["trees"] = [_parse_tree(c) for c in _components(info["union"])]
    if not info["trees"]:
        raise ValueError("at least one comb tree is required")
    return info


def _paint(out, cells, color):
    for r, c in cells:
        out[r, c] = color


def _answer(info):
    out = np.full(info["union"].shape, info["background"], dtype=np.int8)
    for tree in info["trees"]:
        _paint(out, tree["connector"], info["divider"])
        _paint(out, tree["arms"][tree["longest"]], info["long_color"])
        _paint(out, tree["arms"][tree["shortest"]], info["short_color"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution; panel order visibly supplies the output colors."""
    return _answer(_analyze_input(input_grid))


def _sample_tree(rng, height, width, *, orientation=None, bilateral=None,
                 anchor=False, minimum=None, boundary_min=False):
    if orientation is None:
        orientation = bool(rng.integers(2))
    axis, cross = (width, height) if orientation else (height, width)
    max_count = min(6, (axis - 1) // 2)
    count = 4 if anchor else int(rng.integers(3, max_count + 1))
    compressed = np.sort(rng.choice(axis - count - 1, size=count, replace=False))
    roots = compressed + np.arange(count) + 1
    start = int(rng.integers(0, int(roots[0])))
    end = int(rng.integers(int(roots[-1]) + 1, axis))
    if bilateral is None:
        bilateral = bool(rng.random() < .38)
    if bilateral:
        spine = int(rng.integers(3, cross - 3))
        capacity = min(spine, cross - 1 - spine)
        sides = rng.choice((-1, 1), size=count)
        sides[0], sides[-1] = -1, 1
        rng.shuffle(sides)
    else:
        side = int(rng.choice((-1, 1)))
        needed = 4 if minimum == 2 else 3
        offset = int(rng.integers(0, cross - needed))
        spine = offset if side == 1 else cross - 1 - offset
        capacity = cross - 1 - offset
        sides = np.full(count, side)
    capacity = min(capacity, 11)
    low = int(minimum) if minimum is not None else int(rng.integers(1, capacity - 1))
    high = int(rng.integers(low + 2, capacity + 1))
    lengths = rng.integers(low + 1, high, size=count)
    if anchor:
        # An almost-longest tooth, duplicated in both scans, is a deliberate
        # trap for adding source occupancies instead of taking their union.
        lengths[:] = high - 1
        lo, hi = (1, 2) if rng.integers(2) else (2, 1)
    else:
        lo, hi = map(int, rng.choice(count, size=2, replace=False))
        if boundary_min:
            lo = int(rng.choice((0, count - 1)))
            hi = int(rng.choice([i for i in range(count) if i != lo]))
    lengths[lo], lengths[hi] = low, high
    canonical = np.zeros((axis, cross), dtype=bool)
    canonical[start:end + 1, spine] = True
    canonical_parts = np.zeros((axis, cross), dtype=np.int8)
    a, b = sorted((int(roots[lo]), int(roots[hi])))
    canonical_parts[a:b + 1, spine] = 3
    for i, (root, length, side) in enumerate(zip(roots, lengths, sides)):
        cols = spine + int(side) * np.arange(1, int(length) + 1)
        canonical[int(root), cols] = True
        if i in (lo, hi):
            canonical_parts[int(root), cols] = 1 if i == hi else 2
    if orientation:
        return canonical.T.copy(), canonical_parts.T.copy()
    return canonical, canonical_parts


def _scene(rng, groups, anchor=False, second=False):
    width = int(rng.integers(11, 15))
    if groups == 1:
        heights = [int(rng.integers(9, 19))]
    elif groups == 2:
        heights = [int(rng.integers(8, 14)), int(rng.integers(8, 14))]
    else:
        heights = [7, 7, 7]
        # Allocate a variable extra budget without ever exceeding 30 rows.
        for _ in range(int(rng.integers(0, 6))):
            heights[int(rng.integers(groups))] += 1
    gaps = [int(rng.integers(1, 3)) for _ in range(groups - 1)]
    total = sum(heights) + sum(gaps)
    union = np.zeros((total, width), dtype=bool)
    parts = np.zeros((total, width), dtype=np.int8)
    region_paths = []
    offset = 0
    for i, height in enumerate(heights):
        kwargs = {}
        if anchor:
            kwargs = dict(orientation=False, bilateral=False, anchor=True, minimum=1)
        elif second and i == 0:
            kwargs = dict(orientation=True, bilateral=False, minimum=2, boundary_min=True)
        elif second and i == 1:
            kwargs = dict(orientation=False, bilateral=True, minimum=1)
        mask, labels = _sample_tree(rng, height, width, **kwargs)
        union[offset:offset + height] = mask
        parts[offset:offset + height] = labels
        region = np.zeros_like(union)
        region[offset:offset + height] = labels != 0
        region_paths.append(region)
        offset += height + (gaps[i] if i < len(gaps) else 0)
    return union, parts, region_paths


def _scans(rng, union, parts, paths, demo, rank_trap=False):
    mode = int(rng.integers(6)) if not demo else int(rng.integers(3))
    states = np.zeros(union.shape, dtype=np.int8)
    if mode == 0:  # Arbitrary complementary fragments, with overlap.
        probabilities = rng.dirichlet((2., 2., 2.))
        states[union] = rng.choice((1, 2, 3), size=int(union.sum()), p=probabilities)
    elif mode == 1:  # Coherent spatial chunks and an overlap strip.
        rr, cc = np.indices(union.shape)
        coords = rr if rng.integers(2) else cc
        limit = int(rng.integers(1, int(coords.max())))
        labels = np.where(coords < limit, 1, np.where(coords == limit, 3, 2))
        states[union] = labels[union]
    elif mode == 2:  # Repeated stripes, phase and axis varied.
        rr, cc = np.indices(union.shape)
        phase = int(rng.integers(3))
        labels = 1 + ((rr if rng.integers(2) else cc) + phase) % 3
        states[union] = labels[union]
    elif mode == 3:  # One scan contains the other.
        base = int(rng.choice((1, 2)))
        states[union] = rng.choice((base, 3), size=int(union.sum()), p=(.6, .4))
    elif mode == 4:  # No source overlap: OR must still reconnect the tree.
        states[union] = rng.choice((1, 2), size=int(union.sum()))
    else:  # Identical scans, deliberately admissible without extra signals.
        states[union] = 3
    if rank_trap:
        states[union & (parts == 0)] = 3
        states[parts == 1] = rng.choice((1, 2), size=int(np.count_nonzero(parts == 1)))
    if demo:
        for path in paths:
            # Each scan is missing a required spine cell; together they
            # recover a connector containing all three positive OR states.
            indices = rng.choice(np.flatnonzero(path & (parts == 3)), size=3, replace=False)
            states.flat[indices] = (1, 2, 3)
    # Both foreground colors must be visible in every input, even if a
    # coherent chunk happens to fall entirely outside the tree.
    if not np.any(states & 1):
        states.flat[int(rng.choice(np.flatnonzero(union)))] = 3
    if not np.any(states & 2):
        states.flat[int(rng.choice(np.flatnonzero(union)))] = 3
    return states


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bg, divider, first_color, second_color = map(int, rng.choice(10, 4, replace=False))
    puzzle, used = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            for _ in range(100):
                anchor = split == "train" and index == 0
                second = split == "train" and index == 1
                groups = 1 if anchor else 2 if second else (
                    3 if split == "test" else int(rng.integers(1, 3)))
                union, parts, paths = _scene(rng, groups, anchor, second)
                states = _scans(rng, union, parts, paths, anchor or second, anchor)
                if anchor:
                    c1, c2 = first_color, second_color
                elif second:
                    c1, c2 = second_color, first_color
                else:
                    c1, c2 = map(int, rng.choice([c for c in range(10) if c not in (bg, divider)],
                                                2, replace=False))
                scan1 = np.where((states & 1) != 0, c1, bg).astype(np.int8)
                scan2 = np.where((states & 2) != 0, c2, bg).astype(np.int8)
                output = np.full(union.shape, bg, dtype=np.int8)
                output[parts == 1], output[parts == 2], output[parts == 3] = c1, c2, divider
                # Reflections act on both scans and the answer together.
                if rng.integers(2):
                    scan1, scan2, output = (np.flipud(a) for a in (scan1, scan2, output))
                if rng.integers(2):
                    scan1, scan2, output = (np.fliplr(a) for a in (scan1, scan2, output))
                inp = np.concatenate((scan1, np.full((union.shape[0], 1), divider, dtype=np.int8), scan2), axis=1)
                if rng.integers(2):
                    inp, output = inp.T, output.T
                inp, output = inp.copy(), output.copy()
                if _key(inp) in used:
                    continue
                used.add(_key(inp))
                puzzle[split].append({"input": inp, "output": output})
                break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, exact paths, local comparison, evidence and the shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, infos = set(), {"train": [], "test": []}
    conventions = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in used
            used.add(_key(inp))
            info = _analyze_input(inp)
            conventions.add((info["background"], info["divider"]))
            assert out.shape == info["union"].shape
            assert np.array_equal(out, _answer(info))
            kept = out != info["background"]
            assert np.all(~kept | info["union"])
            # Erasure is essential: this is not simply a normalized OR.
            assert np.any(info["union"] & ~kept)
            assert len(_components(kept)) == len(info["trees"])
            for tree in info["trees"]:
                cells = tree["connector"] + tree["arms"][tree["longest"]] + tree["arms"][tree["shortest"]]
                cellset = set(cells)
                degrees = [sum(p in cellset for p in ((r - 1, c), (r + 1, c),
                                                     (r, c - 1), (r, c + 1))) for r, c in cells]
                assert degrees.count(1) == 2 and all(d in (1, 2) for d in degrees)
                assert len(cells) == sum(tree["teeth"][i][1] for i in (tree["shortest"], tree["longest"])) + len(tree["connector"])
            infos[split].append(info)
    assert len(conventions) == 1
    assert len(infos["train"][0]["trees"]) == 1
    assert len(infos["train"][1]["trees"]) == 2
    assert all(len(i["trees"]) <= 2 for i in infos["train"])
    assert all(len(i["trees"]) == 3 for i in infos["test"])
    a, b = infos["train"][:2]
    assert (a["long_color"], a["short_color"]) == (b["short_color"], b["long_color"])
    first_tree = a["trees"][0]
    assert all(0 < first_tree[k] < len(first_tree["teeth"]) - 1 for k in ("shortest", "longest"))
    assert any(t["shortest"] in (0, len(t["teeth"]) - 1) for t in b["trees"])
    # Counting an overlap twice would select a different longest tooth.
    multiplicity = a["first"].astype(int) + a["second"].astype(int)
    doubled_lengths = [sum(int(multiplicity[p]) for p in arm) for arm in first_tree["arms"]]
    assert max(doubled_lengths) > doubled_lengths[first_tree["longest"]]
    for info in (a, b):
        out = _answer(info)
        states = info["first"].astype(np.int8) + 2 * info["second"].astype(np.int8)
        for tree in info["trees"]:
            path = tree["connector"] + tree["arms"][tree["longest"]] + tree["arms"][tree["shortest"]]
            # Copying either scan, AND and XOR all omit a required path cell.
            assert {int(states[p]) for p in path} == {1, 2, 3}
        assert np.any((out == info["background"]) & (states == 0))


def self_test():
    """Bounded 42-seed audit with structural, not merely palette, metrics."""
    shapes, geometries, palettes, counts, axes, sides, lengths, modes = (set() for _ in range(8))
    signatures = set()
    ties = bilateral = local_global_conflicts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                info = _analyze_input(inp)
                shapes.add(inp.shape)
                palettes.add((info["background"], info["divider"], info["long_color"], info["short_color"]))
                axes.add(info["axis"])
                counts.add(len(info["trees"]))
                states = info["first"].astype(int) + 2 * info["second"].astype(int)
                modes.add(tuple(sorted(map(int, np.unique(states)))))
                signature = []
                for tree in info["trees"]:
                    ls = tuple(t[1] for t in tree["teeth"])
                    ss = tuple(t[2] for t in tree["teeth"])
                    lengths.update(ls)
                    sides.update(ss)
                    bilateral += len(set(ss)) == 2
                    ties += len(set(ls)) < len(ls)
                    signature.append((tree["transpose"], tuple(tree["teeth"]), tree["caps"], tree["spine"]))
                geometries.add((out.shape, tuple(signature)))
                local_global_conflicts += len(info["trees"]) > 1
    assert len(signatures) == 42 and len(geometries) >= 180
    assert len(shapes) >= 35 and len(palettes) >= 100
    assert counts == {1, 2, 3} and axes == {0, 1} and sides == {-1, 1}
    assert len(lengths) >= 8 and ties > 20 and bilateral > 30
    assert local_global_conflicts >= 84 and len(modes) >= 3
    # A valid comb outside the sampler's limits: seven teeth and length 12.
    # Build the expected path explicitly, rather than through the parser.
    union = np.zeros((17, 14), dtype=bool)
    union[:, 0] = True
    for r, length in zip((1, 3, 5, 7, 9, 11, 13), (4, 12, 6, 1, 5, 6, 3)):
        union[r, 1:length + 1] = True
    expected = np.zeros((17, 14), dtype=np.int8)
    expected[3:8, 0] = 9
    expected[3, 1:13] = 3
    expected[7, 1] = 7
    rr, cc = np.indices(union.shape)
    states = np.where(union, 1 + (rr + 2 * cc) % 3, 0)
    scan1, scan2 = np.where(states & 1, 3, 0), np.where(states & 2, 7, 0)
    inp = np.concatenate((scan1, np.full((17, 1), 9), scan2), axis=1)
    assert np.array_equal(solve(inp), expected)
    assert np.array_equal(solve(inp.T), expected.T)
    # Arbitrary source ownership and duplicate counts do not affect a fixed
    # union.  In particular identical scans have the same extremal route.
    identical = np.concatenate((np.where(union, 3, 0), np.full((17, 1), 9),
                                np.where(union, 7, 0)), axis=1)
    assert np.array_equal(solve(identical), expected)
    # Four components in a legal, shorter-toothed layout, beyond the sampled
    # one/two/three population schedule.  Locality has no hardcoded maximum.
    forest = np.zeros((23, 11), dtype=bool)
    expected_forest = np.zeros((23, 11), dtype=np.int8)
    for offset in (0, 6, 12, 18):
        forest[offset, :7] = True
        for c, length in ((1, 3), (3, 1), (5, 2)):
            forest[offset + 1:offset + length + 1, c] = True
        expected_forest[offset, 1:4] = 9
        expected_forest[offset + 1:offset + 4, 1] = 3
        expected_forest[offset + 1, 3] = 7
    inp = np.concatenate((np.where(forest, 3, 0), np.full((23, 1), 9),
                          np.where(forest, 7, 0)), axis=1)
    assert np.array_equal(solve(inp), expected_forest)
    # Determinism, all explicit count combinations, and invalid argument types.
    assert tuple(_key(p["input"]) for p in generate_puzzle(91)["train"]) == tuple(
        _key(p["input"]) for p in generate_puzzle(91)["train"])
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + train * 10 + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4., 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                geometry_signatures=len(geometries), palettes=len(palettes),
                tied_middle_trees=ties, bilateral_trees=bilateral,
                local_global_conflict_pairs=local_global_conflicts,
                occupancy_state_sets=len(modes), tooth_lengths=sorted(lengths),
                unsampled_valid_geometries_checked=2)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
