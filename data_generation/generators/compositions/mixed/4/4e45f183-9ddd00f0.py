"""Sector-addressed one-hole completion, with input-only decoding.

No parent code, files, seed, or generated examples are used by solve().
Generation retains at most one puzzle and three small in-memory mask catalogues.
Running this file prints bounded self-test measurements as compact JSON.
"""

from functools import lru_cache
from hashlib import sha256
from itertools import combinations
import json

import numpy as np


ORDERS = (3, 4, 5)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a


def _axis(total, count, n):
    """Equal bands of normalized centroid, with a 1/12-band safety gap.

    Pixel coordinates run from 0 through n-1. Internal band boundaries are
    k*(n-1)/n. Integer arithmetic avoids platform-dependent floating ties.
    """
    numerator, denominator = n * int(total), int(count) * (n - 1)
    if any(12 * abs(numerator - k * denominator) < denominator
           for k in range(1, n)):
        return None
    return min(n - 1, numerator // denominator)


def _address(points, n):
    if not points:
        return None
    r = _axis(sum(p[0] for p in points), len(points), n)
    c = _axis(sum(p[1] for p in points), len(points), n)
    return None if r is None or c is None else (r, c)


def _points(mask):
    return [tuple(map(int, p)) for p in np.argwhere(mask)]


def _origin(position, n):
    return position[0] * (n + 1), position[1] * (n + 1)


def _panel(grid, position, n):
    r, c = _origin(position, n)
    return grid[r:r + n, c:c + n]


def _parse(input_grid):
    """Decode only visible input; no generator catalogue is consulted."""
    a = _grid(input_grid)
    matches = [n for n in ORDERS if a.shape == (n * n + n - 1,) * 2]
    if len(matches) != 1:
        raise ValueError("Expected a square self-sized lattice of order 3, 4, or 5")
    n = matches[0]
    background = int(a[n, 0])
    for k in range(n, len(a), n + 1):
        if not (np.all(a[k, :] == background) and np.all(a[:, k] == background)):
            raise ValueError("Damaged one-cell separator")
    colors = set(map(int, np.unique(a)))
    if len(colors) != 2 or background not in colors:
        raise ValueError("Exactly a separator/background and foreground are required")
    foreground = next(iter(colors - {background}))
    records, used = [], set()
    for r in range(n):
        for c in range(n):
            mask = _panel(a, (r, c), n) == foreground
            if not mask.any():
                continue
            destination = _address(_points(mask), n)
            if destination is None:
                raise ValueError("Ambiguous centroid sector")
            if destination in used:
                raise ValueError("Two motifs address the same destination")
            if mask[destination]:
                raise ValueError("A translated motif occupies its destination's required hole")
            used.add(destination)
            records.append(((r, c), destination, mask))
    if not 1 <= len(records) < n * n:
        raise ValueError("Use a nonempty proper subset of distinct destinations")
    return a, n, background, foreground, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Move whole motifs, then complete just their addressed compartments."""
    a, n, background, foreground, records = _parse(input_grid)
    output = np.full(a.shape, background, dtype=np.int8)
    for _, destination, mask in records:
        tile = _panel(output, destination, n)
        tile[mask] = foreground  # Translation without altering local coordinates.
        tile[tile == background] = foreground
        tile[destination] = background  # Input validation ensures no ink is erased.
    return output


def _completed(n, background, foreground, destinations, hole_transform=0):
    """Direct output construction, separate from the solver's motif relocation."""
    side = n * n + n - 1
    output = np.full((side, side), background, dtype=np.int8)
    for destination in destinations:
        tile = _panel(output, destination, n)
        tile[:] = foreground
        tile[_transform(destination, n, hole_transform)] = background
    return output


def _transform(point, n, mode):
    r, c = point
    if mode >= 4:
        r, c = c, r
    for _ in range(mode % 4):
        r, c = c, n - 1 - r
    return r, c


def _bits_points(bits, n):
    return [divmod(i, n) for i in range(n * n) if bits & (1 << i)]


def _bits_mask(bits, n):
    return np.array([bool(bits & (1 << i)) for i in range(n * n)]).reshape(n, n)


def _bbox_address(points, n):
    rows, cols = zip(*points)
    return _address([(min(rows), min(cols)), (max(rows), max(cols))], n)


def _components(mask):
    remaining = set(_points(mask))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
    return count


@lru_cache(maxsize=3)
def _catalogue(n):
    """Bounded generation aid, not a restriction on accepted solver masks.

    Enumerate 2..4-pixel subsets and their complements, plus geometric shapes.
    At order five there are fewer than 32,000 candidates before filtering.
    Store each retained mask as one integer, never a saved grid/trace payload.
    """
    area = n * n
    full = (1 << area) - 1
    candidates = set()
    for size in range(2, 5):
        for indices in combinations(range(area), size):
            bits = sum(1 << i for i in indices)
            candidates.add(bits)
            candidates.add(full ^ bits)
    for r0 in range(n):
        for c0 in range(n):
            for r1 in range(r0, n):
                for c1 in range(c0, n):
                    rect = [(r, c) for r in range(r0, r1 + 1)
                            for c in range(c0, c1 + 1)]
                    shapes = [rect,
                              [(r, c) for r, c in rect if r in (r0, r1) or c in (c0, c1)],
                              [(r, c) for r, c in rect if r == r0 or c == c0],
                              [(r, c) for r, c in rect if r == r1 or c == c1],
                              [(r, c) for r, c in rect if r == r0 or c == (c0 + c1) // 2]]
                    for points in shapes:
                        candidates.add(sum(1 << (r * n + c) for r, c in points))
    pools = [[] for _ in range(area)]
    witnesses = [[] for _ in range(area)]
    for bits in sorted(candidates):
        points = _bits_points(bits, n)
        if not 2 <= len(points) < area:
            continue
        address = _address(points, n)
        if address is None or bits & (1 << (address[0] * n + address[1])):
            continue
        label = address[0] * n + address[1]
        pools[label].append(bits)
        bbox = _bbox_address(points, n)
        if (bbox is not None and bbox != address
                and _components(_bits_mask(bits, n)) > 1):
            witnesses[label].append(bits)
    if not all(pools) or (n > 3 and not any(witnesses)):
        raise AssertionError("Generation catalogue lacks a required sector or witness")
    return tuple(map(tuple, pools)), tuple(map(tuple, witnesses))


def _subset(rng, n, count):
    """Scatters, strips, corner/central patches, and missing whole axes."""
    positions = [(r, c) for r in range(n) for c in range(n)]
    mode = int(rng.integers(7))
    noise = rng.random((n, n))
    if mode in (5, 6) and count <= n * (n - 1):
        axis, gap = mode - 5, int(rng.integers(n))
        positions = [p for p in positions if p[axis] != gap]
    center = (int(rng.integers(n)), int(rng.integers(n)))
    corner = (int(rng.choice((0, n - 1))), int(rng.choice((0, n - 1))))
    reverse = bool(rng.integers(2))

    def key(p):
        r, c = p
        if mode == 1:
            return (n - 1 - r if reverse else r) + noise[r, c] / 2
        if mode == 2:
            return (n - 1 - c if reverse else c) + noise[r, c] / 2
        if mode == 3:
            return abs(r - corner[0]) + abs(c - corner[1]) + noise[r, c] / 2
        if mode == 4:
            return max(abs(r - center[0]), abs(c - center[1])) + noise[r, c] / 2
        return noise[r, c]

    return sorted(positions, key=key)[:count]


def _asymmetric(destinations, n):
    selected = set(destinations)
    return all({_transform(p, n, m) for p in selected} != selected for m in range(1, 8))


def _make_pair(rng, n, count, background, foreground, anchor=False):
    pools, witnesses = _catalogue(n)
    for _ in range(256):
        if anchor:
            targets = [(r, int(c)) for r, c in enumerate(rng.permutation(n))]
            remaining = [(r, c) for r in range(n) for c in range(n)
                         if (r, c) not in targets]
            indices = rng.choice(len(remaining), count - n, replace=False)
            targets += [remaining[int(i)] for i in indices]
            rng.shuffle(targets)
        else:
            targets = _subset(rng, n, count)
        if anchor and (len({r for r, _ in targets}) != n
                       or len({c for _, c in targets}) != n
                       or not _asymmetric(targets, n)
                       or (n > 3 and not any(witnesses[r * n + c] for r, c in targets))):
            continue
        sources = _subset(rng, n, count)
        if set(sources) == set(targets):
            continue
        rng.shuffle(sources)
        side = n * n + n - 1
        input_grid = np.full((side, side), background, dtype=np.int8)
        witness_used = False
        for source, target in zip(sources, targets):
            label = target[0] * n + target[1]
            options = pools[label]
            if anchor and not witness_used and witnesses[label]:
                options = witnesses[label]
                witness_used = True
            bits = options[int(rng.integers(len(options)))]
            _panel(input_grid, source, n)[_bits_mask(bits, n)] = foreground
        # Both anchors refute first-pixel addressing; the larger one also
        # refutes bbox addressing, which cannot differ at order three here.
        if anchor:
            masks = [_panel(input_grid, p, n) == foreground for p in sources]
            firsts = {_points(m)[0] for m in masks}
            boxes = {_bbox_address(_points(m), n) for m in masks}
            if firsts == set(targets) or (n > 3 and None not in boxes and boxes == set(targets)):
                continue
        return {"input": input_grid,
                "output": _completed(n, background, foreground, targets)}
    raise RuntimeError("Bounded construction could not find an informative layout")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Scale extrapolation, count extrapolation, or unseen intermediate scale.
    regime = int(rng.integers(3))
    train_orders = [3, 5 if regime == 2 else 4]
    train_orders += [int(rng.choice(train_orders)) for _ in range(num_train - 2)]
    train_counts = [int(rng.integers(n + 1, min(n * n - 1, 2 * n) + 1))
                    if i < 2 else int(rng.integers(1, min(7, n * n - 1) + 1))
                    for i, n in enumerate(train_orders)]
    test_order = (5, 4, 4)[regime]
    palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    shared_palette = bool(rng.integers(3) == 0)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            n = train_orders[i] if split == "train" else test_order
            if split == "train":
                motif_count = train_counts[i]
            else:
                low = max(train_counts) + 1 if regime == 1 else 1
                motif_count = int(rng.integers(low, n * n))
            if split == "train" and i == 0:
                background, foreground = palette
            elif split == "train" and i == 1:
                background, foreground = palette[::-1]
            elif shared_palette:
                background, foreground = (palette if rng.integers(2) else palette[::-1])
            else:
                background, foreground = map(int, rng.choice(10, 2, replace=False))
            for _ in range(32):
                pair = _make_pair(rng, n, motif_count, background, foreground,
                                  anchor=split == "train" and i < 2)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded construction could not produce a distinct input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise ValueError("Each pair needs exactly input/output")
    a, n, background, foreground, records = _parse(pair["input"])
    output = _grid(pair["output"])
    if a.shape != output.shape:
        raise ValueError("Input/output shapes differ")
    targets = {d for _, d, _ in records}
    expected = _completed(n, background, foreground, targets)
    if not np.array_equal(output, expected):
        raise ValueError("Output violates addressed one-hole completion")
    for _, destination, mask in records:
        tile = _panel(output, destination, n)
        assert np.all(tile[mask] == foreground), "Translated motif ink was erased"
        assert np.count_nonzero(tile == background) == 1
        assert tile[destination] == background
    assert np.count_nonzero(output == foreground) == len(records) * (n * n - 1)
    assert np.array_equal(solve(a), output)
    return n, background, foreground, records


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Puzzle needs exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("Train/test must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _integer(a, "num_train", 2, 10)
    _integer(b, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            grid = np.asarray(pair["input"], dtype=np.int8)
            key = (grid.shape, grid.tobytes())
            if key in seen:
                raise ValueError("Duplicate input")
            seen.add(key)
            if split == "train":
                train_infos.append(info)
    # Puzzle-level demonstrations refute both missing-parent shortcuts and
    # every rotation/reflection of either the address or the hole map.
    assert len({i[0] for i in train_infos}) >= 2
    assert len({(i[1], i[2]) for i in train_infos}) >= 2
    for mode in range(1, 8):
        assert any({_transform(d, n, mode) for _, d, _ in records}
                   != {d for _, d, _ in records}
                   for n, _, _, records in train_infos), "Ambiguous address orientation"
        assert any(any(_transform(d, n, mode) != d for _, d, _ in records)
                   for n, _, _, records in train_infos), "Ambiguous hole orientation"
    assert any({s for s, _, _ in rec} != {d for _, d, _ in rec}
               for _, _, _, rec in train_infos), "Ignoring A gives the same panels"
    assert any(any(int(m.sum()) < n * n - 1 for _, _, m in rec)
               for n, _, _, rec in train_infos), "Ignoring B gives the same ink"
    assert any(len({d for _, d, _ in rec}) > 1 for _, _, _, rec in train_infos)
    assert any(_components(m) > 1 and _bbox_address(_points(m), n) not in (None, d)
               for n, _, _, rec in train_infos for _, d, m in rec)


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        h.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed seeds, all count combinations, and adversarial checks.

    No puzzles, JSON files, execution traces, images, or logs are saved here.
    All summary sets contain small integers/tuples or SHA-256 digests.
    """
    shapes, counts, palettes, layouts, masks, hashes = (set() for _ in range(6))
    events = {k: 0 for k in ("pairs", "disconnected_motifs", "dense_motifs",
                             "bbox_witnesses", "fixed_motifs", "source_target_overlap",
                             "empty_output_axes", "address_ablation_failures",
                             "completion_ablation_failures", "counterfactual_changes",
                             "scale_extrapolation", "count_extrapolation",
                             "scale_interpolation")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_digest(puzzle))
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        train_orders = {_parse(p["input"])[1] for p in puzzle["train"]}
        test_orders = {_parse(p["input"])[1] for p in puzzle["test"]}
        train_max = max(len(_parse(p["input"])[4]) for p in puzzle["train"])
        events["scale_extrapolation"] += test_orders == {5} and 5 not in train_orders
        events["scale_interpolation"] += train_orders == {3, 5} and test_orders == {4}
        events["count_extrapolation"] += all(len(_parse(p["input"])[4]) > train_max
                                              for p in puzzle["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                n, bg, fg, records = _inspect_pair(pair)
                targets, sources = {d for _, d, _ in records}, {s for s, _, _ in records}
                events["pairs"] += 1
                events["source_target_overlap"] += bool(sources & targets)
                events["empty_output_axes"] += (len({r for r, _ in targets}) < n
                                                or len({c for _, c in targets}) < n)
                events["address_ablation_failures"] += not np.array_equal(
                    _completed(n, bg, fg, sources), pair["output"])
                moved_only = np.full_like(pair["input"], bg)
                for source, destination, mask in records:
                    _panel(moved_only, destination, n)[mask] = fg
                    events["fixed_motifs"] += source == destination
                    events["disconnected_motifs"] += _components(mask) > 1
                    events["dense_motifs"] += int(mask.sum()) > n * n / 2
                    events["bbox_witnesses"] += _bbox_address(_points(mask), n) not in (None, destination)
                    masks.add((n, sha256(mask.tobytes()).hexdigest()))
                events["completion_ablation_failures"] += not np.array_equal(moved_only, pair["output"])
                shapes.add(tuple(pair["input"].shape))
                counts.add((n, len(records)))
                palettes.add((bg, fg))
                layouts.add((n, tuple(sorted(targets))))
        # Keep the same sources and colors but replace one address with an
        # unused sector. The active destination and its coordinate hole change.
        pair = puzzle["train"][0]
        a, n, bg, fg, records = _parse(pair["input"])
        unused = next((r, c) for r in range(n) for c in range(n)
                      if (r, c) not in {d for _, d, _ in records})
        changed = a.copy()
        source, old_destination, _ = records[0]
        tile = _panel(changed, source, n)
        tile[:] = bg
        candidate = _catalogue(n)[0][unused[0] * n + unused[1]][0]
        tile[_bits_mask(candidate, n)] = fg
        new_output = solve(changed)
        assert not np.array_equal(new_output, pair["output"])
        assert np.all(_panel(new_output, old_destination, n) == bg)
        assert _panel(new_output, unused, n)[unused] == bg
        events["counterfactual_changes"] += 1
        before = a.tobytes()
        solve(a)
        assert a.tobytes() == before, "Solver mutated its input"
    assert len(hashes) == 42 and len(shapes) == 3
    assert len(layouts) > 140 and len(masks) > 150 and len(counts) > 25
    assert len(palettes) > 35
    assert all(events[k] > 0 for k in events)
    assert events["address_ablation_failures"] == events["pairs"]
    assert events["completion_ablation_failures"] >= 2 * 42
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(99000 + train * 10 + test, train, test)
            assert (len(p["train"]), len(p["test"])) == (train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Explicitly reject an occupied hole, tied sector, duplicate address,
    # damaged separator, and a saturated address set.
    invalid = []
    a = np.zeros((11, 11), dtype=np.int8)
    _panel(a, (0, 0), 3)[(0, 1, 2), (0, 1, 2)] = 1
    invalid.append(a)
    a = np.zeros((19, 19), dtype=np.int8)
    _panel(a, (0, 0), 4)[(0, 3), (0, 3)] = 1
    invalid.append(a)
    a = np.zeros((11, 11), dtype=np.int8)
    mask = _bits_mask(_catalogue(3)[0][0][0], 3)
    _panel(a, (0, 0), 3)[mask] = 1
    _panel(a, (1, 1), 3)[mask] = 1
    invalid.append(a)
    a = generate_puzzle(8)["train"][0]["input"].copy()
    n = _parse(a)[1]
    a[n, 0] = next(c for c in np.unique(a) if c != a[n, 0])
    invalid.append(a)
    a = np.zeros((11, 11), dtype=np.int8)
    for r in range(3):
        for c in range(3):
            mask = _bits_mask(_catalogue(3)[0][r * 3 + c][0], 3)
            _panel(a, (r, c), 3)[mask] = 1
    invalid.append(a)
    for a in invalid:
        try:
            solve(a)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed family input accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "grid_shapes": sorted(shapes),
            "order_count_combinations": len(counts), "color_role_pairs": len(palettes),
            "destination_layouts": len(layouts), "dimension_aware_motif_masks": len(masks),
            "events": events, "legal_count_combinations": 30,
            "semantic_verification": "generator-owned checks, no independent solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
