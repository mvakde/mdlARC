"""Recursive rightward rods, observed through whole-row equality.

Family conventions: 0 is background, 1 starts falling, all other nonzero
colors are passive rod ink, and 5 marks a monochromatic final row.
Only Python's standard library and NumPy are used. No files are read by solve.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


BACKGROUND = 0
TRIGGER = 1
POSITIVE = 5


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("grid must be a 1..30 rectangle of integer ARC colors")
    return grid


def _parse(input_grid):
    grid = _grid(input_grid)
    height, width = grid.shape
    rods = []
    for x in range(width):
        occupied = np.flatnonzero(grid[:, x] != BACKGROUND)
        if not len(occupied):
            continue
        h = len(occupied)
        color = int(grid[-1, x])
        if (h < 2 or not np.array_equal(occupied, np.arange(height - h, height))
                or not np.all(grid[occupied, x] == color) or x + h > width):
            raise ValueError("each rod must be solid, bottom aligned, and fit when fallen")
        rods.append((x, h, color))
    if not rods:
        raise ValueError("at least one upright rod is required")
    for (x, h, _), (next_x, _, _) in zip(rods, rods[1:]):
        if next_x < x + h:
            raise ValueError("landing intervals may not overlap the next rod")
    return rods


def _active(rods, hops=None):
    """Least fixed point of exact-contact activation; hops is for ablations."""
    by_column = {x: i for i, (x, _, _) in enumerate(rods)}
    active = {i for i, (_, _, c) in enumerate(rods) if c == TRIGGER}
    queue = [(i, 0) for i in sorted(active)]
    cursor = 0
    while cursor < len(queue):
        i, depth = queue[cursor]
        cursor += 1
        if hops is not None and depth >= hops:
            continue
        x, h, _ = rods[i]
        j = by_column.get(x + h)
        if j is not None and j not in active:
            active.add(j)
            queue.append((j, depth + 1))
    return active


def _render(grid, rods, active):
    final = np.array(grid, dtype=np.int8, copy=True)
    height = final.shape[0]
    # Clear all activated originals before drawing any landed run.
    for i in active:
        x, h, _ = rods[i]
        final[height - h:, x] = BACKGROUND
    for i in active:
        x, h, color = rods[i]
        final[-1, x:x + h] = color
    return final


def _row_mask(grid):
    return np.all(grid == grid[:, :1], axis=1)


def _label(grid):
    values = np.where(_row_mask(grid), POSITIVE, BACKGROUND).astype(np.int8)
    return np.repeat(values[:, None], grid.shape[1], axis=1)


def _settled(input_grid):
    grid = _grid(input_grid)
    rods = _parse(grid)
    active = _active(rods)
    return _render(grid, rods, active)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from the input, with no learned or generated state."""
    return _label(_settled(input_grid))


def _make(rods, height, width):
    grid = np.zeros((height, width), dtype=np.int8)
    for x, h, color in rods:
        grid[height - h:, x] = color
    _parse(grid)
    return {"input": grid, "output": solve(grid)}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _longest_chain(rods):
    by_column = {x: i for i, (x, _, _) in enumerate(rods)}
    best = 0
    for i, (_, _, color) in enumerate(rods):
        if color != TRIGGER:
            continue
        length = 0
        while i is not None:
            length += 1
            x, h, _ = rods[i]
            i = by_column.get(x + h)
        best = max(best, length)
    return best


def _anchors(rng, palette, need_miss, need_mixed):
    """One recursive scene, its optional one-cell reach contrast, and a full row."""
    for _ in range(128):
        first = int(rng.integers(3, 5))
        middle = int(rng.integers(2, 5))
        tallest = int(rng.integers(max(first, middle) + 1, 8))
        standing = int(rng.integers(2, tallest - 1))
        lead_height = int(rng.integers(2, min(standing, 4) + 1))
        # Keep both row endpoints blank above the baseline, so an interior
        # standing rod also refutes an endpoint-only equality shortcut.
        lead_x = int(rng.integers(1, 4))
        start = lead_x + lead_height + int(rng.integers(1, 3))
        colors = [int(c) for c in rng.permutation(palette)]
        colors = (colors * 3)[:3]
        rods = [(lead_x, lead_height, int(rng.choice(palette))),
                (start, first, TRIGGER),
                (start + first, middle, colors[0]),
                (start + first + middle, tallest, colors[1])]
        end = start + first + middle + tallest
        rods.append((end + 1, standing, colors[2]))
        end += 1 + standing
        if rng.random() < 0.65:
            extra_h = int(rng.integers(2, min(5, tallest)))
            extra_x = end + int(rng.integers(0, 3))
            rods.append((extra_x, extra_h, TRIGGER))
            end = extra_x + extra_h
        width = end + int(rng.integers(0, 4))
        if width <= 30:
            break
    else:
        raise RuntimeError("bounded anchor construction exhausted")
    height = tallest + int(rng.integers(0, 4))
    hit = _make(rods, height, width)
    anchors = [hit]
    if need_miss:
        miss_rods = rods.copy()
        miss_rods[1] = (start, first - 1, TRIGGER)
        anchors.append(_make(miss_rods, height, width))

    # This teaches that a non-background monochromatic row is also positive.
    # Include height 2, keeping a useful short reach demonstrated for tests.
    heights = [2] + [int(rng.choice([r[1] for r in rods]))
                     for _ in range(int(rng.integers(1, 3)))]
    rng.shuffle(heights)
    packed = []
    x = 0
    for h in heights:
        packed.append((x, h, TRIGGER))
        x += h
    packed_height = max(heights) + int(rng.integers(0, 4))
    anchors.append(_make(packed, packed_height, x))
    if need_mixed:
        mixed = packed.copy()
        last_x, last_h, _ = mixed[-1]
        mixed[-1] = (last_x, last_h, int(rng.choice(palette)))
        anchors.append(_make(mixed, packed_height, x))
    return anchors


def _random_pair(rng, palette, heights, is_test):
    for _ in range(256):
        count = int(rng.integers(1, 9 if is_test else 7))
        hs = [int(rng.choice(heights)) for _ in range(count)]
        gaps = [int(rng.choice((0, 0, 0, 1, 1, 2, 3)))
                for _ in range(count - 1)]
        start = int(rng.integers(0, 5))
        margin = int(rng.integers(0, 5))
        width = start + sum(hs) + sum(gaps) + margin
        if width > 30:
            continue
        cs = [TRIGGER if rng.random() < 0.34 else int(rng.choice(palette))
              for _ in hs]
        if TRIGGER not in cs:
            cs[int(rng.integers(count))] = TRIGGER
        rods = []
        x = start
        for i, (h, c) in enumerate(zip(hs, cs)):
            rods.append((x, h, c))
            if i < count - 1:
                x += h + gaps[i]
        if not is_test and _longest_chain(rods) > 3:
            continue
        headroom = int(rng.integers(0, 9 if is_test else 5))
        return _make(rods, max(hs) + headroom, width)
    raise RuntimeError("bounded random construction exhausted")


def _long_test(rng, palette, heights):
    """A longer chain with a late tall passive whose activation changes labels."""
    tallest = max(heights)
    short = [h for h in heights if h < tallest]
    for _ in range(256):
        count = int(rng.integers(4, 7))
        hs = [int(rng.choice(short)) for _ in range(count)]
        tall_index = int(rng.integers(2, count - 1))
        hs[tall_index] = tallest
        start = int(rng.integers(0, 4))
        rods = []
        if rng.random() < 0.4:
            lead_h = int(rng.choice(short))
            rods.append((start, lead_h, int(rng.choice(palette))))
            start += lead_h + int(rng.integers(1, 3))
        x = start
        for i, h in enumerate(hs):
            color = TRIGGER if i == 0 else int(rng.choice(palette))
            # A downstream trigger tests convergence without changing the tall witness.
            if i == count - 1 and rng.random() < 0.3:
                color = TRIGGER
            rods.append((x, h, color))
            x += h
        if rng.random() < 0.6:
            h = int(rng.choice(short))
            rods.append((x + 1, h, int(rng.choice(palette))))
            x += h + 1
        width = x + int(rng.integers(0, 4))
        if width > 30:
            continue
        height = tallest + int(rng.integers(2, 9))
        return _make(rods, height, width)
    raise RuntimeError("bounded long-chain construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.choice(np.arange(2, 10),
                                        size=int(rng.integers(1, 4)), replace=False)]
    train = _anchors(rng, palette, need_miss=num_train >= 3, need_mixed=num_train >= 4)
    seen = {_key(pair["input"]) for pair in train}
    heights = sorted({h for pair in train for _, h, _ in _parse(pair["input"])})
    puzzle = {"train": train, "test": []}
    for split, target in (("train", num_train), ("test", num_test)):
        while len(puzzle[split]) < target:
            for _ in range(128):
                if split == "test" and not puzzle["test"]:
                    pair = _long_test(rng, palette, heights)
                else:
                    pair = _random_pair(rng, palette, heights, split == "test")
                if split == "test" and num_train < 4:
                    baseline = _settled(pair["input"])[-1]
                    if np.all(baseline != BACKGROUND) and not np.all(baseline == baseline[0]):
                        continue
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_matched_reach_contrast(pairs):
    for a in pairs:
        for b in pairs:
            ia, ib = np.asarray(a["input"]), np.asarray(b["input"])
            if (ia.shape == ib.shape and np.count_nonzero(ia != ib) == 1
                    and not np.array_equal(a["output"], b["output"])):
                ra, rb = _parse(ia), _parse(ib)
                if len(ra) != len(rb):
                    continue
                changes = [(x, h, c, xx, hh, cc) for (x, h, c), (xx, hh, cc)
                           in zip(ra, rb) if (x, h, c) != (xx, hh, cc)]
                if len(changes) == 1:
                    x, h, c, xx, hh, cc = changes[0]
                    if x == xx and c == cc == TRIGGER and abs(h - hh) == 1:
                        return True
    return False


def validate_generated_puzzle(puzzle):
    """Check geometry, causal witnesses, row semantics, and split contracts."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    heights_by_split = {"train": set(), "test": set()}
    palettes_by_split = {"train": set(), "test": set()}
    witnesses = set()
    test_chain = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "exact input/output keys required")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            key = _key(inp)
            _require(key not in seen, "inputs must be distinct within a puzzle")
            seen.add(key)
            rods = _parse(inp)
            active = _active(rods)
            _require(any(c == TRIGGER for _, _, c in rods), "missing trigger")
            for i, (x, h, c) in enumerate(rods):
                heights_by_split[split].add(h)
                palettes_by_split[split].add(c)
                if c != TRIGGER and i in active:
                    _require(any(j in active and xx + hh == x
                                 for j, (xx, hh, _) in enumerate(rods)),
                             "active passive lacks an active exact predecessor")
            final = _render(inp, rods, active)
            _require(np.array_equal(out, _label(final)), "wrong final row labels")
            _require(not np.array_equal(inp, out), "identity example")
            _require(set(int(c) for c in np.unique(out)) <= {BACKGROUND, POSITIVE},
                     "output is not binary")
            standing = [r for i, r in enumerate(rods) if i not in active]
            tallest_standing = max((h for _, h, _ in standing), default=1)
            expected_upper = np.arange(inp.shape[0] - 1) < inp.shape[0] - tallest_standing
            _require(np.array_equal(_row_mask(final)[:-1], expected_upper),
                     "upper-row geometry invariant failed")
            for i, (x, h, _) in enumerate(rods):
                if i not in active:
                    _require(np.array_equal(final[:, x], inp[:, x]),
                             "unreached passive changed")
                else:
                    _require(np.all(final[inp.shape[0] - h:-1, x] == BACKGROUND),
                             "vacated upright cells were not cleared")
            if split == "test":
                test_chain |= _longest_chain(rods) >= 4
                if np.all(final[-1] != BACKGROUND) and not _row_mask(final)[-1]:
                    _require("filled_mixed_negative" in witnesses,
                             "fully occupied mixed test baseline lacks a training contrast")
                continue
            _require(_longest_chain(rods) <= 3, "training chain exceeds split limit")
            if not np.array_equal(out, _label(inp)):
                witnesses.add("toppling")
            leftmost_roles = [(x, h, TRIGGER if i == 0 else 2)
                              for i, (x, h, _) in enumerate(rods)]
            if not np.array_equal(out, _label(_render(inp, rods, _active(leftmost_roles)))):
                witnesses.add("trigger_role")
            if not np.array_equal(out, _label(_render(inp, rods, _active(rods, hops=1)))):
                witnesses.add("recursion")
            if not np.array_equal(out, _label(_render(inp, rods, set(range(len(rods)))))):
                witnesses.add("selectivity")
            if not np.array_equal(out, final):
                witnesses.add("row_predicate")
            if any(np.all(row == row[0]) and row[0] != BACKGROUND for row in final):
                witnesses.add("nonbackground_equality")
            if np.all(final[-1] != BACKGROUND) and not _row_mask(final)[-1]:
                witnesses.add("filled_mixed_negative")
            if any(row[0] == row[-1] and not np.all(row == row[0]) for row in final):
                witnesses.add("whole_width")
            if any(i in active and c != TRIGGER for i, (_, _, c) in enumerate(rods)):
                witnesses.add("passive_hit")
            if standing:
                witnesses.add("passive_unreached")
            for i, (x, h, _) in enumerate(rods[:-1]):
                if (i in active and rods[i + 1][0] == x + h + 1
                        and i + 1 not in active):
                    witnesses.add("one_cell_gap")
    _require(witnesses >= {"toppling", "trigger_role", "recursion", "selectivity", "row_predicate",
                           "nonbackground_equality", "whole_width", "passive_hit",
                           "passive_unreached", "one_cell_gap"},
             f"training lacks causal contrasts: {sorted(witnesses)}")
    _require(heights_by_split["test"] <= heights_by_split["train"],
             "test introduced an undemonstrated height")
    _require(palettes_by_split["test"] <= palettes_by_split["train"],
             "test introduced an undemonstrated rod color")
    _require(test_chain, "test lacks longer-chain transfer")
    if len(puzzle["train"]) >= 3:
        _require(_has_matched_reach_contrast(puzzle["train"]),
                 "missing paired height-dependent hit/miss contrast")
    if len(puzzle["train"]) >= 4:
        _require("filled_mixed_negative" in witnesses,
                 "missing completely occupied but nonuniform baseline contrast")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                shape, data = _key(pair[name])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Bounded, streaming 42-seed sweep plus count and geometry checks."""
    shapes, structures, palettes, masks, fingerprints = set(), set(), set(), set(), set()
    rod_counts, chain_lengths, seen_heights = set(), set(), set()
    all_positive = all_negative = positive_passive = pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed replay failed")
        fingerprints.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                rods = _parse(inp)
                pairs_checked += 1
                shapes.add(inp.shape)
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                masks.add(tuple(bool(v) for v in _row_mask(_settled(inp))))
                rod_counts.add(len(rods))
                chain_lengths.add(_longest_chain(rods))
                seen_heights.update(h for _, h, _ in rods)
                all_positive += bool(np.all(out == POSITIVE))
                all_negative += bool(np.all(out == BACKGROUND))
                positive_passive += any(c == POSITIVE for _, _, c in rods)
                _require(np.array_equal(out, solve(inp.tolist())), "list input failed")
                before = inp.copy()
                _require(np.array_equal(out, solve(inp.copy())), "copy input failed")
                _require(np.array_equal(out, solve(inp)), "array input failed")
                _require(np.array_equal(inp, before), "solve mutated its input")
    _require(len(fingerprints) == 42, "puzzle diversity failed")
    _require(len(shapes) >= 50 and len(structures) >= 170, "geometric diversity failed")
    _require(len(palettes) >= 20 and len(masks) >= 25, "palette/mask diversity failed")
    _require(len(rod_counts) >= 6 and len(chain_lengths) >= 5 and len(seen_heights) >= 5,
             "rod structure diversity failed")
    _require(all_positive >= 42 and all_negative > 0 and positive_passive > 0,
             "equality or color-overlap coverage failed")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(123456 + a * 10 + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b,
                     "explicit count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-computed residual geometry, including contact at a canvas boundary.
    hand = _make([(0, 2, 1), (2, 3, 7), (6, 2, 8)], 4, 8)
    expected_final = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 8, 0],
                               [1, 1, 7, 7, 7, 0, 8, 0]], dtype=np.int8)
    _require(np.array_equal(_settled(hand["input"]), expected_final),
             "quarter-turn, own reach, gap, or preservation check failed")
    _require(np.array_equal(hand["output"][:, 0], [5, 5, 0, 0]),
             "hand row mask failed")
    packed = _make([(0, 2, 1), (2, 3, 1)], 3, 5)
    _require(np.all(packed["output"] == POSITIVE), "occupied uniform baseline failed")
    # Height at the edge and a passive-only input are natural rule extensions.
    _require(np.all(solve(np.array([[1, 0], [1, 0]])) == POSITIVE),
             "exact boundary landing failed")
    _require(np.array_equal(solve(np.array([[0, 7, 0], [0, 7, 0]])),
                            np.zeros((2, 3), dtype=np.int8)), "passive-only extension failed")
    for bad in (np.array([[1, 0, 7], [1, 0, 7]]),
                np.array([[1, 7, 0], [1, 7, 0]])):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("off-canvas or overlapping landing accepted")
    return {"seeds": 42, "pairs_checked": pairs_checked,
            "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "normalized_inputs": len(structures), "palettes": len(palettes),
            "row_masks": len(masks), "rod_counts": sorted(rod_counts),
            "chain_lengths": sorted(chain_lengths), "rod_heights": sorted(seen_heights),
            "all_positive_pairs": all_positive, "all_negative_pairs": all_negative,
            "positive_ink_as_passive_pairs": positive_passive}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
