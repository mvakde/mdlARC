"""Bracket loom: vertically reverse the bracket-selected cells of each column.

Only standard-library modules and NumPy are used. No parent files, saved
examples, generator state, or seed are consulted by solve().
"""

from __future__ import annotations

import numpy as np


OPEN, CLOSE = 1, 8
PAYLOAD_COLORS = (0, 2, 3, 4, 5, 6, 7, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("grids must be integer 2D arrays, size 1..30, colors 0..9")
    return grid


def _selection(grid):
    """Read the visible brackets; a row is either unmarked or fully marked."""
    mask = np.zeros(grid.shape, dtype=bool)
    intervals = []
    for r, row in enumerate(grid):
        left, right = np.flatnonzero(row == OPEN), np.flatnonzero(row == CLOSE)
        if not len(left) and not len(right):
            continue
        if len(left) != 1 or len(right) != 1:
            raise ValueError("a marked row needs exactly one 1 and one 8")
        a, b = int(left[0]), int(right[0])
        if b - a < 2:
            raise ValueError("1 must precede 8 with a nonempty strict interior")
        mask[r, a + 1:b] = True
        intervals.append((r, a, b))
    if not intervals:
        raise ValueError("at least one bracketed row is required")
    return mask, intervals


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reflect each column in the ordered set of its selected positions."""
    grid = _grid(input_grid)
    mask, _ = _selection(grid)
    output = grid.copy()
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        output[rows, c] = grid[rows[::-1], c]
    return output


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _depth(mask):
    return int(mask.sum(axis=0).max())


def _split_row(mask):
    """Does geometry send cells of one row to different destination rows?"""
    destinations = [set() for _ in range(mask.shape[0])]
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        for source, target in zip(rows, rows[::-1]):
            destinations[int(source)].add(int(target))
    return any(len(targets) >= 2 for targets in destinations)


def _reverse_mask(grid, mask, contiguous=False):
    """Counterfactual operation used only to audit demonstrations."""
    output = grid.copy()
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if contiguous else [rows]
        for group in groups:
            output[group, c] = grid[group[::-1], c]
    return output


def _alternatives(grid):
    mask, intervals = _selection(grid)
    # Reflect all payload-colored cells, ignoring interval boundaries.
    unbracketed = _reverse_mask(grid, (grid != OPEN) & (grid != CLOSE))
    horizontal = grid.copy()
    row_permutation = grid.copy()
    bounding_box = grid.copy()
    outermost = grid.copy()
    ascending = grid.copy()
    descending = grid.copy()
    rows = [r for r, _, _ in intervals]
    for i, (r, a, b) in enumerate(intervals):
        horizontal[r, a + 1:b] = grid[r, a + 1:b][::-1]
        row_permutation[r, a + 1:b] = grid[rows[-1 - i], a + 1:b]
        bounding_box[r, a + 1:b] = grid[rows[0] + rows[-1] - r, a + 1:b]
    for c in range(grid.shape[1]):
        selected = np.flatnonzero(mask[:, c])
        if len(selected):
            outermost[selected[0], c] = grid[selected[-1], c]
            outermost[selected[-1], c] = grid[selected[0], c]
        ascending[selected, c] = np.sort(grid[selected, c])
        descending[selected, c] = np.sort(grid[selected, c])[::-1]
    return {
        "omit_B_identity": grid,
        "omit_A_all_payload": unbracketed,
        "whole_grid_flip": grid[::-1],
        "horizontal_interiors": horizontal,
        "one_permutation_of_marked_rows": row_permutation,
        "physical_bbox_reflection": bounding_box,
        "gaps_are_barriers": _reverse_mask(grid, mask, contiguous=True),
        "only_outermost_swap": outermost,
        "sort_values_ascending": ascending,
        "sort_values_descending": descending,
    }


def _choose_rows(rng, n, force_gap=False, broad=False):
    extra = int(rng.integers(2 if force_gap else 0, min(30 - n, 18 if broad else 12) + 1))
    h = n + extra
    mode = str(rng.choice(["all", "block", "scattered", "clustered"]))
    if force_gap:
        rows = np.sort(rng.choice(h, n, replace=False))
        if rows[-1] - rows[0] == n - 1:
            rows = np.arange(n)
            rows[-1] = n + 1
    elif mode == "all":
        h, rows = n, np.arange(n)
    elif mode == "block":
        start = int(rng.integers(h - n + 1))
        rows = np.arange(start, start + n)
    elif mode == "clustered":
        span = min(h, n + int(rng.integers(1, 4)))
        start = int(rng.integers(h - span + 1))
        rows = np.sort(rng.choice(np.arange(start, start + span), n, replace=False))
    else:
        rows = np.sort(rng.choice(h, n, replace=False))
    return h, [int(r) for r in rows]


def _spans(rng, n, width, mode):
    """Inclusive payload endpoints; markers are immediately outside."""
    last = width - 2
    core = int(rng.integers(2, last))
    if mode == "aligned":
        a = int(rng.integers(1, last + 1))
        b = int(rng.integers(a, last + 1))
        return [(a, b)] * n
    if mode == "nested":
        left = sorted(int(x) for x in rng.integers(1, core + 1, n))
        right = sorted((int(x) for x in rng.integers(core, last + 1, n)), reverse=True)
        spans = list(zip(left, right))
        rng.shuffle(spans)
        return spans
    if mode == "fan":
        return [(int(rng.integers(1, core + 1)), int(rng.integers(core, last + 1)))
                for _ in range(n)]
    if mode == "islands":
        cut = int(rng.integers(1, last))
        spans = []
        for _ in range(n):
            lo, hi = (1, cut) if rng.random() < 0.5 else (cut + 1, last)
            a = int(rng.integers(lo, hi + 1))
            spans.append((a, int(rng.integers(a, hi + 1))))
        return spans
    if mode == "staircase":
        length = int(rng.integers(1, max(2, last // 2) + 1))
        starts = sorted(int(x) for x in rng.integers(1, last - length + 2, n))
        if rng.random() < 0.5:
            starts.reverse()
        return [(a, a + length - 1) for a in starts]
    spans = []
    for _ in range(n):
        a = int(rng.integers(1, last + 1))
        spans.append((a, int(rng.integers(a, last + 1))))
    return spans


def _texture(rng, shape, colors, mode):
    h, w = shape
    if mode == "solid":
        return np.full(shape, int(rng.choice(colors)), dtype=np.int8)
    if mode == "bands":
        return np.repeat(rng.choice(colors, h)[:, None], w, axis=1).astype(np.int8)
    if mode == "periodic":
        rr, cc = np.indices(shape)
        return colors[(rr + int(rng.integers(1, len(colors) + 1)) * cc) % len(colors)].copy()
    if mode == "blocks":
        sh, sw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        coarse = rng.choice(colors, ((h + sh - 1) // sh, (w + sw - 1) // sw))
        return np.repeat(np.repeat(coarse, sh, axis=0), sw, axis=1)[:h, :w].copy()
    if mode == "sparse":
        output = np.full(shape, int(colors[0]), dtype=np.int8)
        ink = rng.random(shape) < float(rng.uniform(0.07, 0.25))
        output[ink] = rng.choice(colors, int(ink.sum()))
        return output
    if mode == "runs":
        output = np.empty(shape, dtype=np.int8)
        for r in range(h):
            c = 0
            while c < w:
                end = min(w, c + int(rng.integers(1, 6)))
                output[r, c:end] = rng.choice(colors)
                c = end
        return output
    return rng.choice(colors, shape).astype(np.int8)


def _construct_output(grid, intervals):
    """Construction-side assignment from records, separate from solve()."""
    output = grid.copy()
    for c in range(grid.shape[1]):
        entries = [(r, int(grid[r, c])) for r, a, b in intervals if a < c < b]
        for i, (r, _) in enumerate(entries):
            output[r, c] = entries[-1 - i][1]
    return output


def _make_pair(rng, palette, n, used, witness=0, deeper=False):
    styles = ["random", "runs", "blocks", "bands", "periodic", "sparse", "solid"]
    for _ in range(256):
        w = int(rng.integers(8 if witness else 5, 25 if witness else 31))
        h, rows = _choose_rows(rng, n, force_gap=(witness == 2), broad=deeper)
        if witness or deeper:
            mode = str(rng.choice(["fan", "nested"]))
        else:
            mode = str(rng.choice(["fan", "nested", "aligned", "islands", "staircase", "scattered"]))
        spans = _spans(rng, n, w, mode)
        size = len(palette) if witness else int(rng.integers(2 if deeper else 1, len(palette) + 1))
        colors = np.asarray(rng.choice(palette, size, replace=False), dtype=np.int8)
        grid = _texture(rng, (h, w), colors, str(rng.choice(styles)))
        payload = _texture(rng, (n, w), colors,
                           "bands" if witness == 1 else str(rng.choice(styles[:-1])))
        intervals = []
        for i, (r, (a, b)) in enumerate(zip(rows, spans)):
            grid[r, a:b + 1] = payload[i, a:b + 1]
            grid[r, a - 1], grid[r, b + 1] = OPEN, CLOSE
            intervals.append((r, a - 1, b + 1))
        # A plausible payload in an unmarked row makes color-only selection fail.
        unused = [r for r in range(h) if r not in rows]
        if unused and (witness or rng.random() < 0.45):
            source, a, b = intervals[int(rng.integers(n))]
            target = int(rng.choice(unused))
            grid[target, a + 1:b] = grid[source, a + 1:b]
        mask, _ = _selection(grid)
        output = _construct_output(grid, intervals)
        if _key(grid) in used:
            continue
        if witness:
            counts = mask.sum(axis=0)
            if not (_split_row(mask) and np.any((counts >= 3) & (counts % 2 == 1))
                    and np.any((counts >= 2) & (counts % 2 == 0))):
                continue
            alternatives = _alternatives(grid)
            names = [name for name in alternatives
                     if name not in ("gaps_are_barriers", "only_outermost_swap")]
            if witness == 2:
                names.extend(["gaps_are_barriers", "only_outermost_swap"])
            if any(np.array_equal(output, alternatives[name]) for name in names):
                continue
        if deeper:
            deep_columns = np.flatnonzero(mask.sum(axis=0) == n)
            if (not _split_row(mask) or not len(deep_columns)
                    or all(np.array_equal(output[:, c], grid[:, c]) for c in deep_columns)):
                continue
        used.add(_key(grid))
        return {"input": grid.astype(np.int8), "output": output.astype(np.int8)}
    raise RuntimeError("could not construct a distinct informative pair")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.choice(PAYLOAD_COLORS, int(rng.integers(2, 7)), replace=False)
    n1 = int(rng.integers(3, 5))
    n2 = int(rng.choice([n for n in (4, 5) if n != n1]))
    specs = [(n1, 1), (n2, 2)]
    specs.extend((int(rng.integers(1, 7)), 0) for _ in range(num_train - 2))
    used = set()
    train = [_make_pair(rng, palette, n, used, witness=witness) for n, witness in specs]
    rng.shuffle(train)
    # Extend depth after training has demonstrated both even and odd reversals.
    depth_n = min(10, max(n for n, _ in specs) + int(rng.integers(1, 4)))
    test = [_make_pair(rng, palette, depth_n, used, deeper=True)]
    test.extend(_make_pair(rng, palette, int(rng.integers(1, 11)), used)
                for _ in range(num_test - 1))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, train_masks, test_masks = set(), [], []
    ruled_out = {name: False for name in _alternatives(np.array([[1, 2, 8]]))}
    split_witness = parity_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "the canvas must retain its dimensions")
            mask, _ = _selection(grid)
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            _require(np.array_equal(output[~mask], grid[~mask]), "stationary context changed")
            for c in range(grid.shape[1]):
                rows = [r for r in range(grid.shape[0]) if mask[r, c]]
                _require(all(output[r, c] == grid[rows[-1 - i], c]
                             for i, r in enumerate(rows)), "wrong column rank reflection")
                _require(sorted(int(grid[r, c]) for r in rows)
                         == sorted(int(output[r, c]) for r in rows), "column inventory changed")
            _require(np.array_equal(solve(output), grid), "reflection must be an involution")
            if split == "train":
                train_masks.append(mask)
                split_witness |= _split_row(mask)
                counts = mask.sum(axis=0)
                parity_witness |= bool(np.any((counts >= 3) & (counts % 2 == 1))
                                       and np.any((counts >= 2) & (counts % 2 == 0)))
                for name, alternative in _alternatives(grid).items():
                    ruled_out[name] |= not np.array_equal(alternative, output)
            else:
                test_masks.append(mask)
    _require(split_witness and parity_witness, "training must show fragmentation and both rank parities")
    _require(all(ruled_out.values()), f"training leaves shortcuts: {ruled_out}")
    _require(max(_depth(m) for m in test_masks) > max(_depth(m) for m in train_masks),
             "the test must extend demonstrated column depth")


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded checks, including 42 seeds and every legal explicit count."""
    source = np.array([[int(x) for x in row] for row in
                       ("1234508", "9999999", "7712228", "1555877", "1666668")])
    expected = np.array([[int(x) for x in row] for row in
                         ("1666668", "9999999", "7715228", "1552877", "1234508")])
    _require(np.array_equal(solve(source), expected), "literal ragged/gapped case failed")
    _require(np.array_equal(solve([[1, 2, 8]]), [[1, 2, 8]]), "singleton case failed")
    _require(np.array_equal(solve([[1, 2, 8], [9, 9, 9], [1, 3, 8]]),
                            [[1, 3, 8], [9, 9, 9], [1, 2, 8]]), "gap skipping failed")
    # Rank reflection commutes with inserting stationary rows and side padding.
    added_row = np.full(source.shape[1], 7)
    _require(np.array_equal(solve(np.insert(source, 1, added_row, axis=0)),
                            np.insert(expected, 1, added_row, axis=0)), "gap insertion changed ranks")
    _require(np.array_equal(solve(np.pad(source, ((0, 0), (1, 2)))),
                            np.pad(expected, ((0, 0), (1, 2)))), "translation changed selection")
    mapping = np.arange(10)
    for old, new in zip(PAYLOAD_COLORS, PAYLOAD_COLORS[1:] + PAYLOAD_COLORS[:1]):
        mapping[old] = new
    _require(np.array_equal(solve(mapping[source]), mapping[expected]), "payload color roles leaked")
    # Exercise the rule-level height/width limits beyond the sampled curriculum.
    rr, cc = np.indices((30, 30))
    full = np.asarray(PAYLOAD_COLORS)[(3 * rr + cc) % len(PAYLOAD_COLORS)]
    full[:, 0], full[:, -1] = OPEN, CLOSE
    _require(np.array_equal(solve(full), full[::-1]), "30-rank boundary case failed")
    keys, shapes, masks, palettes, depths, widths, row_counts = set(), set(), set(), set(), set(), set(), set()
    unchanged = aligned = gapped = split_rows = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            mask, intervals = _selection(grid)
            shapes.add(grid.shape)
            masks.add(_key(mask.astype(np.int8)))
            palettes.add(tuple(int(v) for v in np.unique(grid)))
            depths.add(_depth(mask))
            widths.update(b - a - 1 for _, a, b in intervals)
            row_counts.add(len(intervals))
            unchanged += np.array_equal(grid, pair["output"])
            aligned += len({(a, b) for _, a, b in intervals}) == 1
            gapped += intervals[-1][0] - intervals[0][0] + 1 > len(intervals)
            split_rows += _split_row(mask)
            _require(np.array_equal(solve(grid), pair["output"]), "solver/constructor mismatch")
    _require(len(keys) == 42 and len(shapes) >= 45 and len(masks) >= 150, "insufficient structural variation")
    _require(len(palettes) >= 25 and len(depths) >= 6 and len(widths) >= 15
             and len(row_counts) >= 7, "insufficient palette, rank, width, or count variation")
    _require(aligned > 0 and gapped > 0 and split_rows >= 84 and unchanged > 0,
             "missing simple, sparse, split, or unchanged cases")
    _require(_puzzle_key(generate_puzzle(314159, 7, 3))
             == _puzzle_key(generate_puzzle(314159, 7, 3)), "nondeterministic generation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {nt!r}/{ne!r}")
    return {"seeds": 42, "distinct_puzzles": len(keys), "input_shapes": len(shapes),
            "selection_masks": len(masks), "palettes": len(palettes),
            "column_depths": sorted(depths), "payload_widths": sorted(widths),
            "marked_row_counts": sorted(row_counts), "unchanged_pairs": int(unchanged),
            "aligned_pairs": int(aligned), "gapped_pairs": int(gapped),
            "pairs_with_fragmented_rows": int(split_rows)}
