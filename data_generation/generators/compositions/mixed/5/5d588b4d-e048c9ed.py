"""Square digits become symmetric unary seeds; only stdlib and NumPy are used.

No parent generator is imported. Generation constructs inputs; solve reads cells.
The validator inspects labels and the serialized runs separately from solve.
No files, execution traces, caches, or logs are written by this module.
"""
from __future__ import annotations

import hashlib
import numpy as np


BACKGROUNDS = (2, 3, 7, 8)  # Disjoint from every square's units digit.
DIGITS = frozenset((0, 1, 4, 5, 6, 9))
TRAIN_LENGTHS = (2, 3, 4, 5, 6, 7, 11)
HELD_OUT_LENGTHS = (8, 9, 10)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return array


def _digit(length):
    return (length - 1) ** 2 % 10


def _size(digits):
    positive = [d for d in digits if d > 0]
    return sum(d * d + 2 * d - 2 for d in positive) + max(0, len(positive) - 1)


def _read(input_grid):
    """Recover every role from the input, accepting the full specified domain."""
    inp = _grid(input_grid)
    height, width = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    if background not in BACKGROUNDS or int(counts.max()) * 2 <= inp.size:
        raise ValueError("background must be a strict majority and avoid square digits")
    occupied_counts = np.count_nonzero(inp != background, axis=1)
    singletons = np.flatnonzero(occupied_counts == 1)
    if len(singletons) != 1:
        raise ValueError("exactly one singleton reference row is required")
    marker_row = int(singletons[0])
    column = int(np.flatnonzero(inp[marker_row] != background)[0])
    if np.count_nonzero(inp[:, column] != background) != 1:
        raise ValueError("reference column must be initially clear on all bar rows")
    active_rows = np.flatnonzero(occupied_counts)
    if np.any(np.diff(active_rows) < 2):
        raise ValueError("one blank row must separate consecutive objects")
    bars = []
    for row in active_rows:
        row = int(row)
        if row == marker_row:
            continue
        cells = np.flatnonzero(inp[row] != background)
        start, end, length = int(cells[0]), int(cells[-1]), len(cells)
        if length < 2 or end - start + 1 != length:
            raise ValueError("each data row must contain one horizontal bar of length >= 2")
        color = int(inp[row, start])
        if not np.all(inp[row, start:end + 1] == color):
            raise ValueError("bars must be monochrome")
        if not (end < column - 1 or start > column + 1):
            raise ValueError("each bar must lie on one side with a blank column gap")
        bars.append((row, start, length, color))
    if not 1 <= len(bars) <= 6:
        raise ValueError("one to six bars are required")
    return {"input": inp, "height": height, "width": width,
            "background": background, "marker_row": marker_row,
            "column": column, "bars": bars}


def _render(meta, digits):
    """Expand the computed seeds in bar order, without resetting at row edges."""
    bg, width, height = meta["background"], meta["width"], meta["height"]
    cells = []
    for (_, _, _, color), d in zip(meta["bars"], digits):
        if d == 0:
            continue
        for run in list(range(1, d + 1)) + list(range(d - 1, 0, -1)):
            if cells:
                cells.append(bg)
            cells.extend([color] * run)
    extra_rows = 1 + (len(cells) + width - 1) // width if cells else 0
    out = np.full((height + extra_rows, width), bg, dtype=np.int8)
    out[:height] = meta["input"]
    for (row, _, _, _), digit in zip(meta["bars"], digits):
        out[row, meta["column"]] = digit
    if cells:
        out[height + 1:].flat[:len(cells)] = cells
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution: measure, square, project, expand, concatenate, wrap."""
    meta = _read(input_grid)
    digits = [_digit(b[2]) for b in meta["bars"]]
    rows = meta["height"]
    size = _size(digits)
    if size:
        rows += 1 + (size + meta["width"] - 1) // meta["width"]
    if rows > 30:
        raise ValueError("the specified expansion would exceed 30 rows")
    return _render(meta, digits)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    array = np.asarray(grid)
    return array.shape, array.astype(np.uint8, copy=False).tobytes()


def _plan(rng, specs, sides, excluded_widths=(), goal="free"):
    """Enumerate feasible widths BEFORE allocation; no canvas rejection loop."""
    left = max((length for (length, _), side in zip(specs, sides) if side < 0), default=0)
    right = max((length for (length, _), side in zip(specs, sides) if side > 0), default=0)
    left_min = left + 1 if left else 0
    right_min = right + 1 if right else 0
    n = len(specs)
    size = _size([_digit(length) for length, _ in specs])
    widths = []
    for width in range(max(4, left_min + 1 + right_min), 31):
        stream_rows = (size + width - 1) // width
        extra = 1 + stream_rows if size else 0
        # 2*n+1 holds n bars and one reference, with a blank row per gap.
        minimum_height = max(2 * n + 1, (2 * (1 + sum(x[0] for x in specs))) // width + 1)
        if minimum_height + extra <= 30:
            widths.append(width)
    if not widths:
        return None
    preferred = [w for w in widths if w not in excluded_widths]
    if preferred:
        widths = preferred
    if goal == "narrow":
        widths = widths[:max(1, len(widths) // 2)]
    elif goal == "single":
        preferred = [w for w in widths if size <= w]
        if preferred:
            widths = preferred
    elif goal in ("exact", "one"):
        preferred = [w for w in widths if (-size) % w == (goal == "one")]
        if preferred:
            widths = preferred
    width = int(rng.choice(widths))
    extra = 1 + (size + width - 1) // width if size else 0
    min_height = max(2 * n + 1, (2 * (1 + sum(x[0] for x in specs))) // width + 1)
    # Both splits use this identical height/padding policy.
    height = int(rng.integers(min_height, min(30 - extra, min_height + 7) + 1))
    column = int(rng.integers(left_min, width - right_min))
    return height, width, column


def _construct(rng, background, marker, specs, sides, plan):
    height, width, column = plan
    n = len(specs)
    slack = height - (2 * n + 1)
    spaces = rng.multinomial(slack, np.full(n + 2, 1 / (n + 2)))
    rows = [int(spaces[0])]
    for i in range(n):
        rows.append(rows[-1] + 2 + int(spaces[i + 1]))
    marker_slot = int(rng.integers(n + 1))
    marker_row = rows.pop(marker_slot)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[marker_row, column] = marker
    for row, (length, color), side in zip(rows, specs, sides):
        if side < 0:
            start = int(rng.integers(0, column - length))
        else:
            start = int(rng.integers(column + 2, width - length + 1))
        inp[row, start:start + length] = color
    return inp


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a reproducible puzzle with two substantial teaching anchors."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ns = _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    background = int(rng.choice(BACKGROUNDS))
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
    a, b, c = palette[:3]
    marker = a  # At least one training bar explicitly reuses the marker color.
    anchors = [[(4, a), (2, a), (3, b)],
               [(5, c), (6, a), (7, c), (11, b), (3, c)]]
    # Ordering is spatial, never an implicit sort by size, color, or digit.
    anchors[0] = [anchors[0][int(j)] for j in rng.permutation(3)]
    if [v[0] for v in anchors[0]] in ([2, 3, 4], [4, 3, 2]):
        anchors[0][0], anchors[0][1] = anchors[0][1], anchors[0][0]
    nonzero = [v for v in anchors[1] if v[0] != 11]
    nonzero = [nonzero[int(j)] for j in rng.permutation(len(nonzero))]
    nonzero.insert(int(rng.integers(1, len(nonzero))), (11, b))
    anchors[1] = nonzero
    puzzle = {"train": [], "test": []}
    used, train_widths = set(), []
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            # At most 48 tiny construction attempts; no retained failed grids.
            for attempt in range(48):
                if split == "train" and index < 2:
                    specs = anchors[index]
                elif split == "train" and index == 2:
                    specs = [(3, int(rng.choice(palette)))]
                elif split == "train" and index == 3:
                    specs = [(11, int(rng.choice(palette)))]
                else:
                    n = int(rng.integers(1, 7))
                    pool = TRAIN_LENGTHS if split == "train" else TRAIN_LENGTHS + HELD_OUT_LENGTHS
                    lengths = [int(rng.choice(pool)) for _ in range(n)]
                    if split == "test":
                        lengths[int(rng.integers(n))] = int(rng.choice(HELD_OUT_LENGTHS))
                    specs = [(length, int(rng.choice(palette))) for length in lengths]
                sides = [int(rng.choice((-1, 1))) for _ in specs]
                if split == "train" and index == 0:
                    sides[0], sides[1] = -1, 1
                goal = ("narrow" if split == "train" and index == 0 else
                        "single" if split == "train" and index == 2 else
                        str(rng.choice(("free", "exact", "one"))))
                excluded = train_widths if split == "test" or index == 1 else ()
                plan = _plan(rng, specs, sides, excluded, goal)
                if plan is None:
                    continue
                inp = _construct(rng, background, marker, specs, sides, plan)
                key = _key(inp)
                if key in used:
                    continue
                used.add(key)
                if split == "train":
                    train_widths.append(inp.shape[1])
                puzzle[split].append({"input": inp, "output": solve(inp)})
                break
            else:
                raise RuntimeError("bounded input construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    """Check cells directly, rather than comparing two calls to solve."""
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    meta = _read(pair["input"])
    inp, out = meta["input"], _grid(pair["output"])
    bg, height, width = meta["background"], meta["height"], meta["width"]
    digits = [_digit(length) for _, _, length, _ in meta["bars"]]
    size = _size(digits)
    extra = 1 + (size + width - 1) // width if size else 0
    _require(out.shape == (height + extra, width), "wrong minimal output dimensions")
    prefix = inp.copy()
    for (row, _, _, _), d in zip(meta["bars"], digits):
        prefix[row, meta["column"]] = d
    _require(np.array_equal(out[:height], prefix), "incorrect digit or source preservation")
    _require(np.count_nonzero(inp != out[:height]) == len(meta["bars"]), "one visible label per bar")
    if size:
        _require(np.all(out[height] == bg), "missing blank separation row")
        flat = out[height + 1:].reshape(-1)
        position, foreground_cells, run_count = 0, 0, 0
        for (_, _, _, color), d in zip(meta["bars"], digits):
            for j in range(max(0, 2 * d - 1)):
                length = min(j + 1, 2 * d - 1 - j)
                if run_count:
                    _require(int(flat[position]) == bg, "separator must be one background cell")
                    position += 1
                _require(np.all(flat[position:position + length] == color), "wrong symmetric run")
                position += length
                foreground_cells += length
                run_count += 1
        _require(position == size, "wrong serialized stream length")
        _require(foreground_cells == sum(d * d for d in digits), "wrong stream foreground mass")
        _require(np.all(flat[position:] == bg), "nonbackground final padding")
        _require(np.count_nonzero(flat != bg) == foreground_cells, "extra stream foreground")
        _require(int(flat[size - 1]) != bg, "the final row must contain the last run")
    meta["digits"] = digits
    meta["stream_size"] = size
    return meta


def validate_generated_puzzle(puzzle):
    """Check family semantics, input uniqueness, demonstrations, and held-out codes."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal example counts")
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            meta = _inspect_pair(pair)
            key = _key(meta["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            metas[split].append(meta)
    all_metas = metas["train"] + metas["test"]
    _require(len({m["background"] for m in all_metas}) == 1, "puzzle background must be stable")
    _require(len({int(m["input"][m["marker_row"], m["column"]]) for m in all_metas}) == 1,
             "puzzle marker color must be stable")
    train = metas["train"]
    _require({d for m in train for d in m["digits"]} == DIGITS, "training must teach all six square digits")
    length_colors, color_lengths = {}, {}
    for m in train:
        for _, _, length, color in m["bars"]:
            length_colors.setdefault(length, set()).add(color)
            color_lengths.setdefault(color, set()).add(length)
    _require(any(len(v) > 1 for v in length_colors.values()), "same length needs different colors")
    _require(any(len(v) > 1 for v in color_lengths.values()), "same color needs different lengths")
    _require(any(len(m["bars"]) > 1 and 0 in m["digits"] and m["stream_size"] for m in train),
             "training must show zero alongside nonzero expansions")
    _require(any(m["stream_size"] > m["width"] for m in train), "training must show wrapping")
    _require(len({m["width"] for m in train}) >= 2, "training must vary width")
    _require(any(any(start < m["column"] for _, start, _, _ in m["bars"]) and
                 any(start > m["column"] for _, start, _, _ in m["bars"]) for m in train),
             "training must show both sides of the reference")
    # Both parent concepts have observable consequences even at nt=2.
    _require(any(m["stream_size"] for m in train), "removing A would leave every output unchanged")
    _require(any(d != length % 10 and d != (length - 1) % 10
                 for m in train for (_, _, length, _), d in zip(m["bars"], m["digits"])),
             "training does not distinguish direct length/adjacency labels from B")
    learned_lengths = set(length_colors)
    learned_digits = {d for m in train for d in m["digits"]}
    for m in metas["test"]:
        _require(any(length not in learned_lengths for _, _, length, _ in m["bars"]),
                 "test needs at least one unseen physical bar length")
        _require(set(m["digits"]).issubset(learned_digits), "test uses an untaught unary seed")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, cells = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(cells)
    return digest.hexdigest()


def self_test():
    """Bounded, in-memory 42-seed diversity and semantic regression checks."""
    signatures, shapes, layouts, widths, heights = set(), set(), set(), set(), set()
    backgrounds, colors, bar_counts, lengths, digits_seen = set(), set(), set(), set(), set()
    exact = one_pad = zero_only = split_run = boundary_cross = 0
    ablation_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
        for pair in puzzle["train"][:2]:
            m = _read(pair["input"])
            for seeds in ([bar[2] % 10 for bar in m["bars"]],
                          [(bar[2] - 1) % 10 for bar in m["bars"]]):
                _require(not np.array_equal(_render(m, seeds), pair["output"]),
                         "omitting the square did not change an anchor")
            _require(pair["output"].shape[0] > m["height"],
                     "omitting expansion did not change an anchor")
            ablation_pairs += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _inspect_pair(pair)
                _require(np.array_equal(solve(np.asarray(pair["input"])), pair["output"]), "solver disagrees")
                shapes.add((m["height"], m["width"], pair["output"].shape[0]))
                layouts.add((m["width"], m["column"], tuple((r, s, n) for r, s, n, _ in m["bars"])))
                widths.add(m["width"]); heights.add(m["height"])
                backgrounds.add(m["background"]); bar_counts.add(len(m["bars"]))
                colors.update(b[3] for b in m["bars"])
                lengths.update(b[2] for b in m["bars"]); digits_seen.update(m["digits"])
                size, width = m["stream_size"], m["width"]
                zero_only += size == 0
                if size:
                    exact += size % width == 0
                    one_pad += (-size) % width == 1
                    flat = np.asarray(pair["output"])[m["height"] + 1:].reshape(-1)
                    split_run += any(flat[p - 1] == flat[p] != m["background"] for p in range(width, size, width))
                    offset = 0
                    for d in m["digits"]:
                        if d:
                            if offset:
                                boundary_cross += offset % width != 0
                                offset += 1
                            offset += d * d + 2 * d - 2
    _require(len(signatures) == 42 and len(shapes) >= 80 and len(layouts) >= 150, "insufficient structural variation")
    _require(len(widths) >= 15 and len(heights) >= 10, "insufficient geometry variation")
    _require(backgrounds == set(BACKGROUNDS) and colors == set(range(10)), "insufficient color-role variation")
    _require(bar_counts == set(range(1, 7)) and lengths == set(range(2, 12)), "insufficient count/length variation")
    _require(digits_seen == DIGITS and zero_only >= 42, "missing zero cases")
    _require(exact >= 2 and one_pad >= 2 and split_run >= 40 and boundary_cross >= 40, "missing wrapping/padding contrasts")
    # Every legal explicit count; only one puzzle is retained per iteration.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(7100 + 10 * nt + ns, nt, ns)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns), "wrong counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Literal input-only oracle: 3-cell bar -> seed 4; black is visible on bg 2.
    inp = np.array([[2, 2, 2, 2, 2, 8], [2] * 6, [0, 0, 0, 2, 2, 2]], dtype=int)
    cells = [0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0]
    expected = np.full((8, 6), 2, dtype=int)
    expected[:3] = inp
    expected[2, 5] = 4
    expected[4:].flat[:22] = cells
    _require(np.array_equal(solve(inp), expected), "hand-calculated seed-4 fixture failed")
    _inspect_pair({"input": inp, "output": expected})
    # Deliberate mistakes must be detected by the cell-level validator.
    for row, column, wrong in ((2, 5, 3), (4, 0, 2), (4, 1, 0)):
        corrupted = expected.copy()
        corrupted[row, column] = wrong
        try:
            _inspect_pair({"input": inp, "output": corrupted})
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic output corruption escaped validation")
    # Full domain beyond the sampler: a 28-cell bar still has an input-only answer.
    for length in range(2, 29):
        inp = np.full((3, 30), 8, dtype=int)
        inp[0, 29] = 2
        inp[2, :length] = 0
        _inspect_pair({"input": inp, "output": solve(inp)})
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "layouts": len(layouts), "widths": len(widths),
            "heights": len(heights), "bar_counts": sorted(bar_counts),
            "lengths": sorted(lengths), "digits": sorted(digits_seen),
            "exact_fill_pairs": exact, "one_pad_pairs": one_pad,
            "zero_only_pairs": zero_only, "inside_run_wrap_pairs": split_run,
            "unaligned_bar_boundaries": boundary_cross,
            "anchors_changed_by_each_parent_ablation": ablation_pairs}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
