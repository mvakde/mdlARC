"""Reflection-supported recovery followed by maximal full-line run collapse.

Only NumPy and the standard library are used. Generation draws small mirrored
stamps on a shared band lattice; it never calls either parent's generator.
No files, random state, or demonstrations are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


TEST_SEEDS = (0, 64532313, 10119579, 918772633, 657951879, 909891516,
              315790836, 454300702, 240983048, 921740759, 104375679,
              419565927, 562155850, 11808762, 384918564, 908980450,
              536926633, 728829285, 99512146, 729801173, 479628714,
              569448054, 202294053, 849374072, 132638975, 670535928,
              540788023, 157174641, 624210474, 994381101, 106189520,
              397492929, 158103975, 893265063, 425772387, 516106547,
              697535038, 346153101, 138378314, 466978267, 970381680,
              349523207)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return grid


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(counts.argmax())
    if counts[background] * 2 <= grid.size:
        raise ValueError("the background must occupy strictly more than half the input")
    return background


def _supported(mask):
    """Find the unique best vertical axis; axis2 is twice its column coordinate."""
    scores = np.zeros(2 * mask.shape[1] - 1, dtype=np.int64)
    for row in mask.astype(np.int16):
        scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    if len(winners) != 1:
        raise ValueError("foreground color has tied maximum reflection support")
    axis2 = int(winners[0])
    mirror = axis2 - np.arange(mask.shape[1])
    valid = (mirror >= 0) & (mirror < mask.shape[1])
    keep = np.zeros_like(mask, dtype=bool)
    keep[:, valid] = mask[:, valid] & mask[:, mirror[valid]]
    return axis2, keep


def _clean(grid):
    background = _background(grid)
    cleaned = np.full(grid.shape, background, dtype=np.int8)
    axes = {}
    for color in map(int, np.unique(grid)):
        if color == background:
            continue
        axis, keep = _supported(grid == color)
        cleaned[keep] = color
        axes[color] = axis
    return cleaned, axes


def _starts(grid, axis):
    lines = grid if axis == 0 else grid.T
    return np.r_[0, 1 + np.flatnonzero(np.any(lines[1:] != lines[:-1], axis=1))]


def _runs(grid, axis):
    return tuple(map(int, np.diff(np.r_[_starts(grid, axis), grid.shape[axis]])))


def _compress(grid):
    return grid[np.ix_(_starts(grid, 0), _starts(grid, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase unsupported cells per color, THEN collapse adjacent whole-line runs.

    The vertical orientation and strict-majority background are fixed family
    conventions. Ambiguous axes are outside the family and raise ValueError.
    """
    cleaned, _ = _clean(_grid(input_grid))
    return _compress(cleaned)


def _expand(grid, rows, columns):
    return np.repeat(np.repeat(grid, rows, axis=0), columns, axis=1)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bbox(mask):
    rows, columns = np.nonzero(mask)
    return int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max())


def _prune_at_axes(grid, axes):
    """Counterfactual prediction with prescribed, potentially incorrect axes."""
    out = np.full_like(grid, _background(grid))
    for color, axis in axes.items():
        mirror = axis - np.arange(grid.shape[1])
        valid = (mirror >= 0) & (mirror < grid.shape[1])
        mask = grid == color
        keep = np.zeros_like(mask)
        keep[:, valid] = mask[:, valid] & mask[:, mirror[valid]]
        out[keep] = color
    return _compress(out)


def _common_axis_works(inp, output, colors):
    return any(np.array_equal(_prune_at_axes(inp, {c: axis for c in colors}), output)
               for axis in range(2 * inp.shape[1] - 1))


def _bbox_rule_works(inp, output, colors):
    axes = {c: _bbox(inp == c)[2] + _bbox(inp == c)[3] for c in colors}
    return np.array_equal(_prune_at_axes(inp, axes), output)


def _stamp(rng, height, width, mode):
    half = (width + 1) // 2
    left = rng.random((height, half)) < float(rng.uniform(.25, .65))
    if mode == 0:  # hollow contour
        left[:] = False
        left[0] = left[-1] = True
        left[:, 0] = True
    elif mode == 1:  # branches/variable profile
        left[:] = False
        for row in range(height):
            level = int(rng.integers(1, half + 1))
            left[row, half-level:] = True
    elif mode == 2:  # paired islands, possibly an isolated axis tip
        left[:] = False
        left[:, int(rng.integers(half))] = True
        left[int(rng.integers(height)), 0] = True
        if width % 2:
            left[int(rng.integers(height)), -1] = True
    # Establish a real full-width reflected pair, without prescribing a middle.
    left[int(rng.integers(height)), 0] = True
    if not left[0].any():
        left[0, -1] = True
    if not left[-1].any():
        left[-1, -1] = True
    return np.concatenate((left, left[:, :width//2][:, ::-1]), axis=1)


def _draw_lattice(rng, count, first, anchor_parity):
    height = int(rng.integers(5, 10))
    width = int(rng.integers(7, 14))
    palette = list(map(int, rng.choice(10, count + 1, replace=False)))
    background, colors = palette[0], palette[1:]
    grid = np.full((height, width), background, dtype=np.int8)
    for index, color in enumerate(colors):
        for _ in range(64):
            h = int(rng.integers(2, min(5, height) + 1))
            widths = list(range(2, min(7, width) + 1))
            if index == 0 and anchor_parity is not None:
                # Integral axes need an odd span and odd central physical width.
                widths = [w for w in widths if (w - 1) % 2 == anchor_parity]
            w = int(rng.choice(widths))
            if first:
                h = int(rng.integers(3, min(5, height - 1) + 1))
                w = int(rng.choice([3, 5]))
            mask = _stamp(rng, h, w, int(rng.integers(4)))
            if first:
                # Teach that equal vectors separated by another vector survive.
                mask[-1] = mask[0]
                mask[1] = False
                mask[1, w//2] = True
                mask[0, 0] = mask[0, -1] = True
                mask[-1] = mask[0]
            top = int(rng.integers(height - h + 1))
            left = int(rng.integers(width - w + 1))
            view = grid[top:top+h, left:left+w]
            if np.any(view[mask] != background):
                continue
            view[mask] = color
            break
        else:
            return None
    # Vertical reflection does not depend on row multiplicity, so normalizing
    # duplicate row runs here is safe and makes the later row-height shift exact.
    grid = grid[_starts(grid, 0)].copy()
    if grid.shape[0] < 3 or np.count_nonzero(grid != background) > grid.size * .48:
        return None
    return grid, background, colors


def _column_widths(rng, grid, colors, first, anchor_parity, expand):
    """Unify mirror-related band widths for every color's own bounding span.

    Background-only spacings and unrelated components remain independent.
    Overlapping constraints may imply uniform widths, which is valid.
    """
    width = grid.shape[1]
    parent = list(range(width))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for color in colors:
        _, _, left, right = _bbox(grid == color)
        for c in range(left, right + 1):
            parent[root(c)] = root(left + right - c)
    groups = {}
    for c in range(width):
        groups.setdefault(root(c), []).append(c)
    widths = np.ones(width, dtype=int)
    if expand:
        uniform = not first and rng.random() < .18
        common = int(rng.integers(1, 4))
        for columns in groups.values():
            value = common if uniform else int(rng.integers(1, 5))
            if int(widths.sum()) + (value - 1) * len(columns) <= 30:
                widths[columns] = value
    if anchor_parity == 0:
        _, _, left, right = _bbox(grid == colors[0])
        middle = (left + right) // 2
        if widths[middle] % 2 == 0:
            columns = groups[root(middle)]
            widths[columns] -= 1
    return widths


def _row_heights(rng, length, target, first, column_expanded):
    if target:
        runs = rng.integers(1, 4, length)
        anchor = int(rng.integers(length))
        runs[anchor] = int(rng.integers(7, 11))
        while runs.sum() > 30:
            candidates = [i for i in range(length) if i != anchor and runs[i] > 1]
            runs[int(rng.choice(candidates))] -= 1
    elif not first and column_expanded and rng.random() < .15:
        runs = np.ones(length, dtype=int)
    else:
        runs = rng.integers(1, 5, length)
        while runs.sum() > 30:
            i = int(rng.choice(np.flatnonzero(runs > 1)))
            runs[i] -= 1
        if runs.max() == 1:
            runs[0] = 2
        if first and len(set(map(int, runs))) == 1:
            runs[-1] = 1 if runs[0] != 1 else 2
    return runs


def _boundary_features(original, cleaned):
    features = []
    for axis in (0, 1):
        before = original if axis == 0 else original.T
        after = cleaned if axis == 0 else cleaned.T
        old_equal = np.all(before[1:] == before[:-1], axis=1)
        new_equal = np.all(after[1:] == after[:-1], axis=1)
        features.append((int(np.count_nonzero(new_equal & ~old_equal)),
                         bool(np.any(old_equal & ~new_equal))))
    return features  # (number of noise-split boundaries, any merged boundary)


def _corrupt(rng, clean, background, axes):
    inp = clean.copy()
    h, w = inp.shape
    clean_row_equal = np.r_[False, np.all(clean[1:] == clean[:-1], axis=1)]
    clean_col_equal = np.r_[False, np.all(clean[:, 1:] == clean[:, :-1], axis=0)]
    for index, (color, axis) in enumerate(axes.items()):
        core = clean == color
        _, _, left, right = _bbox(core)
        candidates = []
        for r, c in np.argwhere(inp == background):
            r, c = int(r), int(c)
            mirror = axis - c
            if 2 * c == axis or (0 <= mirror < w and core[r, mirror]):
                continue
            # Prefer additions that split true repeats, and protrusions that
            # defeat using the corrupted bounding-box midpoint as the axis.
            score = 3 * (int(clean_row_equal[r]) + int(clean_col_equal[c]))
            score += 2 * int(c < left or c > right)
            candidates.append((score + float(rng.random()), r, c))
        candidates.sort(reverse=True)
        amount = int(rng.integers(1, 6)) if index == 0 or rng.random() < .75 else 0
        accepted = 0
        for _, r, c in candidates[:160]:
            if accepted >= amount or np.count_nonzero(inp != background) + 1 >= inp.size * .49:
                break
            mirror = axis - c
            if 0 <= mirror < w and inp[r, mirror] == color:
                continue
            inp[r, c] = color
            try:
                winner, keep = _supported(inp == color)
                valid = winner == axis and np.array_equal(keep, core)
            except ValueError:
                valid = False
            if valid and not any(merged for _, merged in _boundary_features(inp, clean)):
                accepted += 1
            else:
                inp[r, c] = background
    return inp


def _make_pair(rng, count, target, first=False, anchor_parity=None, independent=False):
    for _ in range(180):
        drawn = _draw_lattice(rng, count, first, anchor_parity)
        if drawn is None:
            continue
        lattice, background, colors = drawn
        expand_columns = first or bool(rng.random() < .82)
        columns = _column_widths(rng, lattice, colors, first, anchor_parity, expand_columns)
        rows = _row_heights(rng, lattice.shape[0], target, first, columns.max() > 1)
        clean = _expand(lattice, rows, columns)
        if np.count_nonzero(clean != background) > clean.size * .43:
            continue
        recovered, axes = _clean(clean)
        if not np.array_equal(recovered, clean):
            continue
        if anchor_parity is not None and axes[colors[0]] % 2 != anchor_parity:
            continue
        if independent and len(set(axes.values())) < 2:
            continue
        if first and (len(set(_runs(clean, 1))) < 2 or max(_runs(clean, 1)) == 1):
            continue
        inp = _corrupt(rng, clean, background, axes)
        features = _boundary_features(inp, clean)
        output = _compress(lattice)  # intended answer predates physical noise
        if not any(splits for splits, _ in features):
            continue
        if first and not all(splits for splits, _ in features):
            continue
        if np.array_equal(_compress(inp), output) or clean.shape == output.shape:
            continue
        if independent:
            # At least one visibly asymmetric raw bounding box in this lesson.
            if not any(_bbox(inp == color)[2] + _bbox(inp == color)[3] != axis
                       for color, axis in axes.items()):
                continue
            if _common_axis_works(inp, output, colors) or _bbox_rule_works(inp, output, colors):
                continue
        if anchor_parity is not None and not np.any((inp == colors[0]) & (clean != colors[0])):
            continue
        return {"input": inp.astype(np.int8), "output": output.astype(np.int8)}
    raise RuntimeError("bounded construction could not satisfy the evidence constraints")


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = None
    for split, length in (("train", num_train), ("test", num_test)):
        for index in range(length):
            first = split == "train" and index == 0
            second = split == "train" and index == 1
            count = 1 if first else int(rng.integers(2, 4) if second else rng.integers(1, 5))
            for _ in range(30):
                pair = _make_pair(rng, count, split == "test", first,
                                  0 if first else 1 if second else None, second)
                key = _key(pair["input"])
                if key in seen or (second and _background(pair["input"]) == first_background):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if first:
                    first_background = _background(pair["input"])
                break
            else:
                raise RuntimeError("bounded duplicate/background resampling exhausted")
    return puzzle


def _coordinate_clean(grid):
    """Validator's explicit coordinate enumeration, separate from convolutions."""
    background = _background(grid)
    out = np.full_like(grid, background)
    axes = {}
    for color in map(int, np.unique(grid)):
        if color == background:
            continue
        cells = set(map(tuple, np.argwhere(grid == color).tolist()))
        scores = [sum((r, axis - c) in cells for r, c in cells)
                  for axis in range(2 * grid.shape[1] - 1)]
        best = max(scores)
        winners = [axis for axis, score in enumerate(scores) if score == best]
        _require(len(winners) == 1, "ambiguous per-color vertical reflection")
        axis = winners[0]
        axes[color] = axis
        for r, c in cells:
            if (r, axis - c) in cells:
                out[r, c] = color
    return out, axes


def _list_compression(grid):
    """Whole-vector run comparison, independent of the vectorized solver path."""
    rows = [0] + [r for r in range(1, grid.shape[0]) if grid[r].tolist() != grid[r-1].tolist()]
    cols = [0] + [c for c in range(1, grid.shape[1]) if grid[:, c].tolist() != grid[:, c-1].tolist()]
    return grid[np.ix_(rows, cols)]


def measure_pair(pair):
    inp, output = np.asarray(pair["input"]), np.asarray(pair["output"])
    clean, axes = _clean(inp)
    bg = _background(inp)
    return {
        "input_shape": tuple(map(int, inp.shape)),
        "output_shape": tuple(map(int, output.shape)),
        "background": bg, "colors": len(axes),
        "axes": tuple(axes.values()), "parities": tuple(sorted({a % 2 for a in axes.values()})),
        "rows": _runs(clean, 0), "columns": _runs(clean, 1),
        "noise": int(np.count_nonzero(inp != clean)),
        "splits": tuple(x[0] for x in _boundary_features(inp, clean)),
        "separated_row": len({tuple(row) for row in output.tolist()}) < output.shape[0],
        "separated_column": len({tuple(col) for col in output.T.tolist()}) < output.shape[1],
        "bbox_misleads": any(_bbox(inp == color)[2] + _bbox(inp == color)[3] != axis
                             for color, axis in axes.items()),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            clean, axes = _coordinate_clean(inp)
            bg = _background(inp)
            _require(1 <= len(axes) <= 4, "unsupported foreground count")
            _require(np.any(inp != clean), "A must actually remove unsupported cells")
            _require(np.all((inp == clean) | (clean == bg)), "cleanup must only erase")
            _require(set(np.unique(inp)) == set(np.unique(clean)) == set(np.unique(output)),
                     "colors must survive with their original values")
            _require(np.array_equal(_list_compression(clean), output), "wrong canonical compact answer")
            _require(np.array_equal(solve(inp), output), "input-only solver mismatch")
            _require(np.array_equal(_list_compression(output), output), "nonmaximal collapse")
            rows, cols = _runs(clean, 0), _runs(clean, 1)
            _require(np.array_equal(_expand(output, rows, cols), clean), "lost a supported boundary")
            _require(clean.shape != output.shape, "B must actually remove multiplicities")
            _require(not np.array_equal(_list_compression(inp), output), "omitting A gives the answer")
            features = _boundary_features(inp, clean)
            _require(any(splits for splits, _ in features), "noise never splits a true band")
            _require(not any(merged for _, merged in features), "noise merged an intended boundary")
            if split == "train":
                _require(max(rows) <= 4, "training row bands exceed the demonstrated range")
            else:
                _require(max(rows) >= 7, "test lacks the intended row-band scale shift")
            if split == "train":
                training.append(measure_pair(pair))
    first, second = training[:2]
    _require(first["colors"] == 1 and 0 in first["parities"], "first lesson must show an integral axis")
    _require(second["colors"] >= 2 and 1 in second["parities"], "second lesson must show a half axis")
    _require(len(set(second["axes"])) >= 2 and second["bbox_misleads"], "missing independent-axis evidence")
    lesson = puzzle["train"][1]
    inp, output = np.asarray(lesson["input"]), np.asarray(lesson["output"])
    colors = set(map(int, np.unique(inp))) - {_background(inp)}
    _require(not _common_axis_works(inp, output, colors), "a single shared axis solves the lesson")
    _require(not _bbox_rule_works(inp, output, colors), "bounding-box centers solve the lesson")
    _require(first["background"] != second["background"], "background roles were not varied")
    _require(all(first["splits"]), "first lesson must repair both axis band runs")
    _require(first["separated_row"] and first["separated_column"], "missing order-preservation evidence")
    _require(max(first["rows"]) > 1 and max(first["columns"]) > 1, "missing two-axis compression lesson")
    _require(len(set(first["rows"])) > 1 and len(set(first["columns"])) > 1,
             "first lesson must show nonuniform runs on both axes")


def self_test():
    """Bounded: 42 supplied seeds, all 30 legal counts, small mutation checks.

    Retain feature sets and hashes, never all puzzles or execution traces.
    No disk writes, subprocesses, workers, retries beyond generation's limits.
    """
    records = []
    puzzles, shapes, layouts, backgrounds, counts, row_runs, col_runs = (set() for _ in range(7))
    for seed in TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, output = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.tobytes())
                readonly = inp.copy()
                readonly.setflags(write=False)
                _require(np.array_equal(solve(readonly), output), "solve changed input or used hidden state")
                item = measure_pair(pair)
                records.append(item)
                shapes.add(item["input_shape"])
                backgrounds.add(item["background"])
                counts.add(item["colors"])
                row_runs.add(item["rows"])
                col_runs.add(item["columns"])
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in output.ravel())
                layouts.add((output.shape, normalized))
        puzzles.add(digest.digest())
        if seed == 0:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle
                         for p, q in zip(puzzle[s], again[s]) for k in p), "nondeterminism")
    _require(len(puzzles) == 42, "identical seeded puzzles")
    _require(len(shapes) >= 40 and len(layouts) >= 80, "insufficient geometric variation")
    _require(backgrounds == set(range(10)) and counts == {1, 2, 3, 4}, "missing palette/count variation")
    _require(len(row_runs) >= 50 and len(col_runs) >= 40, "insufficient band variation")
    _require(any(len(set(r["axes"])) >= 3 for r in records), "missing three independent axes")
    _require(any(max(r["rows"]) == 1 or max(r["columns"]) == 1 for r in records), "missing one-axis case")
    # Wrong answers from both ablations and global deduplication are explicit.
    puzzle = generate_puzzle(0, 2, 1)
    for role in ("input", "output"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        if role == "output":
            g = bad["test"][0][role]
            g[0, 0] = (int(g[0, 0]) + 1) % 10
        else:
            p = bad["train"][0]
            p["input"] = _clean(p["input"])[0]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted a damaged puzzle")
    legal = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(123456 + train * 10 + test, train, test))
            legal += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Ties have no invented tie-break; a singleton at two distinct columns ties.
    try:
        solve(np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous axes were accepted")
    return {"seeds": len(TEST_SEEDS), "pairs": len(records), "legal_counts": legal,
            "distinct_puzzles": len(puzzles), "input_shapes": len(shapes),
            "color_normalized_outputs": len(layouts), "row_schedules": len(row_runs),
            "column_schedules": len(col_runs), "backgrounds": sorted(backgrounds),
            "foreground_counts": sorted(counts),
            "one_axis_pairs": sum(max(r["rows"]) == 1 or max(r["columns"]) == 1 for r in records),
            "different_axis_pairs": sum(len(set(r["axes"])) > 1 for r in records),
            "bbox_misleads_pairs": sum(r["bbox_misleads"] for r in records),
            "separated_row_pairs": sum(r["separated_row"] for r in records),
            "separated_column_pairs": sum(r["separated_column"] for r in records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
