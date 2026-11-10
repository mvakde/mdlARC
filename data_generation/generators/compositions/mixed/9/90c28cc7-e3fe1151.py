"""Repair stretched corner mosaics using their shared *logical* color multiset.

Only the standard library and NumPy are used. No files are read or written.
Generation builds labeled panels; solving separately reads only input pixels.
"""

from __future__ import annotations

import hashlib

import numpy as np


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(train, test):
    _require(_integer(train) and 2 <= train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(test) and 1 <= test <= 4, "num_test must be an integer in 1..4")
    _require(train > test, "num_train must exceed num_test")
    return int(train), int(test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "invalid grid shape")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors must be in 0..9")
    return grid.astype(np.int8, copy=False)


def _hist(grid):
    return np.bincount(grid.ravel().astype(np.int64), minlength=10)


def _visible(grid):
    return (bool(np.all(np.any(grid[1:] != grid[:-1], axis=1)))
            and bool(np.all(np.any(grid[:, 1:] != grid[:, :-1], axis=0))))


def _compress(panel):
    """Consecutive equality, never global deduplication or component counting."""
    rows = np.r_[0, np.flatnonzero(np.any(panel[1:] != panel[:-1], axis=1)) + 1]
    cols = np.r_[0, np.flatnonzero(np.any(panel[:, 1:] != panel[:, :-1], axis=0)) + 1]
    compact = panel[np.ix_(rows, cols)].copy()
    bands_r = np.diff(np.r_[rows, panel.shape[0]])
    bands_c = np.diff(np.r_[cols, panel.shape[1]])
    return compact, rows, cols, bands_r, bands_c


def _expand(panel, rows, cols):
    return np.repeat(np.repeat(panel, rows, axis=0), cols, axis=1)


def _cross(grid):
    """The only full monochrome rows and columns are the separator arms."""
    rows = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    cols = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    _require(1 <= len(rows) <= 2 and 1 <= len(cols) <= 2, "need a cross of width 1 or 2")
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    _require(r0 > 0 and r1 < grid.shape[0] and c0 > 0 and c1 < grid.shape[1],
             "cross must separate four nonempty corners")
    _require(len(rows) == r1 - r0 and len(cols) == c1 - c0, "cross arms must be contiguous")
    structural = int(grid[r0, c0])
    _require(bool(np.all(grid[rows, :] == structural) and np.all(grid[:, cols] == structural)),
             "cross arms must have the same color")
    panels = ((slice(0, r0), slice(0, c0)),
              (slice(0, r0), slice(c1, grid.shape[1])),
              (slice(r1, grid.shape[0]), slice(0, c0)),
              (slice(r1, grid.shape[0]), slice(c1, grid.shape[1])))
    return structural, (r0, r1, c0, c1), panels


def _analyze(grid):
    structural, cross, slices = _cross(grid)
    panels = [_compress(grid[rs, cs]) for rs, cs in slices]
    shape = panels[0][0].shape
    _require(shape in _SHAPES and all(p[0].shape == shape for p in panels),
             "compact panels must have one common 2..4 by 2..4 shape")
    observed = []
    for compact, _, _, bands_r, bands_c in panels:
        _require(int(np.count_nonzero(compact == structural)) == 1,
                 "each panel needs exactly one logical placeholder")
        _require(_visible(compact), "logical boundaries must be visible")
        _require(int(bands_r.max()) > 1 or int(bands_c.max()) > 1,
                 "each panel must be expanded")
        counts = _hist(compact)
        counts[structural] = 0
        observed.append(counts)
    # With one hole per panel, a full pointwise maximum is exactly the unique
    # common multiset. Otherwise the missing unit could be assigned arbitrarily.
    common = np.maximum.reduce(observed)
    _require(int(common.sum()) == shape[0] * shape[1], "common multiset is not unique")
    missing = []
    for counts in observed:
        difference = common - counts
        _require(int(difference.sum()) == 1 and np.count_nonzero(difference) == 1,
                 "each panel must have a single color deficit")
        missing.append(int(np.flatnonzero(difference)[0]))
    return structural, cross, slices, panels, observed, common, missing


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all repairs from this input alone; retain its cross and pixel scale."""
    grid = _grid(input_grid)
    structural, _, slices, _, _, _, missing = _analyze(grid)
    output = grid.copy()
    for (rs, cs), color in zip(slices, missing):
        panel = output[rs, cs]
        panel[panel == structural] = color
    return output


def _seam(compact):
    """A locally invisible seam exposed by exactly one other row or column."""
    vertical = np.count_nonzero(compact[1:] != compact[:-1], axis=1)
    horizontal = np.count_nonzero(compact[:, 1:] != compact[:, :-1], axis=0)
    return bool(np.any(vertical == 1) or np.any(horizontal == 1))


def _position_trap(compacts, missing, structural):
    for index, panel in enumerate(compacts):
        row, col = np.argwhere(panel == structural)[0]
        votes = [int(other[row, col]) for j, other in enumerate(compacts)
                 if j != index and other[row, col] != structural]
        counts = np.bincount(votes, minlength=10)
        winners = np.flatnonzero(counts == counts.max())
        if len(winners) == 1 and counts[winners[0]] >= 2 and winners[0] != missing[index]:
            return True
    return False


def _raw_trap(expanded, structural, missing):
    """A unique largest deficit in pixel counts recommends a wrong color."""
    histograms = [_hist(panel) for panel in expanded]
    for counts in histograms:
        counts[structural] = 0
    maximum = np.maximum.reduce(histograms)
    for counts, true_color in zip(histograms, missing):
        difference = maximum - counts
        winners = np.flatnonzero(difference == difference.max())
        if len(winners) == 1 and difference[winners[0]] > 0 and winners[0] != true_color:
            return True
    return False


def _bands(rng, length, total, strong):
    # All compositions are tiny: length <= 4, total <= 14; no search artifacts.
    for _ in range(300):
        cuts = np.sort(rng.choice(np.arange(1, total), length - 1, replace=False))
        values = np.diff(np.r_[0, cuts, total])
        if len(set(map(int, values))) == 1:
            continue
        if (strong and values.max() >= 5) or (not strong and values.max() <= 4):
            return values
    raise RuntimeError("bounded axis sampling exhausted")


def _geometry(rng, shape, regime):
    height, width = shape
    strong_rows = regime in ("rows", "both")
    strong_cols = regime in ("cols", "both")

    def axis(length, strong):
        low = length + 4 if strong else length + 1
        high = 14 if strong else min(14, 4 * length - 1)
        extent = int(rng.integers(low, high + 1))
        return _bands(rng, length, extent, strong), _bands(rng, length, extent, strong)

    top_l, top_r = axis(height, strong_rows)
    bottom_l, bottom_r = axis(height, strong_rows)
    left_t, left_b = axis(width, strong_cols)
    right_t, right_b = axis(width, strong_cols)
    rows = (top_l, top_r, bottom_l, bottom_r)
    cols = (left_t, right_t, left_b, right_b)
    return rows, cols, int(rng.integers(1, 3)), int(rng.integers(1, 3))


def _arrange(rng, shape, structural, witness):
    area = shape[0] * shape[1]
    for _ in range(240):
        size = int(rng.integers(2, min(7, area - int(witness)) + 1))
        colors = rng.choice([c for c in range(10) if c != structural], size, replace=False)
        values = np.r_[colors, rng.choice(colors, area - size)].astype(np.int8)
        common = _hist(values)
        diversity = int(rng.integers(2, min(4, size) + 1))
        omitted = rng.choice(colors, diversity, replace=False)
        missing = np.r_[omitted, rng.choice(omitted, 4 - diversity)].astype(np.int8)
        rng.shuffle(missing)
        complete, damaged, holes = [], [], []
        for color in missing:
            for _ in range(100):
                arrangement = rng.permutation(values).reshape(shape).copy()
                candidates = np.flatnonzero(arrangement.ravel() == color)
                hole = int(rng.choice(candidates))
                corrupted = arrangement.copy()
                corrupted.ravel()[hole] = structural
                if _visible(corrupted):
                    complete.append(arrangement)
                    damaged.append(corrupted)
                    holes.append(hole)
                    break
            else:
                break
        if len(damaged) != 4 or len(set(holes)) < 2:
            continue
        if len({x.tobytes() for x in complete}) < 2:
            continue
        if witness and not (any(common[int(c)] >= 2 for c in missing)
                                and any(_seam(p) for p in damaged)
                                and _position_trap(damaged, missing, structural)):
            continue
        return complete, damaged, [int(c) for c in missing]
    raise RuntimeError("bounded compact-panel sampling exhausted")


def _assemble(panels, structural, gap_r, gap_c):
    tl, tr, bl, br = panels
    top, left = tl.shape
    bottom, right = br.shape
    result = np.full((top + gap_r + bottom, left + gap_c + right), structural, dtype=np.int8)
    result[:top, :left] = tl
    result[:top, left + gap_c:] = tr
    result[top + gap_r:, :left] = bl
    result[top + gap_r:, left + gap_c:] = br
    return result


def _make_pair(rng, shape, structural, regime, witness):
    for _ in range(32):
        complete, damaged, missing = _arrange(rng, shape, structural, witness)
        for _ in range(24):
            rows, cols, gap_r, gap_c = _geometry(rng, shape, regime)
            expanded_in = [_expand(p, r, c) for p, r, c in zip(damaged, rows, cols)]
            expanded_out = [_expand(p, r, c) for p, r, c in zip(complete, rows, cols)]
            if not _raw_trap(expanded_in, structural, missing):
                continue
            if len({tuple(_hist(p)) for p in expanded_out}) < 2:
                continue
            # Independent panel segmentation must matter, rather than merely
            # having four copies of one global row/column expansion profile.
            if (np.array_equal(rows[0], rows[1]) and np.array_equal(rows[2], rows[3])
                    and np.array_equal(cols[0], cols[2]) and np.array_equal(cols[1], cols[3])):
                continue
            return {"input": _assemble(expanded_in, structural, gap_r, gap_c),
                    "output": _assemble(expanded_out, structural, gap_r, gap_c)}
    raise RuntimeError("bounded expanded-panel sampling exhausted")


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # Two shuffled teaching witnesses expose each separate distortion axis.
    anchor_shapes = [(2, 2), ((2, 3), (3, 2), (3, 3))[int(rng.integers(3))]]
    anchor_regimes = ["rows", "cols"]
    rng.shuffle(anchor_regimes)
    specifications = [(shape, regime, True) for shape, regime in zip(anchor_shapes, anchor_regimes)]
    for _ in range(num_train - 2):
        specifications.append((_SHAPES[int(rng.integers(len(_SHAPES)))],
                               str(rng.choice(("rows", "cols", "mild"))), False))
    rng.shuffle(specifications)
    train_shapes = [spec[0] for spec in specifications]
    specifications += [(train_shapes[int(rng.integers(num_train))], "both", False)
                       for _ in range(num_test)]
    fixed_color = bool(rng.integers(2))
    first_color = int(rng.integers(10))
    used = set()
    puzzle = {"train": [], "test": []}
    for index, (shape, regime, witness) in enumerate(specifications):
        if fixed_color or index == 0:
            structural = first_color
        elif index == 1:
            structural = int(rng.choice([c for c in range(10) if c != first_color]))
        else:
            structural = int(rng.integers(10))
        for _ in range(24):
            pair = _make_pair(rng, shape, structural, regime, witness)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle["train" if index < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _details(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output dimensions must be preserved")
    structural, cross, slices, panels, observed, common, missing = _analyze(inp)
    r0, r1, c0, c1 = cross
    _require(np.array_equal(out[r0:r1], inp[r0:r1])
             and np.array_equal(out[:, c0:c1], inp[:, c0:c1]), "cross changed")
    # Enumerate all nine possible first-panel repairs as a second uniqueness
    # check, rather than trusting the maximum calculation used by solve.
    candidates = []
    for color in range(10):
        if color == structural:
            continue
        candidate = observed[0].copy()
        candidate[color] += 1
        differences = [candidate - counts for counts in observed]
        if all(np.all(d >= 0) and d.sum() == 1 and np.count_nonzero(d) == 1
               for d in differences):
            candidates.append(candidate)
    _require(len(candidates) == 1 and np.array_equal(candidates[0], common),
             "common multiset must have exactly one candidate")
    raw_completed = []
    patterns, hole_positions = set(), set()
    for (rs, cs), (compact, starts_r, starts_c, rows, cols), color in zip(slices, panels, missing):
        ip, op = inp[rs, cs], out[rs, cs]
        mask = ip == structural
        _require(np.array_equal(ip, _expand(compact, rows, cols)), "input is not rectangular replication")
        _require(np.array_equal(ip[~mask], op[~mask]), "visible payload changed")
        _require(bool(np.all(op[mask] == color)), "wrong placeholder repair")
        completed = op[np.ix_(starts_r, starts_c)]
        _require(np.array_equal(op, _expand(completed, rows, cols)), "repair distorted a macrocell")
        _require(np.array_equal(_hist(completed), common), "completed compact histograms differ")
        _require(not np.any(op == structural), "unfilled placeholder")
        raw_completed.append(tuple(_hist(op)))
        patterns.add(completed.tobytes())
        hole_positions.add(tuple(map(int, np.argwhere(compact == structural)[0])))
    _require(len(patterns) >= 2 and len(hole_positions) >= 2, "arrangements and holes must vary")
    _require(len(set(raw_completed)) >= 2, "pixel histograms must not be the invariant")
    _require(_raw_trap([inp[rs, cs] for rs, cs in slices], structural, missing),
             "need a wrong raw-count repair witness")
    row_vectors = [p[3] for p in panels]
    col_vectors = [p[4] for p in panels]
    _require(not (np.array_equal(row_vectors[0], row_vectors[1])
                  and np.array_equal(row_vectors[2], row_vectors[3])
                  and np.array_equal(col_vectors[0], col_vectors[2])
                  and np.array_equal(col_vectors[1], col_vectors[3])),
             "panel expansion profiles must differ across a shared edge")
    strong_rows = all(int(v.max()) >= 5 for v in row_vectors)
    strong_cols = all(int(v.max()) >= 5 for v in col_vectors)
    return {"shape": panels[0][0].shape, "structural": structural,
            "multiplicities": tuple(sorted(int(x) for x in common if x)),
            "missing": tuple(missing), "cross_widths": (r1-r0, c1-c0),
            "regime": (strong_rows, strong_cols),
            "witness": (any(common[c] >= 2 for c in missing)
                        and any(_seam(p[0]) for p in panels)
                        and _position_trap([p[0] for p in panels], missing, structural))}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    tests = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            details = _details(pair)
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            (training if split == "train" else tests).append(details)
    _require(len({d["shape"] for d in training}) >= 2, "need two teaching shapes")
    _require(any(d["witness"] and d["regime"] == (True, False) for d in training),
             "training needs a repeated-color, positional-trap row-distortion witness")
    _require(any(d["witness"] and d["regime"] == (False, True) for d in training),
             "training needs a repeated-color, positional-trap column-distortion witness")
    _require(all(d["regime"] != (True, True) for d in training), "training holds out joint strong distortion")
    _require(all(d["regime"] == (True, True) for d in tests), "tests combine the two strong distortions")
    _require(all(d["shape"] in {x["shape"] for x in training} for d in tests), "test shape must be demonstrated")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = _grid(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test() -> dict:
    """42 seeds streamed through bounded checks; return only aggregate evidence."""
    hashes, shapes, canvas_shapes, colors, kinds, gaps, profiles = (set() for _ in range(7))
    replacements, fixed, varying, pairs = set(), 0, 0, 0
    repeated_repairs = distinct_repairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _digest(puzzle)
        _require(signature == _digest(generate_puzzle(seed)), "nondeterministic puzzle")
        hashes.add(signature)
        local_colors = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                before = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]), "solver mismatch")
                _require(np.array_equal(inp, before), "solver mutated its input")
                d = _details(pair)
                shapes.add(d["shape"])
                canvas_shapes.add(inp.shape)
                colors.add(d["structural"])
                local_colors.add(d["structural"])
                kinds.add(d["multiplicities"])
                gaps.add(d["cross_widths"])
                replacements.add(len(set(d["missing"])))
                repeated_repairs += len(set(d["missing"])) < 4
                distinct_repairs += len(set(d["missing"])) == 4
                _, _, _, panel_data, _, _, _ = _analyze(inp)
                profiles.add(tuple((tuple(p[3]), tuple(p[4])) for p in panel_data))
                pairs += 1
        fixed += len(local_colors) == 1
        varying += len(local_colors) > 1
    _require(len(hashes) == 42, "duplicate puzzles across seeds")
    _require(shapes == set(_SHAPES), "compact shape coverage")
    _require(len(colors) == 10 and len(canvas_shapes) >= 35, "palette/geometry diversity")
    _require(len(kinds) >= 15 and len(profiles) >= 150 and len(gaps) == 4, "structural diversity")
    _require(replacements == {2, 3, 4} and fixed > 0 and varying > 0, "repair/color-role diversity")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
            _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test,
                     "explicit counts ignored")
    illegal = ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
               (True, 1), (4.0, 1), (4, False), (4, 1.0), (None, 1), (4, None))
    for train, test in illegal:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Metamorphic checks exercise preserved runs and input-only color roles.
    pair = generate_puzzle(654321, 2, 1)["test"][0]
    inp, out = pair["input"], pair["output"]
    palette = np.array([6, 0, 9, 2, 7, 4, 1, 8, 5, 3], dtype=np.int8)
    _require(np.array_equal(solve(palette[inp]), palette[out]), "color equivariance")
    for turns in (1, 2, 3):
        _require(np.array_equal(solve(np.rot90(inp, turns)), np.rot90(out, turns)), "rotation equivariance")
    _require(np.array_equal(solve(inp[:, ::-1]), out[:, ::-1]), "reflection equivariance")
    _require(np.array_equal(solve(inp.tolist()), out), "nested-list input")
    return {"seeds_checked": 42, "pairs_checked": pairs, "distinct_puzzles": len(hashes),
            "compact_shapes": sorted(shapes), "canvas_shape_count": len(canvas_shapes),
            "structural_colors": sorted(colors), "multiplicity_profile_count": len(kinds),
            "expansion_profile_count": len(profiles), "cross_widths": sorted(gaps),
            "replacement_diversities": sorted(replacements), "fixed_color_puzzles": fixed,
            "varying_color_puzzles": varying, "repeated_repair_pairs": repeated_repairs,
            "four_distinct_repair_pairs": distinct_repairs}
