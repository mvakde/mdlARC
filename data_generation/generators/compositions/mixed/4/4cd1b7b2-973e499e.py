"""Complete a four-symbol Latin square, then expand its completed color masks.

Only NumPy and the standard library are used. No files are read or written.
The tiny, exhaustive Latin catalog is mathematical data, not stored examples.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations
import hashlib
import json

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


@lru_cache(maxsize=1)
def _latin_catalog():
    """Enumerate all 576 labeled order-four Latin squares, in a fixed order."""
    row_options = tuple(permutations(range(4)))
    results = []

    def visit(rows):
        if len(rows) == 4:
            results.append(rows)
            return
        for row in row_options:
            if all(row[c] not in [old[c] for old in rows] for c in range(4)):
                visit(rows + (row,))

    visit(())
    catalog = np.asarray(results, dtype=np.int8)
    catalog.flags.writeable = False
    return catalog


def _grid(value, shape=None):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 2D integer ARC grid with colors 0..9")
    if shape is not None and grid.shape != shape:
        raise ValueError(f"expected shape {shape}")
    return grid


def _roles(grid):
    """The missing-entry marker is the sole line-repeated color."""
    colors = tuple(int(c) for c in np.unique(grid))
    repeated = [c for c in colors if np.any(np.sum(grid == c, axis=0) > 1)
                or np.any(np.sum(grid == c, axis=1) > 1)]
    if len(colors) != 5 or len(repeated) != 1:
        raise ValueError("need four visible symbols and one line-repeated blank")
    blank = repeated[0]
    return blank, tuple(c for c in colors if c != blank)


def _candidates(grid, blank, symbols):
    labels = np.full((4, 4), 4, dtype=np.int8)
    for i, color in enumerate(symbols):
        labels[grid == color] = i
    known = grid != blank
    catalog = _latin_catalog()
    return catalog[np.all((catalog == labels) | ~known, axis=(1, 2))]


def _complete(grid):
    blank, symbols = _roles(grid)
    candidates = _candidates(grid, blank, symbols)
    if len(candidates) != 1:
        raise ValueError("clues must have exactly one Latin completion")
    return np.asarray(symbols, dtype=np.int8)[candidates[0]], blank


def _expand(square, blank):
    """Each macro cell selects its own color's entire completed support."""
    selector = np.repeat(np.repeat(square, 4, axis=0), 4, axis=1)
    reference = np.tile(square, (4, 4))
    return np.where(selector == reference, selector, blank).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; reject ambiguous or out-of-family inputs."""
    square, blank = _complete(_grid(input_grid, (4, 4)))
    return _expand(square, blank)


def _unique_mask(square, mask):
    # square has abstract symbols 0..3; use 4 exclusively as a temporary blank.
    if len(np.unique(square[~mask])) != 4:
        return False
    catalog = _latin_catalog()
    return int(np.count_nonzero(np.all((catalog == square) | mask, axis=(1, 2)))) == 1


def _sample_square(rng, kind, orientation=False, no_extra_constraints=False):
    rows, cols = np.indices((4, 4))
    base = (rows + cols) % 4 if kind == 0 else rows ^ cols
    for _ in range(128):
        square = rng.permutation(4)[base[np.ix_(rng.permutation(4), rng.permutation(4))]]
        if orientation and any(np.array_equal(square, other) for other in
                               (square.T, np.rot90(square), np.rot90(square, 2))):
            continue
        if no_extra_constraints:
            boxes = [square[r:r + 2, c:c + 2] for r in (0, 2) for c in (0, 2)]
            if all(len(np.unique(box)) == 4 for box in boxes):
                continue
            if len(np.unique(np.diag(square))) == 4:
                continue
        return square.astype(np.int8)
    raise RuntimeError("failed to sample a Latin geometry")


def _mask(rng, square, style, target, axis):
    if style == "rectangle":
        options = []
        for rows in combinations(range(4), 2):
            for cols in combinations(range(4), 2):
                mask = np.zeros((4, 4), dtype=bool)
                mask[np.ix_(rows, cols)] = True
                if _unique_mask(square, mask):
                    options.append(mask)
        return options[int(rng.integers(len(options)))].copy()
    best = None
    for _ in range(16):
        mask = np.zeros((4, 4), dtype=bool)
        focus = int(rng.integers(4))
        positions = rng.permutation(4)
        if style == "easy":
            if axis == 0:
                mask[focus, positions[:target]] = True
            else:
                mask[positions[:target], focus] = True
            return mask
        if style == "line":
            if axis == 0:
                mask[focus, :] = True
            else:
                mask[:, focus] = True
        elif target < 5:
            # Locally identify the marker even in a lightly erased example.
            if axis == 0:
                mask[focus, positions[:2]] = True
            else:
                mask[positions[:2], focus] = True
        preference = rng.random((4, 4))
        if style == "cluster":
            center = rng.integers(4, size=2)
            rr, cc = np.indices((4, 4))
            preference += 0.3 * (abs(rr - center[0]) + abs(cc - center[1]))
        for flat in np.argsort(preference, axis=None):
            if int(mask.sum()) >= target:
                break
            r, c = divmod(int(flat), 4)
            if mask[r, c]:
                continue
            mask[r, c] = True
            if not _unique_mask(square, mask):
                mask[r, c] = False
        if best is None or int(mask.sum()) > int(best.sum()):
            best = mask.copy()
        if int(mask.sum()) == target:
            break
    if best is None or (target >= 8 and int(best.sum()) < 8):
        raise RuntimeError("failed to construct a sufficiently sparse unique mask")
    return best


def _key(grid):
    return (grid.shape, tuple(int(c) for c in grid.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    blank = int(rng.integers(10))
    available = np.asarray([c for c in range(10) if c != blank], dtype=np.int8)
    common_palette = rng.choice(available, size=4, replace=False)
    changing_palette = bool(rng.integers(2))
    first_kind = int(rng.integers(2))
    easy_axis = int(rng.integers(2))
    plans = [("easy", int(rng.integers(2, 4)), easy_axis, first_kind),
             ("rectangle", 4, 1 - easy_axis, 1 - first_kind)]
    for _ in range(num_train - 2):
        style = ("few", "rectangle", "line", "random", "cluster")[int(rng.integers(5))]
        low, high = {"few": (3, 5), "rectangle": (4, 4), "line": (4, 7),
                     "random": (5, 7), "cluster": (5, 7)}[style]
        plans.append((style, int(rng.integers(low, high + 1)), int(rng.integers(2)), int(rng.integers(2))))
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, plan):
        style, target, axis, kind = plan
        for _ in range(128):
            square = _sample_square(rng, kind, orientation=style == "easy",
                                    no_extra_constraints=style == "rectangle")
            mask = _mask(rng, square, style, target, axis)
            palette = rng.choice(available, size=4, replace=False) if changing_palette else common_palette
            completed = palette[square]
            inp = completed.copy()
            inp[mask] = blank
            key = _key(inp)
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append({"input": inp, "output": _expand(completed, blank)})
            return
        raise RuntimeError("failed to generate a distinct pair")

    for plan in plans:
        add("train", plan)
    # The anchors have no fixed position in the returned demonstrations.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    for _ in range(num_test):
        add("test", ("sparse", int(rng.integers(8, 12)), int(rng.integers(2)), int(rng.integers(2))))
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _intersection_mask(mask):
    return (int(mask.sum()) == 4
            and sorted(np.sum(mask, axis=0).tolist()) == [0, 0, 2, 2]
            and sorted(np.sum(mask, axis=1).tolist()) == [0, 0, 2, 2])


def _intercalate_count(square):
    return sum(len(np.unique(square[np.ix_(rows, cols)])) == 2
               for rows in combinations(range(4), 2)
               for cols in combinations(range(4), 2))


def validate_generated_puzzle(puzzle):
    """Check returned grids, Latin uniqueness, completed masks, and teaching cases."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "illegal counts")
    seen, blanks, train_kinds = set(), set(), set()
    easy = intersection = oriented = no_boxes = no_diagonal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"], (4, 4)), _grid(pair["output"], (16, 16))
            blank, symbols = _roles(inp)
            blanks.add(blank)
            mask = inp == blank
            count = int(mask.sum())
            _require(2 <= count <= 11, "invalid clue density")
            _require(count <= 7 if split == "train" else count >= 8, "missing sparsity shift")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            # Recover the selector at each macro position from the returned tile.
            completed = np.empty((4, 4), dtype=np.int8)
            for r in range(4):
                for c in range(4):
                    tile = out[4*r:4*r + 4, 4*c:4*c + 4]
                    colors = set(int(v) for v in tile.flat) - {blank}
                    _require(len(colors) == 1 and np.count_nonzero(tile != blank) == 4,
                             "each tile needs four copies of exactly one symbol")
                    completed[r, c] = colors.pop()
            for line in list(completed) + list(completed.T):
                _require(set(int(v) for v in line) == set(symbols), "not a Latin square")
            _require(np.array_equal(completed[~mask], inp[~mask]), "changed a clue")
            matches = _candidates(inp, blank, symbols)
            _require(len(matches) == 1, "completion not unique")
            _require(np.array_equal(np.asarray(symbols)[matches[0]], completed), "wrong completion")
            # Scalar checks intentionally express the defining rule directly.
            for r in range(4):
                for c in range(4):
                    for u in range(4):
                        for v in range(4):
                            expected = int(completed[r, c]) if completed[r, c] == completed[u, v] else blank
                            _require(int(out[4*r + u, 4*c + v]) == expected, "wrong completed support mask")
            _require(np.count_nonzero(out != blank) == 64, "wrong foreground area")
            _require(all(np.count_nonzero(out == color) == 16 for color in symbols), "wrong category area")
            _require(np.array_equal(solve(inp.copy()), out), "solver disagreement")
            # Necessity: keep the correct selectors but omit completion in masks.
            original_masks = np.where(
                np.repeat(np.repeat(completed, 4, 0), 4, 1) == np.tile(inp, (4, 4)),
                np.repeat(np.repeat(completed, 4, 0), 4, 1), blank)
            _require(np.count_nonzero(original_masks != out) == 4 * count,
                     "completion must affect four mask copies per erased cell")
            uniform = np.repeat(np.repeat(completed, 4, 0), 4, 1)
            _require(np.count_nonzero(uniform != out) == 192, "support-mask mechanism is vacuous")
            if split == "train":
                easy |= count <= 3
                intersection |= _intersection_mask(mask)
                oriented |= not np.array_equal(completed, completed.T)
                train_kinds.add(_intercalate_count(completed))
                no_boxes |= any(len(np.unique(completed[r:r+2, c:c+2])) < 4 for r in (0, 2) for c in (0, 2))
                no_diagonal |= len(np.unique(np.diag(completed))) < 4
    _require(len(blanks) == 1, "blank must be shared within a puzzle")
    _require(easy and intersection and oriented, "missing teaching anchor")
    _require(train_kinds == {4, 12}, "demonstrate both order-four Latin types")
    _require(no_boxes and no_diagonal, "extra Sudoku/diagonal rules not ruled out")


def _canonical(grid):
    mapping = {}
    return tuple(mapping.setdefault(int(c), len(mapping)) for c in grid.flat)


def _signature(puzzle):
    data = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
            for s in ("train", "test")}
    return hashlib.sha256(json.dumps(data, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded, disk-free 42-seed sweep, including negative and metamorphic checks."""
    signatures, masks, structures, completed_structures = set(), set(), set(), set()
    blanks, hole_counts, palettes, kinds = set(), set(), set(), set()
    missing_rows = missing_cols = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        _require(_signature(generate_puzzle(seed)) == _signature(puzzle), "not deterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                square, blank = _complete(inp)
                mask = inp == blank
                masks.add(tuple(bool(v) for v in mask.flat))
                structures.add(_canonical(inp))
                completed_structures.add(_canonical(square))
                blanks.add(blank)
                hole_counts.add(int(mask.sum()))
                palettes.add(tuple(sorted(int(v) for v in np.unique(square))))
                kinds.add(_intercalate_count(square))
                missing_rows += int(np.any(np.all(mask, axis=1)))
                missing_cols += int(np.any(np.all(mask, axis=0)))
                # Recoloring, transpose, and quarter-turn preserve the rule.
                recolor = np.asarray([(c + 3) % 10 for c in range(10)], dtype=np.int8)
                _require(np.array_equal(solve(recolor[inp]), recolor[pair["output"]]), "recolor equivariance")
                _require(np.array_equal(solve(inp.T), pair["output"].T), "transpose equivariance")
                _require(np.array_equal(solve(np.rot90(inp)), np.rot90(pair["output"])), "rotation equivariance")
    _require(len(signatures) == 42 and len(masks) >= 100, "insufficient mask variation")
    _require(len(structures) >= 150 and len(completed_structures) >= 15, "insufficient geometry variation")
    _require(len(blanks) >= 8 and len(palettes) >= 30 and len(hole_counts) >= 7, "insufficient role/count variation")
    _require(kinds == {4, 12} and missing_rows and missing_cols, "missing Latin/boundary coverage")
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * tr + te, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for bad in (np.zeros((4, 4), dtype=int), np.ones((3, 3), dtype=int),
                np.asarray([[0, 0, 3, 4], [0, 0, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]])):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or ambiguous input accepted")
    # Corrupting an exceptional output must be noticed by the validator.
    broken = generate_puzzle(99, 2, 1)
    broken["train"][0]["output"][0, 0] = (int(broken["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("validator accepted a corrupt output")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "blank_colors": sorted(blanks),
            "blank_counts": sorted(hole_counts), "mask_layouts": len(masks),
            "normalized_inputs": len(structures), "normalized_completions": len(completed_structures),
            "symbol_palettes": len(palettes), "missing_rows": missing_rows,
            "missing_columns": missing_cols, "latin_intercalate_counts": sorted(kinds)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
