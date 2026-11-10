"""Uniquely complete an order-four Latin square, then shear its intact rows.

All color roles are recoverable from the input. This module performs no I/O.
The generated training curriculum is narrower than the solver's input domain.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 785603539, 538278068, 150897480, 492942188, 469139457,
              150022496, 932957810, 408391068, 302217804, 338542080, 73615504,
              684838871, 173455730, 732055141, 952066049, 813157839, 28955135,
              591567553, 322342932, 911386697, 356976464, 576033301, 706317895,
              777400846, 746591741, 259652293, 578838602, 748450268, 843505453,
              222672090, 142393260, 145871104, 283008016, 10834989, 997137029,
              377201509, 537746953, 664137537, 265720446, 160440529, 699268137)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value, shape=None):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a rectangular integer ARC grid")
    if shape is not None and grid.shape != shape:
        raise ValueError(f"expected grid shape {shape}")
    return grid.astype(np.int8, copy=True)


def _roles(grid):
    """Only the missing-entry color may repeat in a single input line."""
    colors = tuple(int(v) for v in np.unique(grid))
    if len(colors) != 5:
        raise ValueError("input must expose four symbols and one blank color")
    repeated = [color for color in colors if any(
        np.count_nonzero(line == color) >= 2 for line in list(grid) + list(grid.T))]
    if len(repeated) != 1:
        raise ValueError("blank must be the sole color repeated within a line")
    blank = repeated[0]
    return blank, tuple(color for color in colors if color != blank)


def _completions(grid, blank, symbols, limit=2):
    """Count completions up to limit; search uses only the supplied clues."""
    symbols = tuple(sorted(int(v) for v in symbols))
    allowed = set(symbols)
    work = grid.copy()
    row_used, col_used = [set() for _ in range(4)], [set() for _ in range(4)]
    for r in range(4):
        for c in range(4):
            value = int(work[r, c])
            if value == blank:
                continue
            if value not in allowed or value in row_used[r] or value in col_used[c]:
                return 0, None
            row_used[r].add(value)
            col_used[c].add(value)
    count, first = 0, None

    def search():
        nonlocal count, first
        cell, candidates = None, None
        for r in range(4):
            for c in range(4):
                if int(work[r, c]) != blank:
                    continue
                available = [v for v in symbols if v not in row_used[r] and v not in col_used[c]]
                if not available:
                    return
                if candidates is None or len(available) < len(candidates):
                    cell, candidates = (r, c), available
        if cell is None:
            count += 1
            if first is None:
                first = work.copy()
            return
        r, c = cell
        for value in candidates:
            work[r, c] = value
            row_used[r].add(value)
            col_used[c].add(value)
            search()
            row_used[r].remove(value)
            col_used[c].remove(value)
            work[r, c] = blank
            if count >= limit:
                break

    search()
    return count, first


def _complete(input_grid):
    grid = _grid(input_grid, (4, 4))
    blank, symbols = _roles(grid)
    count, completed = _completions(grid, blank, symbols)
    if count != 1:
        raise ValueError("clues must have exactly one Latin completion")
    return completed, blank


def _shear(square, blank):
    out = np.full((4, 7), blank, dtype=np.int8)
    for r in range(4):
        out[r, 3-r:7-r] = square[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete clues before placing (r,c) at (r,c+3-r), without clipping."""
    square, blank = _complete(input_grid)
    return _shear(square, blank)


def _latin(rng, kind):
    # These are the two order-four isotopy classes, not arithmetic color rules.
    base = np.array([[(r+c) % 4 if kind == 0 else r ^ c
                      for c in range(4)] for r in range(4)], dtype=np.int8)
    base = base[np.ix_(rng.permutation(4), rng.permutation(4))]
    return rng.permutation(4)[base].astype(np.int8)


def _recoverable(square, mask):
    if len(np.unique(square[~mask])) != 4:
        return False
    if not (np.any(np.sum(mask, axis=0) >= 2) or np.any(np.sum(mask, axis=1) >= 2)):
        return False
    clues = square.copy()
    clues[mask] = -1
    count, answer = _completions(clues, -1, range(4))
    return count == 1 and np.array_equal(answer, square)


def _rectangle_mask(rng, square):
    candidates = []
    for rows in combinations(range(4), 2):
        for cols in combinations(range(4), 2):
            mask = np.zeros((4, 4), dtype=bool)
            mask[np.ix_(rows, cols)] = True
            if _recoverable(square, mask):
                candidates.append(mask)
    if not candidates:
        raise RuntimeError("no uniquely completable rectangle")
    return candidates[int(rng.integers(len(candidates)))].copy()


def _mask(rng, square, low, high, bias):
    """Bounded uniqueness-preserving deletion, with random axis/cluster bias."""
    target = int(rng.integers(low, high+1))
    for _ in range(48):
        mask = np.zeros((4, 4), dtype=bool)
        priorities = rng.random((4, 4))
        focus = int(rng.integers(4))
        style = int(rng.integers(4))
        if style == 0:
            if bias == 0:
                priorities[focus, :] -= 0.75
            else:
                priorities[:, focus] -= 0.75
        elif style == 1:
            rr, cc = rng.integers(4, size=2)
            rows, cols = np.indices((4, 4))
            priorities += 0.20 * (abs(rows-rr)+abs(cols-cc))
        for flat in np.argsort(priorities, axis=None):
            r, c = divmod(int(flat), 4)
            mask[r, c] = True
            # During deletion the role can become identifiable only later.
            trial = square.copy()
            trial[mask] = -1
            count, _ = _completions(trial, -1, range(4))
            if count != 1 or len(np.unique(square[~mask])) != 4:
                mask[r, c] = False
            if np.count_nonzero(mask) == target:
                break
        # Reaching the sampled target is optional, but the requested range is not.
        if low <= np.count_nonzero(mask) <= high and _recoverable(square, mask):
            return mask
    raise RuntimeError("bounded mask construction failed")


def _key(grid):
    return tuple(int(v) for v in grid.flat)


def _sudoku_blocks(square):
    return all(len(np.unique(square[r:r+2, c:c+2])) == 4
               for r in (0, 2) for c in (0, 2))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shared_marker = bool(rng.integers(2))
    common_blank = int(rng.integers(10))
    bias = int(rng.integers(2))
    first_class = int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(32):
                kind = (first_class + index) % 2 if section == "train" and index < 2 else int(rng.integers(2))
                square = _latin(rng, kind)
                if section == "train" and index == 0 and _sudoku_blocks(square):
                    continue  # Explicitly distinguish Latin completion from Sudoku.
                if section == "train" and index == 0:
                    mask = _rectangle_mask(rng, square)
                elif section == "train":
                    low = 5 if index == 1 else 2
                    mask = _mask(rng, square, low, 7, bias)
                else:
                    mask = _mask(rng, square, 8, 11, 1-bias)
                blank = common_blank if shared_marker else int(rng.integers(10))
                if not shared_marker and section == "train" and index == 1:
                    # Show that numeric marker identity is irrelevant in this regime.
                    previous_blank = _roles(puzzle["train"][0]["input"])[0]
                    blank = int(rng.choice([v for v in range(10) if v != previous_blank]))
                symbols = rng.choice([v for v in range(10) if v != blank], size=4, replace=False)
                completed = symbols[square].astype(np.int8)
                inp = completed.copy()
                inp[mask] = blank
                key = _key(inp)
                if key in seen:
                    continue
                seen.add(key)
                puzzle[section].append({"input": inp, "output": _shear(completed, blank)})
                break
            else:
                raise RuntimeError("unable to make a distinct input")
    # Shuffle demonstration order without changing their collective guarantees.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _unshear(out):
    return np.stack([out[r, 3-r:7-r] for r in range(4)])


def _rectangle(mask):
    return (int(np.sum(mask)) == 4
            and sorted(np.sum(mask, axis=0).tolist()) == [0, 0, 2, 2]
            and sorted(np.sum(mask, axis=1).tolist()) == [0, 0, 2, 2])


def _intercalates(square):
    # A palette-independent distinction between the two Latin structure classes.
    return sum(len(np.unique(square[np.ix_(rr, cc)])) == 2
               for rr in combinations(range(4), 2) for cc in combinations(range(4), 2))


def _line_passes(mask):
    mask = mask.copy()
    passes = 0
    while np.any(mask):
        singles = mask & ((np.sum(mask, axis=1)[:, None] == 1)
                          | (np.sum(mask, axis=0)[None, :] == 1))
        if not np.any(singles):
            return None
        mask[singles] = False
        passes += 1
    return passes


def validate_generated_puzzle(puzzle):
    """Check rule invariants and the generated curriculum, without hidden state."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_classes, train_holes, test_holes = set(), set(), [], []
    rectangle_demo = non_sudoku_demo = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"], (4, 4)), _grid(pair["output"], (4, 7))
            blank, symbols = _roles(inp)
            mask = inp == blank
            completed = _unshear(out)
            assert np.array_equal(inp[~mask], completed[~mask]), "clue changed"
            assert not np.any(completed == blank), "an internal hole remains"
            for line in list(completed) + list(completed.T):
                assert set(int(v) for v in line) == set(symbols), "not Latin"
            for r in range(4):
                assert np.all(out[r, :3-r] == blank)
                assert np.all(out[r, 7-r:] == blank)
                assert np.array_equal(out[r, 3-r:7-r], completed[r])
            count, answer = _completions(inp, blank, symbols)
            assert count == 1 and np.array_equal(answer, completed)
            assert np.array_equal(solve(inp), out)
            # A shear without completion has blanks inside its four-cell vectors.
            assert not np.array_equal(_shear(inp, blank), out), "B contributes nothing"
            # Completion without A gives a 4x4 square instead of the required canvas.
            assert completed.shape != out.shape and np.count_nonzero(out == blank) == 12
            assert all(np.count_nonzero(out == s) == 4 for s in symbols)
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            holes = int(np.sum(mask))
            if section == "train":
                assert 2 <= holes <= 7
                train_holes.append(holes)
                train_classes.add(_intercalates(completed))
                rectangle_demo |= _rectangle(mask)
                non_sudoku_demo |= not _sudoku_blocks(completed)
            else:
                assert 8 <= holes <= 11
                test_holes.append(holes)
    assert rectangle_demo and non_sudoku_demo and train_classes == {4, 12}
    assert max(train_holes) >= 5 and min(test_holes) > max(train_holes)


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """42 specified seeds, bounded aggregate diversity and rejection checks."""
    masks, canonical_squares, palettes, markers, holes, signatures = (set() for _ in range(6))
    full_rows = full_cols = intersection_cases = long_chains = changing_markers = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        local_markers = set()
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            blank, symbols = _roles(inp)
            local_markers.add(blank)
            markers.add(blank)
            mask = inp == blank
            square = _unshear(out)
            masks.add(tuple(bool(v) for v in mask.flat))
            mapping = {int(v): i for i, v in enumerate(square[0])}
            canonical_squares.add(tuple(mapping[int(v)] for v in square.flat))
            palettes.add((blank, symbols))
            holes.add(int(np.sum(mask)))
            full_rows += int(np.any(np.sum(mask, axis=1) == 4))
            full_cols += int(np.any(np.sum(mask, axis=0) == 4))
            passes = _line_passes(mask)
            intersection_cases += int(passes is None)
            long_chains += int(passes is not None and passes >= 3)
            original = inp.copy()
            assert np.array_equal(solve(inp), out) and np.array_equal(inp, original)
            # Coordinate and equality reasoning must survive unseen color names
            # and transposed clue geometry. These do not change the shear direction.
            permutation = np.random.default_rng(seed + 2718).permutation(10)
            assert np.array_equal(solve(permutation[inp]), permutation[out])
            assert np.array_equal(solve(inp.T), _shear(square.T, blank))
        changing_markers += len(local_markers) > 1
    assert len(signatures) == 42 and len(masks) >= 100
    assert len(canonical_squares) >= 20 and len(palettes) >= 150
    assert markers == set(range(10)) and set(range(2, 11)) <= holes
    assert full_rows >= 5 and full_cols >= 5 and intersection_cases >= 42
    assert long_chains >= 5 and changing_markers >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1234567, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Invalid inputs must not quietly choose an arbitrary completion or role.
    ambiguous = np.array([[0, 0, 3, 4], [0, 0, 4, 3],
                          [3, 4, 1, 2], [4, 3, 2, 1]])
    for bad in (np.zeros((4, 4), dtype=int), np.zeros((3, 3), dtype=int),
                ambiguous, ambiguous.astype(float)):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
