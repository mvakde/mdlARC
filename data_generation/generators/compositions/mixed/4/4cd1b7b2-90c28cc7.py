"""Recover a stretched 4x4 mosaic, then uniquely complete its Latin square.

The blank is the sole color repeated in a logical row or column. No color
identity, band length, palette, or Latin-square template is a solving code.
Only standard-library modules and NumPy are used; nothing is written to disk.
"""

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np


_PERMS = np.array(list(permutations(range(4))), dtype=np.int8)
_BASES = np.array(([[ (r + c) % 4 for c in range(4)] for r in range(4)],
                   [[r ^ c for c in range(4)] for r in range(4)]), dtype=np.int8)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array.astype(np.int8, copy=True)


def _collapse(grid):
    """Maximal runs of complete vectors, independently along both axes."""
    row_starts = np.r_[0, 1 + np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1))]
    col_starts = np.r_[0, 1 + np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0))]
    compact = grid[np.ix_(row_starts, col_starts)].copy()
    heights = np.diff(np.r_[row_starts, grid.shape[0]])
    widths = np.diff(np.r_[col_starts, grid.shape[1]])
    return compact, heights, widths


def _roles(compact):
    colors = sorted(map(int, np.unique(compact)))
    if compact.shape != (4, 4) or len(colors) != 5:
        raise ValueError("the collapsed mosaic must be 4x4 with five visible colors")
    repeated = [color for color in colors if
                np.any(np.sum(compact == color, axis=0) >= 2)
                or np.any(np.sum(compact == color, axis=1) >= 2)]
    if len(repeated) != 1:
        raise ValueError("the missing-entry marker must be the sole line-repeated color")
    blank = repeated[0]
    return blank, tuple(color for color in colors if color != blank)


def _complete(compact, blank, symbols):
    """Input-only MRV search; count at most two solutions, never break ties."""
    work = compact.copy()
    alphabet = set(symbols)
    rows, columns = [set() for _ in range(4)], [set() for _ in range(4)]
    for r in range(4):
        for c in range(4):
            value = int(work[r, c])
            if value == blank:
                continue
            if value not in alphabet or value in rows[r] or value in columns[c]:
                return 0, None
            rows[r].add(value)
            columns[c].add(value)
    count, first = 0, None

    def search():
        nonlocal count, first
        best, options = None, None
        for r in range(4):
            for c in range(4):
                if work[r, c] != blank:
                    continue
                available = alphabet - rows[r] - columns[c]
                if not available:
                    return
                if options is None or len(available) < len(options):
                    best, options = (r, c), available
        if best is None:
            count += 1
            if first is None:
                first = work.copy()
            return
        r, c = best
        for value in sorted(options):
            work[r, c] = value
            rows[r].add(value)
            columns[c].add(value)
            search()
            rows[r].remove(value)
            columns[c].remove(value)
            work[r, c] = blank
            if count == 2:
                return

    search()
    return count, first


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the compact, completed square using only the supplied pixels."""
    compact, _, _ = _collapse(_grid(input_grid))
    blank, symbols = _roles(compact)
    count, answer = _complete(compact, blank, symbols)
    if count != 1:
        raise ValueError("the recovered clues must have exactly one Latin completion")
    return answer


@lru_cache(maxsize=1)
def _all_latin():
    """Enumerate the 576 order-four Latin squares directly from constraints."""
    normalized = []

    def extend(rows):
        if len(rows) == 4:
            normalized.append(np.array(rows, dtype=np.int8))
            return
        for row in _PERMS:
            if all(not np.any(row == previous) for previous in rows):
                extend(rows + [row])

    extend([_PERMS[0]])
    result = np.concatenate([_PERMS[:, square] for square in normalized], axis=0)
    assert result.shape == (576, 4, 4)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=2)
def _mask_catalogue(base_index):
    """All locally readable, unique deletion masks for one Latin isotopy class.

    A deletion set is ambiguous iff it contains the complete difference set
    between this square and another Latin square. An upward Boolean closure
    finds every such superset in 16 bounded vectorized passes. This is a
    mathematical catalogue, not saved puzzle examples or execution traces.
    """
    base = _BASES[base_index]
    bits = np.left_shift(np.uint32(1), np.arange(16, dtype=np.uint32))
    differences = np.sum((_all_latin().reshape(-1, 16) != base.ravel()) * bits,
                         axis=1, dtype=np.uint32)
    ambiguous = np.zeros(65536, dtype=bool)
    ambiguous[differences[differences != 0]] = True
    for bit in range(16):
        blocks = ambiguous.reshape(-1, 2, 1 << bit)
        blocks[:, 1] |= blocks[:, 0]
    codes = np.arange(65536, dtype=np.uint32)
    masks = ((codes[:, None] & bits) != 0).reshape(-1, 4, 4)
    row_counts = masks.sum(axis=2, dtype=np.uint8)
    col_counts = masks.sum(axis=1, dtype=np.uint8)
    readable = (np.max(row_counts, axis=1) >= 2) | (np.max(col_counts, axis=1) >= 2)
    visible = np.ones(65536, dtype=bool)
    for symbol in range(4):
        visible &= np.any(~masks[:, base == symbol], axis=1)
    good = ~ambiguous & readable & visible
    features = {
        "codes": codes[good].astype(np.uint16),
        "holes": masks.sum(axis=(1, 2), dtype=np.uint8)[good],
        "intersection": (~np.any(row_counts == 1, axis=1)
                         & ~np.any(col_counts == 1, axis=1))[good],
        "row": np.any(row_counts == 4, axis=1)[good],
        "column": np.any(col_counts == 4, axis=1)[good],
    }
    for array in features.values():
        array.flags.writeable = False
    return features


def _sample_mask(rng, base_index, kind, low, high, exclude=None):
    catalogue = _mask_catalogue(base_index)
    allowed = (catalogue["holes"] >= low) & (catalogue["holes"] <= high)
    if exclude is not None:
        allowed &= catalogue["holes"] != exclude
    if kind in ("intersection", "row", "column"):
        allowed &= catalogue[kind]
    counts = np.unique(catalogue["holes"][allowed])
    if not len(counts):
        raise RuntimeError("empty mask plan")
    target = int(rng.choice(counts))
    indices = np.flatnonzero(allowed & (catalogue["holes"] == target))
    code = int(catalogue["codes"][int(rng.choice(indices))])
    return ((code & (1 << np.arange(16))) != 0).reshape(4, 4)


def _adjacent_blank(mask):
    return bool(np.any(mask[:, 1:] & mask[:, :-1])
                or np.any(mask[1:] & mask[:-1]))


def _block_latin(square):
    return all(len(np.unique(square[r:r + 2, c:c + 2])) == 4
               for r in (0, 2) for c in (0, 2))


def _orient(rng, base_index, mask, witness):
    base = _BASES[base_index]
    if not witness:
        rows, columns = _PERMS[int(rng.integers(24))], _PERMS[int(rng.integers(24))]
        return base[np.ix_(rows, columns)].copy(), mask[np.ix_(rows, columns)].copy()
    # Choose among all 576 geometries, with two concrete demonstration goals:
    # a locally invisible blank seam and an output that refutes Sudoku blocks.
    candidates = []
    for ri, rows in enumerate(_PERMS):
        for ci, columns in enumerate(_PERMS):
            oriented = base[np.ix_(rows, columns)]
            holes = mask[np.ix_(rows, columns)]
            if _adjacent_blank(holes) and not _block_latin(oriented):
                candidates.append((ri, ci))
    if not candidates:
        raise RuntimeError("no orientation exposes the required evidence")
    ri, ci = candidates[int(rng.integers(len(candidates)))]
    rows, columns = _PERMS[ri], _PERMS[ci]
    return base[np.ix_(rows, columns)].copy(), mask[np.ix_(rows, columns)].copy()


def _uneven_axis(rng, witness=False):
    bands = rng.integers(1, 7, size=4)
    if np.all(bands == bands[0]):
        bands[0] = 1 if bands[0] != 1 else 2
    if witness and bands.sum() % 4 == 0:
        changes = [(i, value) for i in range(4) for value in range(1, 7)
                   if value != bands[i] and (int(bands.sum()) - int(bands[i]) + value) % 4
                   and len(set(map(int, np.r_[bands[:i], value, bands[i + 1:]]))) > 1]
        i, value = changes[int(rng.integers(len(changes)))]
        bands[i] = value
    return bands


def _bands(rng, profile):
    if profile == "witness":
        return _uneven_axis(rng, True), _uneven_axis(rng, True)
    if profile == "strong":
        axes = []
        for _ in range(2):
            bands = rng.integers(1, 7, size=4)
            peak, thin = rng.choice(4, size=2, replace=False)
            bands[peak], bands[thin] = int(rng.integers(8, 18)), 1
            axes.append(bands)
        return tuple(axes)
    mode = int(rng.integers(4))
    rows = np.full(4, int(rng.integers(2, 7))) if mode in (0, 1) else _uneven_axis(rng)
    columns = np.full(4, int(rng.integers(2, 7))) if mode in (0, 2) else _uneven_axis(rng)
    return rows, columns


def _intercalates(square):
    return sum(len(np.unique(square[np.ix_(rows, columns)])) == 2
               for rows in combinations(range(4), 2) for columns in combinations(range(4), 2))


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    shift, color_mode = int(rng.integers(3)), int(rng.integers(3))
    shared_palette = rng.choice(10, size=5, replace=False)
    teaching_order = rng.permutation(num_train)
    witness_index, companion_index = map(int, teaching_order[:2])
    witness_base = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    witness_holes = None
    # Build the two teaching roles first, then restore the shuffled display order.
    order = [("train", int(i)) for i in teaching_order]
    order += [("test", i) for i in range(num_test)]
    generated = {}
    for split, index in order:
        witness = split == "train" and index == witness_index
        companion = split == "train" and index == companion_index
        base_index = witness_base if witness else 1 - witness_base if companion else int(rng.integers(2))
        if witness:
            low, high = ((4, 9), (4, 7), (8, 9))[shift]
            kind, profile = "intersection", "witness"
        elif split == "test":
            low, high = ((2, 9), (8, 11), (8, 11))[shift]
            kind = str(rng.choice(("random", "random", "row", "column"))) if low >= 4 else "random"
            profile = "normal" if shift == 1 else "strong"
        else:
            low, high = (2, 7) if shift == 1 else (2, 9)
            profile = "normal"
            if shift == 2:
                if companion or bool(rng.integers(2)):
                    low, high, profile = 2, 6, "strong"
                else:
                    low, high = 8, 10
            kind = "random"
            if int(rng.integers(3)) == 0:
                low = max(4, low)
                kind = str(rng.choice(("row", "column")))
        exclude = witness_holes if companion else None
        mask = _sample_mask(rng, base_index, kind, low, high, exclude)
        if witness:
            witness_holes = int(mask.sum())
        square, mask = _orient(rng, base_index, mask, witness)
        rows, columns = _bands(rng, profile)
        for attempt in range(32):
            palette = shared_palette.copy() if color_mode == 0 else rng.choice(10, size=5, replace=False)
            if color_mode == 1:
                blank = int(shared_palette[0])
                palette = np.r_[blank, rng.choice([c for c in range(10) if c != blank], size=4, replace=False)]
            blank, symbols = int(palette[0]), palette[1:]
            # Symbol permutations are independent of geometric permutations.
            output = symbols[rng.permutation(4)[square]].astype(np.int8)
            compact = output.copy()
            compact[mask] = blank
            expanded = np.repeat(np.repeat(compact, rows, axis=0), columns, axis=1)
            key = _signature(expanded)
            if key not in seen:
                seen.add(key)
                generated[split, index] = {"input": expanded, "output": output}
                break
            rows, columns = _bands(rng, profile)
        else:
            raise RuntimeError("unable to construct a distinct input in 32 attempts")
    for split, count in (("train", num_train), ("test", num_test)):
        puzzle[split] = [generated[split, i] for i in range(count)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the mathematical family and the promised demonstration evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, classes, train_counts = set(), set(), set()
    witness_found = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, output = _grid(pair["input"]), _grid(pair["output"])
            assert output.shape == (4, 4) and min(source.shape) > 4
            key = _signature(source)
            assert key not in seen
            seen.add(key)
            compact, rows, columns = _collapse(source)
            blank, symbols = _roles(compact)
            assert set(map(int, np.unique(output))) == set(symbols)
            assert all(set(map(int, line)) == set(symbols) for line in list(output) + list(output.T))
            mask = compact == blank
            assert 2 <= int(mask.sum()) <= 11
            assert np.array_equal(compact[~mask], output[~mask])
            assert np.array_equal(compact != output, mask)
            rebuilt = np.repeat(np.repeat(compact, rows, axis=0), columns, axis=1)
            assert np.array_equal(source, rebuilt)
            assert np.array_equal(solve(source), output)
            # Cross-check uniqueness by exhaustive enumeration, a different
            # algorithm from solve's MRV search and from mask closure.
            encoded = np.full((4, 4), -1, dtype=np.int8)
            for value, symbol in enumerate(symbols):
                encoded[compact == symbol] = value
            matches = np.all((_all_latin() == encoded) | mask, axis=(1, 2))
            assert np.count_nonzero(matches) == 1
            assert np.array_equal(np.array(symbols)[_all_latin()[matches][0]], output)
            # Both parent ablations change every answer: B is needed for size,
            # A for cells. Each one-axis-only collapse also has the wrong size.
            assert not np.array_equal(compact, output)
            assert int(rows.sum()) > 4 and int(columns.sum()) > 4
            if split == "train":
                classes.add(_intercalates(output))
                train_counts.add(int(mask.sum()))
                no_single = not np.any(mask.sum(axis=0) == 1) and not np.any(mask.sum(axis=1) == 1)
                witness_found |= (no_single and _adjacent_blank(mask) and not _block_latin(output)
                                  and len(set(rows)) > 1 and len(set(columns)) > 1
                                  and source.shape[0] % 4 != 0 and source.shape[1] % 4 != 0)
    assert classes == {4, 12}, "training must show both Latin isotopy classes"
    assert len(train_counts) >= 2 and witness_found


def _puzzle_signature(puzzle):
    return tuple(_signature(np.asarray(pair[key])) for split in ("train", "test")
                 for pair in puzzle[split] for key in ("input", "output"))


def self_test():
    """Bounded in-memory checks across 42 seeds, counts, and semantic edges."""
    signatures, shapes, masks, palettes, structures, profiles, holes = (set() for _ in range(7))
    marker_colors, symbol_colors, classes = set(), set(), set()
    missing_rows = missing_columns = strong = thin = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, output = pair["input"], pair["output"]
                saved = source.copy()
                assert np.array_equal(solve(source), output) and np.array_equal(source, saved)
                compact, rows, columns = _collapse(source)
                blank, symbols = _roles(compact)
                mask = compact == blank
                shapes.add(source.shape)
                masks.add(mask.tobytes())
                palettes.add((blank, symbols))
                mapping = {int(color): i for i, color in enumerate(output[0])}
                structures.add(tuple(mapping[int(color)] for color in output.flat))
                profiles.add((tuple(map(int, rows)), tuple(map(int, columns))))
                holes.add(int(mask.sum()))
                marker_colors.add(blank)
                symbol_colors.update(symbols)
                classes.add(_intercalates(output))
                missing_rows += bool(np.any(mask.sum(axis=1) == 4))
                missing_columns += bool(np.any(mask.sum(axis=0) == 4))
                strong += max(max(rows), max(columns)) >= 8
                thin += 1 in rows or 1 in columns
    assert len(signatures) == 42 and len(shapes) >= 60
    assert len(masks) >= 120 and len(palettes) >= 40 and len(structures) >= 18
    assert len(profiles) >= 100 and holes == set(range(2, 12))
    assert marker_colors == symbol_colors == set(range(10)) and classes == {4, 12}
    assert min(missing_rows, missing_columns, strong, thin) >= 10
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(10000 + train * 10 + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for invalid_seed in (-1, True, 1.5, None):
        try:
            generate_puzzle(invalid_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Visible colors but a two-symbol trade: do not select an arbitrary answer.
    ambiguous = np.array([[0, 0, 3, 4], [0, 0, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]])
    try:
        solve(np.repeat(np.repeat(ambiguous, 2, axis=0), 3, axis=1))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous Latin clues accepted")
    # A uniform blank row is retained; it is not removable background.
    compact = np.array([[0, 0, 0, 0], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
    expected = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]])
    assert np.array_equal(solve(np.repeat(np.repeat(compact, [1, 2, 3, 4], axis=0),
                                        [4, 1, 3, 2], axis=1)), expected)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, semantic edge cases)")
