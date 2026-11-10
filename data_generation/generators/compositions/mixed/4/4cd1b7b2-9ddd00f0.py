"""Latin holes: an order-four Latin square encoded by negative-space tiles.

Only Python's standard library and NumPy are used. All search is in memory;
generation, validation, and self_test write no files. Running this module prints
one compact JSON metrics record for its bounded 42-seed self-test.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, permutations
import json

import numpy as np


def _latin_catalog() -> np.ndarray:
    """Enumerate the entire mathematical domain, without sampled examples."""
    rows = tuple(permutations(range(4)))
    squares = []

    def extend(prefix):
        if len(prefix) == 4:
            squares.append(tuple(prefix))
            return
        for row in rows:
            if all(all(row[c] != previous[c] for c in range(4))
                   for previous in prefix):
                extend(prefix + [row])

    extend([])
    catalog = np.asarray(squares, dtype=np.int8)
    catalog.flags.writeable = False
    return catalog


_LATINS = _latin_catalog()
_SEPARATORS = (4, 9, 14)
_PAIRS = tuple(combinations(range(4), 2))


def _integer(value, name, low, high=None) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its allowed range")
    return value


def _grid(value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape != (19, 19)
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError("expected a 19x19 integer compartment grid")
    if np.any((array < 0) | (array > 9)):
        raise ValueError("colors must be in 0..9")
    return array


def _decode(value):
    """Recover every semantic role from this grid alone."""
    grid = _grid(value)
    background = int(grid[4, 0])
    for index in _SEPARATORS:
        if (not np.all(grid[index, :] == background)
                or not np.all(grid[:, index] == background)):
            raise ValueError("damaged background separators")
    other_colors = set(int(x) for x in np.unique(grid)) - {background}
    if len(other_colors) != 1:
        raise ValueError("expected exactly one foreground color")
    foreground = next(iter(other_colors))
    coordinates = {}
    for r in range(4):
        for c in range(4):
            tile = grid[5*r:5*r+4, 5*c:5*c+4]
            holes = np.argwhere(tile == background)
            if len(holes) == 16:
                continue
            if len(holes) != 1 or np.count_nonzero(tile == foreground) != 15:
                raise ValueError("a tile must be erased or have exactly one hole")
            coordinates[r, c] = tuple(int(x) for x in holes[0])
    alphabet = tuple(sorted(set(coordinates.values())))
    if len(alphabet) != 4:
        raise ValueError("all four distinct hole coordinates must be visible")
    labels = {position: i for i, position in enumerate(alphabet)}
    macro = np.full((4, 4), -1, dtype=np.int8)
    for position, hole in coordinates.items():
        macro[position] = labels[hole]
    return grid, macro, alphabet, background, foreground


def _matches(macro) -> np.ndarray:
    return np.flatnonzero(np.all((macro < 0) | (_LATINS == macro), axis=(1, 2)))


def _render(square, alphabet, background, foreground, erased=None) -> np.ndarray:
    grid = np.full((19, 19), background, dtype=np.int8)
    for r in range(4):
        for c in range(4):
            if erased is not None and erased[r, c]:
                continue
            r0, c0 = 5*r, 5*c
            grid[r0:r0+4, c0:c0+4] = foreground
            u, v = alphabet[int(square[r, c])]
            grid[r0+u, c0+v] = background
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete any valid family input, with no generator state or training data."""
    _, macro, alphabet, background, foreground = _decode(input_grid)
    matches = _matches(macro)
    if len(matches) != 1:
        raise ValueError("hole clues must have exactly one Latin completion")
    return _render(_LATINS[int(matches[0])], alphabet, background, foreground)


def _unique(square, erased) -> bool:
    if len(set(int(x) for x in square[~erased])) != 4:
        return False
    clues = square.copy()
    clues[erased] = -1
    return len(_matches(clues)) == 1


def _rectangle_mask(rng, square):
    candidates = []
    for rows in _PAIRS:
        for columns in _PAIRS:
            mask = np.zeros((4, 4), dtype=bool)
            mask[np.ix_(rows, columns)] = True
            if _unique(square, mask):
                candidates.append(mask)
    if not candidates:
        raise RuntimeError("no joint-constraint rectangle exists")
    return candidates[int(rng.integers(len(candidates)))].copy()


def _erasure_mask(rng, square, kind, target, axis):
    """At most 12 x 16 deletions; retain only uniquely completable candidates."""
    if kind == "rectangle" and target == 4:
        return _rectangle_mask(rng, square)
    best = None
    for _ in range(12):
        mask = np.zeros((4, 4), dtype=bool)
        if kind == "line":
            index = int(rng.integers(4))
            if axis == "row":
                mask[index, :] = True
            else:
                mask[:, index] = True
        elif kind == "rectangle":
            mask = _rectangle_mask(rng, square)
        priority = rng.random((4, 4))
        if kind in ("row", "column"):
            line = int(rng.integers(4))
            if kind == "row":
                priority[line, :] -= 1
            else:
                priority[:, line] -= 1
        elif kind == "cluster":
            center = rng.integers(4, size=2)
            for r in range(4):
                for c in range(4):
                    priority[r, c] += 0.4 * (abs(r-center[0]) + abs(c-center[1]))
        for index in np.argsort(priority, axis=None):
            if int(mask.sum()) >= target:
                break
            r, c = divmod(int(index), 4)
            if mask[r, c]:
                continue
            mask[r, c] = True
            if not _unique(square, mask):
                mask[r, c] = False
        if best is None or int(mask.sum()) > int(best.sum()):
            best = mask.copy()
        if int(best.sum()) == target:
            return best
    # A target of eleven may saturate at ten. This does not change the rule
    # or the guaranteed density shift (test targets and accepted counts >= 8).
    if int(best.sum()) < min(target, 8):
        raise RuntimeError("bounded erasure search could not reach its budget")
    return best


def _sample_alphabet(rng):
    mode = int(rng.integers(3))
    if mode == 0:
        rows = rng.choice(4, size=2, replace=False)
        columns = rng.choice(4, size=2, replace=False)
        return tuple(sorted((int(r), int(c)) for r in rows for c in columns))
    if mode == 1:
        return tuple((r, int(c)) for r, c in enumerate(rng.permutation(4)))
    for _ in range(32):
        positions = tuple(sorted(divmod(int(x), 4)
                                 for x in rng.choice(16, size=4, replace=False)))
        if len({p[0] for p in positions}) >= 2 and len({p[1] for p in positions}) >= 2:
            return positions
    return ((0, 0), (0, 3), (2, 1), (3, 2))


def _key(grid):
    return np.asarray(grid, dtype=np.int8).tobytes()


def _macro_coordinate_shortcut_wrong(square, alphabet, erased) -> bool:
    return any(alphabet[int(square[r, c])] != (r, c)
               for r, c in np.argwhere(erased))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared_background = int(rng.integers(10)) if rng.random() < 0.5 else None
    shared_alphabet = _sample_alphabet(rng) if rng.random() < 0.5 else None
    axis = ("row", "column")[int(rng.integers(2))]
    train_budget = int(rng.integers(5, 8))
    train_plan = [("rectangle", 4), ("line", int(rng.integers(4, train_budget+1)))]
    kinds = ("scattered", "row", "column", "cluster", "rectangle", "line")
    for _ in range(num_train-2):
        kind = kinds[int(rng.integers(len(kinds)))]
        low = 4 if kind in ("line", "rectangle") else 1
        train_plan.append((kind, int(rng.integers(low, train_budget+1))))
    test_plan = [(kinds[int(rng.integers(4))], int(rng.integers(8, 12)))
                 for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    used = set()
    previous_foreground = None
    previous_alphabet = None
    for split, plan in (("train", train_plan), ("test", test_plan)):
        for index, (kind, target) in enumerate(plan):
            background = (int(rng.integers(10)) if shared_background is None
                          else shared_background)
            colors = [x for x in range(10) if x != background]
            if split == "train" and index == 1 and previous_foreground in colors:
                colors.remove(previous_foreground)
            foreground = int(rng.choice(colors))
            for _ in range(32):
                square = _LATINS[int(rng.integers(len(_LATINS)))].copy()
                alphabet = shared_alphabet or _sample_alphabet(rng)
                if (shared_alphabet is None and split == "train" and index == 1
                        and alphabet == previous_alphabet):
                    continue
                erased = _erasure_mask(rng, square, kind, target, axis)
                if not _macro_coordinate_shortcut_wrong(square, alphabet, erased):
                    continue
                inp = _render(square, alphabet, background, foreground, erased)
                key = _key(inp)
                if key in used:
                    continue
                used.add(key)
                out = _render(square, alphabet, background, foreground)
                puzzle[split].append({"input": inp, "output": out})
                previous_foreground, previous_alphabet = foreground, alphabet
                break
            else:
                raise RuntimeError("bounded search failed to construct a distinct pair")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _latin(square):
    return all(set(int(x) for x in line) == {0, 1, 2, 3}
               for line in list(square) + list(square.T))


def _rectangle(mask):
    return (int(mask.sum()) == 4
            and sorted(int(x) for x in mask.sum(axis=0)) == [0, 0, 2, 2]
            and sorted(int(x) for x in mask.sum(axis=1)) == [0, 0, 2, 2])


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have exactly input/output keys")
    inp, macro, alphabet, background, foreground = _decode(pair["input"])
    out, completed, output_alphabet, out_bg, out_fg = _decode(pair["output"])
    erased = macro < 0
    _require(1 <= int(erased.sum()) <= 11, "generated erasure count must be 1..11")
    _require((background, foreground, alphabet) == (out_bg, out_fg, output_alphabet),
             "output changed the input's visible symbol/color roles")
    _require(len({p[0] for p in alphabet}) >= 2 and len({p[1] for p in alphabet}) >= 2,
             "generated alphabet must span both local axes")
    _require(_latin(completed), "output hole coordinates are not a Latin square")
    _require(np.array_equal(macro[~erased], completed[~erased]), "clues changed")
    matches = _matches(macro)
    _require(len(matches) == 1, "clues are not uniquely completable")
    _require(np.array_equal(completed, _LATINS[int(matches[0])]), "wrong completion")
    expected = _render(completed, alphabet, background, foreground)
    _require(np.array_equal(out, expected), "output has incorrect hole rendering")
    changed = inp != out
    _require(int(changed.sum()) == 15 * int(erased.sum()), "wrong change count")
    _require(np.all(inp[changed] == background) and np.all(out[changed] == foreground),
             "changes must only add foreground in erased compartments")
    _require(np.count_nonzero(out == foreground) == 240, "expected 15 cells per tile")
    return {"macro": macro, "completed": completed, "alphabet": alphabet,
            "background": background, "foreground": foreground, "erased": erased}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train <= 10 and 1 <= test <= 4 and train > test, "invalid counts")
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _require(any(_rectangle(info["erased"]) for info in infos["train"]),
             "training needs a joint-constraint rectangle")
    _require(any(np.any(info["erased"].all(axis=0)) or np.any(info["erased"].all(axis=1))
                 for info in infos["train"]), "training needs an erased macro line")
    _require(all(int(info["erased"].sum()) <= 7 for info in infos["train"]),
             "training must use at most seven erasures")
    _require(all(int(info["erased"].sum()) >= 8 for info in infos["test"]),
             "tests must use at least eight erasures")
    _require(any(_macro_coordinate_shortcut_wrong(info["completed"], info["alphabet"],
                                                 info["erased"])
                 for info in infos["train"]), "training fails to refute macro-coordinate copying")


def _digest(grid):
    return sha256(_key(grid)).hexdigest()


def _intercalates(square):
    return sum(len(set(int(x) for x in square[np.ix_(rs, cs)].flat)) == 2
               for rs in _PAIRS for cs in _PAIRS)


def _single_axis_counterexamples(info):
    """Actual alternate tile answers that expose the need for each Latin axis."""
    full, mask = info["completed"], info["erased"]
    for transpose in (False, True):
        square = full.T.copy() if transpose else full.copy()
        blanks = mask.T if transpose else mask
        r = next(r for r in range(4) if int(blanks[r].sum()) >= 2)
        c1, c2 = np.flatnonzero(blanks[r])[:2]
        square[r, c1], square[r, c2] = square[r, c2], square[r, c1]
        _require(all(len(set(int(x) for x in row)) == 4 for row in square),
                 "single-axis alternate lost its preserved constraint")
        _require(not _latin(square), "single-axis alternate unexpectedly remained Latin")
        alternate = square.T if transpose else square
        _require(np.array_equal(alternate[~mask], full[~mask]), "alternate changed a clue")
        wrong = _render(alternate, info["alphabet"], info["background"], info["foreground"])
        right = _render(full, info["alphabet"], info["background"], info["foreground"])
        _require(not np.array_equal(wrong, right), "omitting an axis did not change the answer")


def self_test() -> dict:
    """Bounded streaming checks over 42 seeds, all explicit counts, and bad inputs."""
    _require(_LATINS.shape == (576, 4, 4) and all(_latin(s) for s in _LATINS),
             "Latin domain is incomplete or malformed")
    alphabets, masks, completed_shapes, palettes, classes, puzzles = (set() for _ in range(6))
    erasure_counts = {"train": set(), "test": set()}
    full_lines = {"row": 0, "column": 0}
    rectangles = ablations = pair_count = non_sudoku = non_diagonal = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_hash = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                macro, full, mask = info["macro"], info["completed"], info["erased"]
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                _require(np.array_equal(solve(pair["output"]), pair["output"]), "fixed-point failure")
                pair_count += 1
                puzzle_hash.update(_key(pair["input"]))
                puzzle_hash.update(_key(pair["output"]))
                alphabets.add(info["alphabet"])
                masks.add(mask.tobytes())
                completed_shapes.add(full.tobytes())
                palettes.add((info["background"], info["foreground"]))
                classes.add(_intercalates(full))
                erasure_counts[split].add(int(mask.sum()))
                full_lines["row"] += int(np.any(mask.all(axis=1)))
                full_lines["column"] += int(np.any(mask.all(axis=0)))
                non_sudoku += int(any(len(set(int(x) for x in full[r:r+2, c:c+2].flat)) < 4
                                      for r in (0, 2) for c in (0, 2)))
                non_diagonal += int(len(set(int(x) for x in np.diag(full))) < 4)
                solid = np.asarray(pair["input"]).copy()
                for r, c in np.argwhere(mask):
                    solid[5*r:5*r+4, 5*c:5*c+4] = info["foreground"]
                _require(np.count_nonzero(solid != pair["output"]) == int(mask.sum()),
                         "solid-fill ablation does not distinguish local holes")
                ablations += 1
                if split == "train" and _rectangle(mask):
                    _require(not np.any(mask.sum(axis=0) == 1)
                             and not np.any(mask.sum(axis=1) == 1), "rectangle has a line single")
                    _single_axis_counterexamples(info)
                    rectangles += 1
        puzzles.add(puzzle_hash.hexdigest())
    _require(len(puzzles) == 42, "42 seeds did not yield distinct puzzles")
    _require(len(alphabets) >= 50 and len(masks) >= 90 and len(completed_shapes) >= 100,
             "structural variation collapsed")
    _require(len(palettes) >= 40 and classes == {4, 12}, "palette or Latin-class variation collapsed")
    _require(rectangles >= 42 and min(full_lines.values()) > 0, "reasoning/axis coverage missing")
    _require(len(erasure_counts["train"]) >= 6 and len(erasure_counts["test"]) >= 3,
             "erasure-count diversity missing")
    _require(non_sudoku > 0 and non_diagonal > 0, "unintended extra constraints")
    # Every legal explicit count combination, streamed without retaining runs.
    count_cases = 0
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            puzzle = generate_puzzle(42000 + 10*train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "count mismatch")
            count_cases += 1
    first, second = generate_puzzle(0), generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for split in first
                 for a, b in zip(first[split], second[split]) for k in ("input", "output")),
             "whole-puzzle determinism failed")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
    # Four diagonal givens expose the full alphabet but have multiple completions.
    ambiguous = np.full((4, 4), -1, dtype=np.int8)
    np.fill_diagonal(ambiguous, np.arange(4))
    for bad in (np.zeros((19, 19), dtype=np.int8),
                _render(ambiguous, ((0, 0), (0, 3), (2, 1), (3, 2)), 0, 8, ambiguous < 0)):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or ambiguous input was accepted")
    return {"valid": True, "seeds": 42, "pairs": pair_count,
            "distinct_puzzles": len(puzzles), "hole_alphabets": len(alphabets),
            "erasure_masks": len(masks), "labeled_latin_patterns": len(completed_shapes),
            "color_role_pairs": len(palettes), "latin_intercalate_counts": sorted(classes),
            "train_erasures": sorted(erasure_counts["train"]),
            "test_erasures": sorted(erasure_counts["test"]),
            "erased_line_pairs": full_lines, "joint_rectangle_demonstrations": rectangles,
            "local_hole_ablation_pairs": ablations, "non_sudoku_pairs": non_sudoku,
            "non_diagonal_pairs": non_diagonal, "legal_count_cases": count_cases,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
