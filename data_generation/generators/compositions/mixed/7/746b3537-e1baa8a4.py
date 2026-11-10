"""Faithful implementation of proposal.json, including its documented redundancy.

The execution order is B (whole-line run collapse), then A (stripe compression).
On this domain B already finishes the task. Tests explicitly measure that fact;
they do not claim that the two parents are independently necessary.
No files, generators, examples, random state, or seed are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


# Hashes of the 21 provided original/selected-variation inputs. Only generation
# and validation use these exclusions; solve has no example lookup.
_PARENT_INPUT_HASHES = frozenset((
    "3e0dbba8e69a71acc3e04efc7404978d10b55538b434082c640f4f358e6b0502",
    "cb48274078ce7ffc5e5fdb2b81085a4cb02cd1a794f237876765be08f6221f86",
    "fbe1dbc2640f6a7f5f99a04a91c10ba54dc716fde07a0681832ea05b3ad01c28",
    "f9bc82eaf0276c1c3bd8cdab441bee36deaecc5fba6d256821598ed0fec4e87a",
    "ebe5963682e37e60b0723b471ded315ef505325091e90656401e975e27a8b907",
    "6d120cbb86ff9026f5b89c1381b0d2f7b6b4792b25b1fba0cd9dc280e7e54467",
    "f30870ad9eb8baaa072fd8cd14cf9cbc19ab68a70de6e570abe2fa48251adad8",
    "7774361869a82569e90757c6037b594f77b958c903de2b33fab9615fa5ed0529",
    "ae407229ca89893c3304d1ea91e7bea9e7f506123602a6a82cc2e56b599a1825",
    "816052668bcd4871e42a2210cfa70c95908bd4aafff2890ef8f8d2903d9dc4e3",
    "44553ebeffd77fdb4d04a2284e96fc6cb350b1c750a8b696804fc16f923a6ea7",
    "f292ab7c7488a9e18e7137ddce326e81e3b50ee4e8aefd3b94a7a0cda75cbaf6",
    "85e50d1ca14c8a75b3eea7cf64fbcfb15d9e26501c6196e635848f802ece1b09",
    "06e48453767fce82f754d45e007a59742bb611a2d1fbca5bb49009d9cf260ff0",
    "903288eaca8a3babdda5e7728a0a8e46886dfea178615a73bb4fd362c3fa3afa",
    "23676a4656dc0ffc41eca13b4920a7708d6234af51beab779149ffebb20776e2",
    "1df01b0449ec03449c2ceffeaf4a1c4da6690e052058e3d39e6a682ac9b72c79",
    "3f8399862bbff4ee0f9a111fa4c9639418a7d2d872cd5bdda236e8e65f483b65",
    "80f470f3f300c392e12dd126da653d9e6c1b9b133aea78d0b3ea487bb1b17014",
    "d4a655409cfc1c34c18aadd3fb5f625ae14148aa260d62bd8be98003c6d2d5b8",
    "8c004e0c955221107407b62a2ab8ae7a56c33377d0b650865f43417ecc32d30a",
))


def _grid(value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not bool(np.all((0 <= array) & (array <= 9)))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array


def _fingerprint(grid: np.ndarray) -> str:
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


def _collapse_lines(grid: np.ndarray) -> np.ndarray:
    """B: retain the first complete vector of each consecutive identical run."""
    rows = [0] + [i for i in range(1, grid.shape[0])
                  if not np.array_equal(grid[i], grid[i - 1])]
    columns = [0] + [j for j in range(1, grid.shape[1])
                     if not np.array_equal(grid[:, j], grid[:, j - 1])]
    return grid[np.ix_(rows, columns)].copy()


def _compress_stripes(grid: np.ndarray) -> np.ndarray:
    """A: full-length solid stripes to a strip, including the uniform limit."""
    rows_equal = bool(np.all(grid == grid[:1, :]))
    columns_equal = bool(np.all(grid == grid[:, :1]))
    if rows_equal and columns_equal:
        return grid[:1, :1].copy()
    if not (rows_equal or columns_equal):
        raise ValueError("input must consist of parallel full-length solid stripes")
    values = grid[0] if rows_equal else grid[:, 0]
    keep = np.r_[True, values[1:] != values[:-1]]
    strip = values[keep].copy()
    return strip.reshape(1, -1) if rows_equal else strip.reshape(-1, 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the prescribed B -> A order, using only this input."""
    return _compress_stripes(_collapse_lines(_grid(input_grid)))


def _features(grid: np.ndarray) -> dict:
    """Read maximal physical stripes directly, separately from the staged solve."""
    horizontal_sequence = bool(np.all(grid == grid[:1]))
    vertical_sequence = bool(np.all(grid == grid[:, :1]))
    if horizontal_sequence == vertical_sequence:
        raise ValueError("generated inputs require one unambiguous changing axis")
    axis = 1 if horizontal_sequence else 0
    values = grid[0] if axis == 1 else grid[:, 0]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return {"axis": axis, "colors": tuple(int(v) for v in values[starts]),
            "widths": tuple(int(v) for v in np.diff(np.r_[starts, len(values)])),
            "cross_section": grid.shape[1 - axis]}


def _expand(colors, widths, cross_section: int, axis: int) -> np.ndarray:
    row = np.repeat(np.asarray(colors, dtype=np.int8), widths)
    grid = np.tile(row, (cross_section, 1))
    return grid if axis == 1 else grid.T.copy()


def _sequence(rng: np.random.Generator, length: int, mode: str) -> tuple:
    if mode == "distinct":
        count = length
    elif mode == "alternate":
        count = 2
    else:
        maximum = min(10, length - (mode == "return"))
        count = int(rng.integers(2, maximum + 1))
    palette = [int(c) for c in rng.choice(10, count, replace=False)]
    if mode == "alternate":
        return tuple(palette[i % 2] for i in range(length))
    sequence = list(palette)
    while len(sequence) < length:
        sequence.append(int(rng.choice([c for c in palette if c != sequence[-1]])))
    # Move returning colors away from a fixed suffix without losing run boundaries.
    for _ in range(2 * length):
        a, b = (int(i) for i in rng.integers(0, length, size=2))
        sequence[a], sequence[b] = sequence[b], sequence[a]
        if any(sequence[i] == sequence[i + 1] for i in range(length - 1)):
            sequence[a], sequence[b] = sequence[b], sequence[a]
    return tuple(sequence)


def _widths(rng: np.random.Generator, count: int, mode: str) -> tuple:
    if mode == "unit":
        values = np.ones(count, dtype=int)
    elif mode == "uniform":
        values = np.full(count, int(rng.integers(1, min(6, 30 // count) + 1)))
    elif mode == "wide":
        values = np.ones(count, dtype=int)
        values[int(rng.integers(count))] = int(rng.integers(9, min(20, 31 - count) + 1))
    elif mode in ("train", "demo"):
        values = rng.integers(1, 5, size=count)
        if mode == "demo" and bool(np.all(values == values[0])):
            values[int(rng.integers(count))] = 1 if values[0] != 1 else 2
    else:
        extra = int(rng.integers(0, 31 - count))
        values = 1 + rng.multinomial(extra, np.full(count, 1 / count))
    return tuple(int(v) for v in values)


def _candidate(rng, *, axis, split, shift, mode, demo=False) -> dict:
    if split == "train":
        count = int(rng.integers(3 if mode == "return" else 2, 8))
        thickness_mode = "demo" if demo else (
            "unit", "uniform", "train")[int(rng.integers(3))]
        # Training uniform schedules are also capped at four cells per stripe.
        if thickness_mode == "uniform":
            widths = (int(rng.integers(1, 5)),) * count
        else:
            widths = _widths(rng, count, thickness_mode)
    else:
        count = int(rng.integers(8, 19)) if shift == "length" else int(rng.integers(2, 8))
        thickness_mode = "wide" if shift == "width" else (
            "unit", "uniform", "mixed")[int(rng.integers(3))]
        widths = _widths(rng, count, thickness_mode)
        mode = ("free", "alternate", "return", "distinct")[int(rng.integers(4))]
        if mode == "return" and count == 2:
            mode = "free"
        if mode == "distinct" and count > 10:
            mode = "free"
    colors = _sequence(rng, count, mode)
    if split == "train":
        cross_section = int(rng.integers(2 if demo else 1, 11))
    elif shift == "cross_section":
        cross_section = int(rng.integers(12, 31))
    else:
        cross_section = sum(widths) if rng.random() < .2 else int(rng.integers(1, 31))
    if cross_section == 1 and max(widths) == 1:
        changed = list(widths)
        changed[int(rng.integers(count))] = 2
        widths = tuple(changed)
    input_grid = _expand(colors, widths, cross_section, axis)
    # Output is constructed from the intended sequence, then checked from input.
    output = np.asarray(colors, dtype=np.int8).reshape(1, -1)
    if axis == 0:
        output = output.T.copy()
    return {"input": input_grid, "output": output}


def _count(value, name, low, high) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    shift = ("length", "width", "cross_section")[int(rng.integers(3))]
    axes = [0, 1] + [int(v) for v in rng.integers(0, 2, size=num_train - 2)]
    rng.shuffle(axes)
    returning, distinct = (int(i) for i in rng.choice(num_train, 2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            axis = axes[index] if split == "train" else int(rng.integers(2))
            mode = ("return" if index == returning else
                    "distinct" if index == distinct else "free")
            for _ in range(96):
                pair = _candidate(rng, axis=axis, split=split, shift=shift,
                                  mode=mode, demo=(split == "train" and index == returning))
                key = _fingerprint(pair["input"])
                if key not in seen and key not in _PARENT_INPUT_HASHES:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ("train", "test")):
        raise ValueError("splits must be lists")
    a = _count(len(puzzle["train"]), "num_train", 2, 10)
    b = _count(len(puzzle["test"]), "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    info = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[role]) for role in ("input", "output"))
            key = _fingerprint(inp)
            assert key not in seen and key not in _PARENT_INPUT_HASHES
            seen.add(key)
            features = _features(inp)
            axis, colors = features["axis"], features["colors"]
            expected = np.asarray(colors).reshape((1, -1) if axis == 1 else (-1, 1))
            assert np.array_equal(out, expected), "incorrect ordered stripe strip"
            assert np.array_equal(solve(inp), expected), "staged solver disagrees"
            assert np.array_equal(_collapse_lines(inp), out), "B-only equivalence failed"
            assert np.array_equal(_compress_stripes(inp), out), "A-only equivalence failed"
            assert np.array_equal(_expand(colors, features["widths"],
                                          features["cross_section"], axis), inp)
            assert not np.array_equal(inp, out), "generated pair should compress"
            assert np.array_equal(np.unique(inp), np.unique(out)), "colors must be copied"
            assert np.array_equal(solve(out), out), "output must be irreducible"
            info[split].append(features)
    training = info["train"]
    assert {f["axis"] for f in training} == {0, 1}, "demonstrate both orientations"
    assert any(len(set(f["colors"])) < len(f["colors"]) for f in training)
    assert any(len(set(f["colors"])) == len(f["colors"]) for f in training)
    assert any(len(set(f["widths"])) > 1 and f["cross_section"] > 1 for f in training)
    for f in info["test"]:
        assert (len(f["colors"]) > max(len(t["colors"]) for t in training)
                or max(f["widths"]) > max(max(t["widths"]) for t in training)
                or f["cross_section"] > max(t["cross_section"] for t in training)), \
            "test must extend at least one demonstrated structural dimension"


def _puzzle_key(puzzle) -> tuple:
    return tuple(tuple((_fingerprint(np.asarray(p["input"])),
                        _fingerprint(np.asarray(p["output"]))) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> dict:
    """Stream 42 seeds; retain small signatures/metrics, never a puzzle corpus."""
    shapes, structures, palettes, lengths, widths, crosses, fingerprints = (set() for _ in range(7))
    axes, shifts = set(), set()
    pairs = returning_pairs = distinct_pairs = omit_a_changes = omit_b_changes = 0
    test_axes, aspect_classes, test_width_classes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                features = _features(inp)
                colors, stripe_widths = features["colors"], features["widths"]
                axes.add(features["axis"])
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(set(colors))))
                lengths.add(len(colors))
                widths.add(stripe_widths)
                crosses.add(features["cross_section"])
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                pairs += 1
                returning_pairs += len(set(colors)) < len(colors)
                distinct_pairs += len(set(colors)) == len(colors)
                omit_a_changes += not np.array_equal(_collapse_lines(inp), out)
                omit_b_changes += not np.array_equal(_compress_stripes(inp), out)
                assert np.array_equal(solve(inp.T), out.T), "transpose covariance"
                original = inp.copy()
                solve(inp)
                assert np.array_equal(inp, original), "solve must not mutate input"
                if split == "test":
                    test_axes.add(features["axis"])
                    aspect_classes.add(int(np.sign(inp.shape[0] - inp.shape[1])))
                    test_width_classes.add("unit" if max(stripe_widths) == 1 else
                                           "uniform" if len(set(stripe_widths)) == 1 else "unequal")
                    train_features = [_features(p["input"]) for p in puzzle["train"]]
                    for name, measure in (("length", lambda f: len(f["colors"])),
                                          ("width", lambda f: max(f["widths"])),
                                          ("cross_section", lambda f: f["cross_section"])):
                        if measure(features) > max(map(measure, train_features)):
                            shifts.add(name)
        # A bijective palette relabeling cannot alter the structure of the rule.
        mapping = np.random.default_rng(seed + 100).permutation(10)
        first = puzzle["train"][0]
        assert np.array_equal(solve(mapping[first["input"]]), mapping[first["output"]])
        if seed in (0, 41):
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(fingerprints) == 42 and len(shapes) >= 45 and len(structures) >= 100
    assert len(palettes) >= 30 and len(widths) >= 40 and len(crosses) >= 15
    assert axes == test_axes == {0, 1} and shifts == {"length", "width", "cross_section"}
    assert aspect_classes == {-1, 0, 1}
    assert test_width_classes == {"unit", "uniform", "unequal"}
    assert returning_pairs and distinct_pairs
    # This is an explicit FAILED conceptual necessity test, not a quality claim.
    assert omit_a_changes == omit_b_changes == 0
    legal_count_combinations = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * a + b, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
            legal_count_combinations += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5, None):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # Fixed boundary checks include leading zero, recurrence, maximum dimensions,
    # already compact strips, and the orientation-neutral monochrome limit.
    example = _expand((0, 9, 0, 3), (1, 7, 2, 20), 30, 1)
    assert np.array_equal(solve(example), np.array([[0, 9, 0, 3]]))
    assert np.array_equal(solve(example.T), np.array([[0], [9], [0], [3]]))
    assert np.array_equal(solve(np.full((30, 30), 5, dtype=int)), [[5]])
    unit = np.asarray([[i % 2 for i in range(30)]])
    assert np.array_equal(solve(unit), unit)
    assert np.array_equal(solve(unit.T), unit.T)
    for invalid in (np.array([[1, 2], [3, 4]]), np.zeros((0, 2), dtype=int),
                    np.zeros((31, 2), dtype=int), np.array([[10]]), np.array([[1.0]])):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "normalized_input_structures": len(structures),
            "palettes": len(palettes), "run_counts": sorted(lengths),
            "width_schedules": len(widths), "cross_section_sizes": sorted(crosses),
            "returning_color_pairs": returning_pairs, "all_distinct_color_pairs": distinct_pairs,
            "test_axes": sorted(test_axes), "test_shift_dimensions": sorted(shifts),
            "test_width_classes": sorted(test_width_classes),
            "legal_count_combinations": legal_count_combinations,
            "omitting_A_changes_answer": omit_a_changes,
            "omitting_B_changes_answer": omit_b_changes,
            "conceptual_composition_valid": False,
            "limitation": "On solid full-length stripes, A(B(input))=B(input)=A(input)."}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
