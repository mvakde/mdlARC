"""Whole-row equality, followed by adjacent-vector run contraction.

The required proposal necessarily yields a one-column answer. Output labels
are fixed at 5 (monochromatic) and 0 (mixed), so solve needs only its input.
Generation and all checks are bounded and perform no file or network I/O.
"""

from __future__ import annotations

import hashlib
import itertools

import numpy as np


POSITIVE = 5
BACKGROUND = 0


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its allowed range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    try:
        a = np.asarray(value)
    except (ValueError, TypeError) as error:
        raise ValueError("grid must be a rectangular integer array") from error
    if (a.ndim != 2 or a.dtype.kind not in "iu"
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("grid must have integer colors 0..9 and shape 1..30 by 1..30")
    return a.astype(np.int8, copy=False)


def _compress(a):
    """Keep the first vector of each adjacent-equal run, on both axes."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    reduced = a[rows]
    columns = np.r_[True, np.any(reduced[:, 1:] != reduced[:, :-1], axis=0)]
    return reduced[:, columns].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    positive = np.all(a == a[:, :1], axis=1)
    labels = np.where(positive, POSITIVE, BACKGROUND).astype(np.int8)
    # Materialize the same-size intermediate prescribed by the proposal.
    mask = np.repeat(labels[:, None], a.shape[1], axis=1)
    return _compress(mask)


def _key(a):
    return a.shape, a.tobytes()


def _runs(values):
    """A loop-based description used by validation, separate from solve."""
    labels, lengths = [], []
    for value, group in itertools.groupby(values):
        labels.append(bool(value))
        lengths.append(sum(1 for _ in group))
    return labels, lengths


def _row_truth(a):
    return [len(set(int(c) for c in row)) == 1 for row in a]


def _answer_from_truth(truth):
    labels, _ = _runs(truth)
    return np.array([[POSITIVE if b else BACKGROUND] for b in labels], dtype=np.int8)


def _mixed_row(rng, palette, width):
    a, b = (int(c) for c in rng.choice(palette, 2, replace=False))
    style = int(rng.integers(5))
    if style == 0:
        row = np.full(width, a, dtype=np.int8)
        row[int(rng.integers(width))] = b
    elif style == 1:
        row = np.array([a if j % 2 == 0 else b for j in range(width)], dtype=np.int8)
    elif style == 2:
        cut = int(rng.integers(1, width))
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
    elif style == 3:
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = rng.choice(palette, width).astype(np.int8)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _guard_row(rng, palette, width):
    """Mixed even under endpoints-only, first-two, or last-two comparison."""
    if width == 3:
        return np.array(palette[:3], dtype=np.int8)
    row = rng.choice(palette, width).astype(np.int8)
    row[0], row[1], row[-2], row[-1] = palette[0], palette[1], palette[0], palette[2]
    return row


def _exception(palette, width, position):
    row = np.full(width, palette[0], dtype=np.int8)
    row[position] = palette[1]
    return row


def _lengths(rng, count, high):
    lengths = rng.integers(1, high + 1, count).astype(int).tolist()
    # At most three decrements are needed for the largest sampled inventory.
    while sum(lengths) > 30:
        candidates = [i for i, n in enumerate(lengths) if n > 1]
        lengths[int(rng.choice(candidates))] -= 1
    return lengths


def _make_input(rng, start, lengths, width, anchor=False):
    palette_size = int(rng.integers(3 if anchor else 2, 11))
    palette = rng.choice(10, palette_size, replace=False).astype(np.int8)
    labels = [bool(start) != bool(i % 2) for i in range(len(lengths))]
    bands = []
    for positive, length in zip(labels, lengths):
        rows = []
        for _ in range(length):
            if positive:
                rows.append(np.full(width, int(rng.choice(palette)), dtype=np.int8))
            elif anchor:
                # Outside the protected block, keep all three shortcuts correct.
                # Otherwise removed and inserted bands can cancel in the final
                # alternating word despite a locally visible classification error.
                rows.append(_guard_row(rng, palette, width))
            elif rows and rng.random() < 0.2:
                rows.append(rows[-1].copy())
            else:
                rows.append(_mixed_row(rng, palette, width))
        bands.append(np.stack(rows))
    if anchor:
        negative_band = next(i for i, b in enumerate(labels) if not b and lengths[i] == 7)
        middle = int(rng.integers(1, width - 1))
        bands[negative_band] = np.stack([
            _guard_row(rng, palette, width), _exception(palette, width, 0),
            _guard_row(rng, palette, width), _exception(palette, width, middle),
            _guard_row(rng, palette, width), _exception(palette, width, width - 1),
            _guard_row(rng, palette, width),
        ])
        positive_band = next(i for i, b in enumerate(labels) if b and lengths[i] >= 2)
        # Adjacent unequal raw vectors must merge after semantic classification.
        bands[positive_band][0] = palette[0]
        bands[positive_band][1] = palette[1]
    return np.concatenate(bands, axis=0)


def _fresh(a, used):
    """No retries or retained candidates; preserve each row's equality class."""
    if _key(a) not in used:
        return a
    for i, positive in enumerate(_row_truth(a)):
        if positive:
            for color in range(10):
                candidate = a.copy()
                candidate[i] = color
                if _key(candidate) not in used:
                    return candidate
        else:
            for left in range(10):
                for right in range(10):
                    if left == right:
                        continue
                    candidate = a.copy()
                    candidate[i, :2] = (left, right)
                    if _key(candidate) not in used:
                        return candidate
    # Only four test inputs can be single-cell, so ten colors suffice even there.
    raise RuntimeError("finite uniqueness construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # These change the sampled structures, never the transformation or markers.
    regime = int(rng.integers(4))
    width_cap = int(rng.choice((5, 10, 20, 30)))
    first_start = bool(rng.integers(2))
    first_count = int(rng.integers(3, 7))
    specs = []
    for index in range(num_train):
        if index == 0:
            count, start = first_count, first_start
            lengths = _lengths(rng, count, 3)
            negatives = [i for i in range(count) if start == bool(i % 2)]
            negative_band = int(rng.choice(negatives))
            lengths[negative_band] = 7
            positives = [i for i in range(count) if start != bool(i % 2)]
            positive_band = int(rng.choice(positives))
            lengths[positive_band] = max(2, lengths[positive_band])
            width = int(rng.integers(3, width_cap + 1))
        else:
            candidates = [n for n in range(2, 8) if index != 1 or n != first_count]
            count = int(rng.choice(candidates))
            start = not first_start if index == 1 else bool(rng.integers(2))
            lengths = _lengths(rng, count, 4)
            width = int(rng.integers(2, width_cap + 1))
        specs.append((start, lengths, width, index == 0))

    max_train_bands = max(len(spec[1]) for spec in specs)
    max_train_duration = max(max(spec[1]) for spec in specs)
    boundary_start = bool(rng.integers(2))
    for index in range(num_test):
        start = bool(rng.integers(2))
        width = int(rng.integers(2, width_cap + 1))
        if regime == 0:  # More alternations than in any training answer.
            count = max_train_bands + 1 + index
            lengths = _lengths(rng, count, 3)
        elif regime == 1:  # A longer semantically constant band, not more labels.
            count = int(rng.integers(2, 7))
            lengths = _lengths(rng, count, 3)
            target = int(rng.integers(count))
            available = 30 - sum(lengths) + lengths[target]
            lengths[target] = int(rng.integers(max_train_duration + 1, min(16, available) + 1))
        elif regime == 2:  # No row-axis contraction is needed in these tests.
            count = int(rng.integers(2, 31))
            lengths = [1] * count
        else:  # All rows have one class; both classes are taught in training.
            start = boundary_start != bool(index % 2)
            lengths = [int(rng.integers(1, 31))]
            if start and rng.random() < 0.35:
                width = 1
        specs.append((start, lengths, width, False))

    result = {"train": [], "test": []}
    used = set()
    for index, (start, lengths, width, anchor) in enumerate(specs):
        a = _fresh(_make_input(rng, start, lengths, width, anchor), used)
        output = solve(a)
        if np.array_equal(a, output):
            # The only possible identity here is a 1x1 input already equal to 5.
            a = _fresh(((a.astype(int) + 1) % 10).astype(np.int8), used)
            output = solve(a)
        used.add(_key(a))
        split = "train" if index < num_train else "test"
        result[split].append({"input": a, "output": output})
    validate_generated_puzzle(result)
    return result


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _shortcut_truth(a, kind):
    if kind == "endpoints":
        return (a[:, 0] == a[:, -1]).tolist()
    if kind == "first_two":
        return (a[:, 0] == a[:, 1]).tolist()
    return (a[:, -2] == a[:, -1]).tolist()


def validate_generated_puzzle(puzzle) -> None:
    """Validate the sampler's teaching contract as well as the exact family rule."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_analysis, test_analysis = set(), [], []
    rejected_shortcuts = set()
    merged_different = {True: False, False: False}
    no_a = no_b = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            truth = _row_truth(a)
            labels, lengths = _runs(truth)
            expected = np.array([[POSITIVE if b else BACKGROUND] for b in labels], dtype=np.int8)
            _require(np.array_equal(output, expected), "wrong semantic run word")
            _require(output.shape == (len(labels), 1), "wrong compressed dimensions")
            _require(np.all(output[1:, 0] != output[:-1, 0]), "adjacent duplicate answer labels")
            # Re-expand the answer: it must exactly reconstruct A's full mask.
            mask = np.array([[POSITIVE if b else BACKGROUND] * a.shape[1] for b in truth], dtype=np.int8)
            expanded = np.repeat(np.repeat(output, lengths, axis=0), a.shape[1], axis=1)
            _require(np.array_equal(expanded, mask), "mask reconstruction failed")
            _require(np.array_equal(_compress(mask.T).T, output), "axis order changed the result")
            analysis = (labels, lengths, a.shape)
            (train_analysis if split == "train" else test_analysis).append(analysis)
            if split == "train":
                _require(a.shape[1] >= 2 and len(labels) >= 2, "training must teach both row classes")
                for kind in ("endpoints", "first_two", "last_two"):
                    wrong = _answer_from_truth(_shortcut_truth(a, kind))
                    if not np.array_equal(wrong, output):
                        rejected_shortcuts.add(kind)
                for i in range(1, len(a)):
                    if truth[i] == truth[i - 1] and not np.array_equal(a[i], a[i - 1]):
                        merged_different[truth[i]] = True
                no_a |= not np.array_equal(_compress(a), output)
                no_b |= not np.array_equal(mask, output)
    _require(rejected_shortcuts == {"endpoints", "first_two", "last_two"}, "training shortcuts survive")
    _require(all(merged_different.values()), "missing abstraction-before-compression contrasts")
    _require(no_a and no_b, "a parent omission leaves all teaching answers unchanged")
    train_words = {tuple(labels) for labels, _, _ in train_analysis}
    _require(len(train_words) >= 2, "training answers must vary structurally")
    _require({labels[0] for labels, _, _ in train_analysis} == {False, True}, "training starts need both labels")
    train_max_bands = max(len(labels) for labels, _, _ in train_analysis)
    train_max_duration = max(max(lengths) for _, lengths, _ in train_analysis)
    for labels, lengths, _ in test_analysis:
        _require(len(labels) > train_max_bands or max(lengths) > train_max_duration
                 or all(n == 1 for n in lengths) or len(labels) == 1,
                 "test lacks the promised structural shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _expect_value_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test():
    """42 seeds, all 30 legal counts, finite boundary and counterfactual checks.

    Returns small descriptive measurements; writes no artifacts and retains no
    corpus. These are generator-owned tests, not independent semantic approval.
    """
    shapes, words, palettes, layouts, fingerprints = set(), set(), set(), set(), set()
    normalized = set()
    shift_counts = dict(more_bands=0, longer_band=0, unit_runs=0, single_class=0)
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic puzzle")
        fingerprints.add(_fingerprint(puzzle))
        train_bands = max(p["output"].shape[0] for p in puzzle["train"])
        train_duration = max(max(_runs(_row_truth(p["input"]))[1]) for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, output = pair["input"], pair["output"]
                labels, lengths = _runs(_row_truth(a))
                pairs += 1
                shapes.add(a.shape)
                words.add(tuple(labels))
                palettes.add(tuple(int(c) for c in np.unique(a)))
                layouts.add((tuple(labels), tuple(lengths), a.shape[1]))
                mapping = {}
                normalized.add((a.shape, bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)))
                # Bijections of the input palette cannot change either predicate.
                permutation = np.random.default_rng(seed + 1000).permutation(10)
                _require(np.array_equal(solve(permutation[a]), output), "literal input colors affected answer")
                _require(np.array_equal(solve(a[:, ::-1]), output), "column reflection affected equality")
                if a.shape[0] < 30:
                    repeated = np.insert(a, 1, a[0], axis=0)
                    _require(np.array_equal(solve(repeated), output), "repetition changed the run word")
                if split == "test":
                    shift_counts["more_bands"] += len(labels) > train_bands
                    shift_counts["longer_band"] += max(lengths) > train_duration
                    shift_counts["unit_runs"] += all(n == 1 for n in lengths)
                    shift_counts["single_class"] += len(labels) == 1
    _require(len(fingerprints) == 42, "42 seeds did not give distinct puzzles")
    _require(len(shapes) >= 40 and len(layouts) >= 150 and len(normalized) >= 180,
             "insufficient geometry or structure variation")
    _require(len(words) >= 16 and len(palettes) >= 35, "insufficient run words or input palettes")
    _require(all(n >= 3 for n in shift_counts.values()), "missing test-shift regimes")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(100 + a * 10 + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "wrong explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        _expect_value_error(lambda a=a, b=b: generate_puzzle(0, a, b))
    for seed in (-1, 0.5, True):
        _expect_value_error(lambda seed=seed: generate_puzzle(seed))
    for a, expected in (
        (np.array([[7]]), [[5]]),
        (np.array([[0], [5], [9]]), [[5]]),
        (np.array([[7, 2]]), [[0]]),
        (np.full((30, 30), 7), [[5]]),
        (np.tile([1, 2, 3], (30, 10)), [[0]]),
        (np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 2, 1], [3, 3, 3]]), [[5], [0], [5]]),
    ):
        before = a.copy()
        _require(np.array_equal(solve(a), expected), "boundary or separated recurrence failed")
        _require(np.array_equal(a, before), "solve mutated its input")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "run_words": len(words), "palettes": len(palettes),
            "layouts": len(layouts), "color_normalized_inputs": len(normalized),
            "test_shift_counts": shift_counts, "legal_count_combinations": 30,
            "independent_semantic_review": False}
