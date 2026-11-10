"""Generate ARC rows by complementing membership in each row's signal."""

from collections import Counter
from itertools import permutations

import numpy as np


SOURCE_INPUTS = {
    (3, 10, "666666607788880708880270444444"),
    (3, 10, "666660707044400777008888007070"),
    (3, 10, "040404040460606060600808080808"),
    (3, 10, "888003088830444444000606360606"),
}


def _key(grid):
    return grid.shape + ("".join(map(str, grid.flat)),)


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train is None:
        low = max(2, int(num_test) + 1) if num_test is not None else 2
        num_train = int(rng.integers(low, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _runs(mask):
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def _mask(rng, width, kind):
    mask = np.zeros(width, dtype=bool)
    if kind == "all":
        mask[:] = True
    elif kind == "absent":
        pass
    elif kind == "alternating":
        mask[:] = (np.arange(width) + int(rng.integers(2))) % 2 == 0
    elif kind == "run":
        length = int(rng.integers(2, width))
        start = int(rng.integers(width - length + 1))
        mask[start:start + length] = True
    elif kind == "dense":
        mask[:] = True
        n = int(rng.integers(1, min(4, width - 1)))
        mask[rng.choice(width, size=n, replace=False)] = False
    else:
        n = int(rng.integers(2, width - 1))
        mask[rng.choice(width, size=n, replace=False)] = True
        if kind == "split" and _runs(mask) == 1:
            mask[:] = False
            left = int(rng.integers(width - 3))
            right = int(rng.integers(left + 2, width))
            mask[left] = mask[right] = True
            if right + 1 < width and rng.random() < .5:
                mask[right + 1] = True
    return mask


def _noise_positions(rng, positions, count):
    if count > 1 and rng.random() < .55:
        free = set(map(int, positions))
        starts = [int(p) for p in positions
                  if all(int(p) + k in free for k in range(count))]
        if starts:
            start = int(rng.choice(starts))
            return np.arange(start, start + count)
    return rng.choice(positions, size=count, replace=False)


def _add_noise(rng, row, mask, palette, two_colors=False, block=False):
    free = np.flatnonzero(~mask)
    if not len(free) or not palette:
        return
    low = 3 if block else 2 if two_colors else 1
    low = min(low, len(free))
    high = min(6, len(free))
    count = int(rng.integers(low, high + 1))
    if block:
        available = set(map(int, free))
        starts = []
        while count >= 3 and not starts:
            starts = [int(p) for p in free
                      if all(int(p) + k in available for k in range(count))]
            if not starts:
                count -= 1
        assert starts
        start = int(rng.choice(starts))
        positions = np.arange(start, start + count)
    else:
        positions = _noise_positions(rng, free, count)
    if two_colors and count >= 2:
        chosen = rng.choice(palette, size=2, replace=False)
        row[positions] = int(chosen[0])
        row[positions[-1]] = int(chosen[1])
    elif not block and count >= 2 and len(palette) >= 2 and rng.random() < .30:
        row[positions] = rng.choice(palette, size=count)
        if len(set(map(int, row[positions]))) == 1:
            row[positions[-1]] = int(next(c for c in palette
                                          if c != row[positions[0]]))
    else:
        row[positions] = int(rng.choice(palette))


def _make_pair(rng, background, order, width, train_noise, pair_noise,
               anchor=False, special=False, absent=False, is_test=False,
               mixed_test=False):
    height = len(order)
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    composition = str(rng.choice(("independent",) * 3 +
                                 ("runs", "irregular", "alternating")))
    kinds = []
    for _ in range(height):
        if composition == "runs":
            kind = "run"
        elif composition == "irregular":
            kind = str(rng.choice(("split", "scattered")))
        elif composition == "alternating":
            kind = "alternating"
        else:
            kind = str(rng.choice(("run", "split", "scattered",
                                   "alternating", "dense", "all"),
                                  p=(.23, .19, .19, .20, .13, .06)))
        kinds.append(kind)
    special_row = int(rng.integers(height)) if special else -1
    absent_row = int(rng.integers(height)) if absent else -1
    noisy_row = int(rng.integers(height)) if anchor else -1
    clean_row = (noisy_row + 1) % height if anchor else -1
    mixed_row = int(rng.integers(height)) if mixed_test else -1
    for r, signal in enumerate(order):
        if r == special_row:
            mask = np.zeros(width, dtype=bool)
            length = int(rng.integers(2, 4))
            start = int(rng.integers(width - length + 1))
            mask[start:start + length] = True
        else:
            mask = _mask(rng, width, "absent" if r == absent_row else kinds[r])
        if anchor and (np.all(mask) or not np.any(mask)
                       or np.count_nonzero(~mask) < 3):
            mask[:] = False
            mask[:width - 3] = True
        if r == mixed_row and np.count_nonzero(~mask) < 2:
            mask[:] = False
            mask[:width - 2] = True
        inp[r, mask] = signal
        out[r, ~mask] = signal
        if r == clean_row or np.all(mask):
            continue
        if r == special_row:
            _add_noise(rng, inp[r], mask, train_noise, block=True)
        elif r == noisy_row:
            _add_noise(rng, inp[r], mask, train_noise, two_colors=True)
        elif r == mixed_row:
            free = np.flatnonzero(~mask)
            chosen = rng.choice(free, size=2, replace=False)
            inp[r, chosen[0]] = int(rng.choice(train_noise))
            novel = [color for color in pair_noise if color not in train_noise]
            inp[r, chosen[1]] = int(rng.choice(novel))
        elif rng.random() < (.68 if is_test else .55):
            _add_noise(rng, inp[r], mask, pair_noise)
    return inp, out


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    height = int(rng.choice((2, 3, 3, 3, 4)))
    background = int(rng.choice(10, p=(.36,) + (.64 / 9,) * 9))
    palette = list(map(int, rng.permutation(
        [c for c in range(10) if c != background])))
    signals = tuple(palette[:height])
    available_noise = palette[height:]
    train_noise = available_noise[:int(rng.integers(
        2, min(4, len(available_noise)) + 1))]
    novel_noise = [c for c in available_noise if c not in train_noise]
    orders = list(permutations(signals))
    first = orders[int(rng.integers(len(orders)))]
    second = orders[int(rng.integers(len(orders)))]
    if second == first:
        second = next(order for order in orders if order != first)
    train_orders = [first, second] + [
        orders[int(rng.integers(len(orders)))]
        for _ in range(num_train - 2)]
    test_orders = [
        train_orders[int(rng.integers(num_train))]
        if rng.random() < .5 else orders[int(rng.integers(len(orders)))]
        for _ in range(num_test)]
    widths = [int(rng.integers(6, 21))
              for _ in range(num_train + num_test)]
    widths[0] = max(8, widths[0])
    special = rng.random() < .38
    if special:
        widths[1] = max(9, widths[1])
    absent_train = rng.random() < .24 and not special
    absent_index = int(rng.integers(1, num_train)) if absent_train else -1
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used = set()
    for i in range(num_train + num_test):
        local = np.random.default_rng(int(pair_seeds[i]))
        training = i < num_train
        order = train_orders[i] if training else test_orders[i - num_train]
        is_absent = (training and i == absent_index
                     or not training and absent_train and local.random() < .28)
        if training:
            pair_noise = train_noise
        else:
            mode = int(local.integers(4))  # none, familiar, new, mixed
            pair_noise = ([], train_noise, novel_noise,
                          train_noise + novel_noise)[mode]
        for _ in range(40):
            inp, out = _make_pair(
                local, background, order, widths[i], train_noise, pair_noise,
                anchor=training and i == 0,
                special=training and i == 1 and special,
                absent=is_absent, is_test=not training,
                mixed_test=not training and mode == 3)
            key = _key(inp)
            if key not in used and key not in SOURCE_INPUTS:
                break
        else:
            raise RuntimeError("Failed to construct a distinct pair")
        used.add(key)
        puzzle["train" if training else "test"].append(
            {"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _learn_roles(train):
    outputs = [set(map(int, row)) for row in train[0]["output"]]
    common = set.intersection(*outputs)
    assert len(common) == 1, "Training must reveal the background"
    background = common.pop()
    signals = set().union(*outputs) - {background}
    assert all(len(colors) == 2 for colors in outputs)
    assert len(signals) == train[0]["input"].shape[0]
    return background, signals


def _infer_output(train, inp):
    background, signals = _learn_roles(train)
    assert inp.shape[0] == len(signals)
    visible = [set(map(int, row)) & signals for row in inp]
    assert all(len(colors) <= 1 for colors in visible)
    seen = set().union(*visible)
    assert len(seen) == sum(bool(colors) for colors in visible)
    missing = signals - seen
    assert len(missing) == sum(not colors for colors in visible)
    assert len(missing) <= 1
    order = [next(iter(colors)) if colors else next(iter(missing))
             for colors in visible]
    return np.asarray(
        [np.where(row == signal, background, signal)
         for row, signal in zip(inp, order)], dtype=np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert type(train) is list and type(test) is list
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    used = set()
    height = None
    for pair in train + test:
        assert type(pair) is dict and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        assert inp.shape == out.shape and inp.shape[0] in (2, 3, 4)
        height = inp.shape[0] if height is None else height
        assert inp.shape[0] == height
        key = _key(inp)
        assert key not in used and key not in SOURCE_INPUTS
        used.add(key)
        assert not np.array_equal(inp, out)
    background, signals = _learn_roles(train)
    assert background not in signals and len(signals) == height
    clean_row = mixed_nuisance_row = False
    train_orders = []
    train_has_absent = False
    test_has_absent = False
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            assert np.array_equal(out, _infer_output(train, inp))
            visible = [set(map(int, row)) & signals for row in inp]
            missing = signals - set().union(*visible)
            if any(not colors for colors in visible):
                if section == "train":
                    train_has_absent = True
                else:
                    test_has_absent = True
            order = []
            for row, colors, result in zip(inp, visible, out):
                signal = next(iter(colors)) if colors else next(iter(missing))
                order.append(signal)
                nuisance = set(map(int, row)) - {background, signal}
                assert nuisance.isdisjoint(signals)
                assert set(map(int, result)) <= {background, signal}
                if section == "train":
                    clean_row |= not bool(nuisance)
                    mixed_nuisance_row |= len(nuisance) >= 2
            assert set(order) == signals
            if section == "train":
                train_orders.append(tuple(order))
    assert clean_row and mixed_nuisance_row
    assert len(set(train_orders)) >= 2
    assert not test_has_absent or train_has_absent


def _equal_puzzles(a, b):
    return all(
        len(a[section]) == len(b[section]) and all(
            np.array_equal(x[name], y[name])
            for x, y in zip(a[section], b[section])
            for name in ("input", "output"))
        for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) >= 40
    assert len({p["train"][0]["input"].shape[1] for p in puzzles}) >= 8
    assert len({tuple(sorted(_learn_roles(p["train"])[1]))
                for p in puzzles}) >= 12
    assert len({_learn_roles(p["train"])[0] for p in puzzles}) >= 6
    assert len({p["train"][0]["input"].shape[0] for p in puzzles}) == 3
    assert _equal_puzzles(generate_puzzle(21), generate_puzzle(21))
    assert not _equal_puzzles(generate_puzzle(21), generate_puzzle(22))
    masks, noise_counts, orders, widths = set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background, signals = _learn_roles(puzzle["train"])
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            assert np.array_equal(_infer_output(puzzle["train"], inp), out)
            widths.add(inp.shape[1])
            order = []
            for row, result in zip(inp, out):
                local = set(map(int, row)) & signals
                if local:
                    signal = next(iter(local))
                    order.append(signal)
                    mask = row == signal
                    masks.add(tuple(map(int, mask)))
                    noise_counts.add(int(np.count_nonzero(
                        (row != signal) & (row != background))))
                    assert np.array_equal(
                        result, np.where(mask, background, signal))
            orders.add(tuple(order))
    assert len(masks) >= 150 and len(widths) >= 12
    assert len(noise_counts) >= 5 and len(orders) >= 12
    coverage = Counter()
    for puzzle in puzzles:
        background, signals = _learn_roles(puzzle["train"])
        known_noise = {
            int(value) for pair in puzzle["train"]
            for value in pair["input"].flat
        } - {background} - signals
        trained_orders = []
        for pair in puzzle["train"]:
            trained_orders.append(tuple(
                next(iter(set(map(int, row)) & signals))
                if set(map(int, row)) & signals else -1
                for row in pair["input"]))
        if len({pair["input"].shape[1]
                for pair in puzzle["train"] + puzzle["test"]}) > 1:
            coverage["within_width"] += 1
        for section in ("train", "test"):
            for index, pair in enumerate(puzzle[section]):
                inp = pair["input"]
                visible = [set(map(int, row)) & signals for row in inp]
                missing = signals - set().union(*visible)
                kinds, noise_sets = [], []
                for row, colors in zip(inp, visible):
                    signal = next(iter(colors)) if colors else next(iter(missing))
                    mask = row == signal
                    nuisance = set(map(int, row)) - {background, signal}
                    noise_sets.append(nuisance)
                    noise_count = int(np.count_nonzero(
                        (row != background) & ~mask))
                    if mask.all():
                        kind = "all"
                    elif not mask.any():
                        kind = "absent"
                    elif np.all(mask == (np.arange(len(row)) % 2 == 0)) or np.all(
                            mask == (np.arange(len(row)) % 2 == 1)):
                        kind = "alternating"
                    elif _runs(mask) == 1:
                        kind = "run"
                    else:
                        kind = "irregular"
                    kinds.append(kind)
                    if section == "train":
                        if 0 < int(mask.sum()) <= 3:
                            coverage["short_signal"] += 1
                        if noise_count >= int(mask.sum()) and noise_count:
                            coverage["noise_tie"] += 1
                        if any(np.all(row[j:j + 3] == color)
                               for color in nuisance
                               for j in range(len(row) - 2)):
                            coverage["noise_block"] += 1
                        if kind == "alternating" and nuisance:
                            coverage["noisy_alternating"] += 1
                        if kind == "all":
                            coverage["all_signal"] += 1
                        if kind == "absent":
                            coverage["absent"] += 1
                if section == "train":
                    if not any(noise_sets) and any(
                            kind != "alternating" for kind in kinds):
                        coverage["clean_non_alternating"] += 1
                    if not any(noise_sets) and index < len(puzzle["train"]) - 1:
                        coverage["clean_before_last"] += 1
                    if any(noise_sets) and not all(noise_sets):
                        coverage["mixed_clean_noisy_rows"] += 1
                else:
                    present_noise = set().union(*noise_sets)
                    familiar = bool(present_noise & known_noise)
                    novel = bool(present_noise - known_noise)
                    coverage[f"noise_{'both' if familiar and novel else 'familiar' if familiar else 'new' if novel else 'none'}"] += 1
                    order = tuple(
                        next(iter(colors)) if colors else next(iter(missing))
                        for colors in visible)
                    coverage["repeated_order" if order in trained_orders
                             else "new_order"] += 1
                    if all(kind == "run" for kind in kinds):
                        coverage["all_runs"] += 1
                    if all(kind == "irregular" for kind in kinds):
                        coverage["all_irregular"] += 1
                    if all(kind == "alternating" for kind in kinds):
                        coverage["all_alternating"] += 1
                    if len(set(kinds)) > 1 and present_noise:
                        coverage["mixed_with_noise"] += 1
    required = ("within_width", "short_signal", "noise_tie",
                "noise_block", "noisy_alternating", "all_signal", "absent",
                "clean_non_alternating", "clean_before_last",
                "mixed_clean_noisy_rows", "noise_both", "noise_familiar",
                "noise_new", "noise_none", "repeated_order", "new_order",
                "all_runs", "all_irregular", "all_alternating",
                "mixed_with_noise")
    assert all(coverage[name] > 0 for name in required), coverage
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1000, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0),
                   (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1000, *counts)
        except ValueError:
            continue
        raise AssertionError(f"Accepted illegal counts {counts}")
