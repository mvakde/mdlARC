"""Generate camouflage puzzles whose answer is the duplicated complete plus label."""

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_HASHES = {
    (22, 22, "0193a45e3bf75e038e4f84fd110f518af96b96ab66f7cfe7d2a74ee439d165ab"),
    (26, 24, "b89cf7ddef9b9c83fc488b4be30f6e4bbffb5f0a81d186f7b66885a2011c0fdf"),
    (20, 22, "c2a2a1873e41fc06d1d543da78309f2a438695854da2e1a4e9bcefe7bd14f7f5"),
    (27, 22, "1875f1cc93b5dc7adf1fbc185863f75c1cd6d80c63f5e1ca624daddcc0011451"),
}


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _count(num_train, "num_train")
    test = None if num_test is None else _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2 through 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1 through 4")
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _plus(row, col):
    return ((row, col), (row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1))


def _motifs(grid, hub, fillers):
    found = []
    height, width = grid.shape
    for row, col in np.argwhere(grid == hub):
        row, col = int(row), int(col)
        if not (0 < row < height - 1 and 0 < col < width - 1):
            continue
        arms = [int(grid[row + dr, col + dc]) for dr, dc in _DIRECTIONS]
        if len(set(arms)) == 1 and arms[0] not in fillers and arms[0] != hub:
            found.append((row, col, arms[0]))
    return found


def _inspect(grid):
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    order = sorted(range(10), key=lambda color: (-int(counts[color]), color))
    fillers = tuple(order[:2])
    candidate = []
    for hub in order[2:]:
        if counts[hub] == 0:
            continue
        motifs = _motifs(grid, hub, fillers)
        if len(motifs) >= 2:
            candidate.append((hub, motifs))
    if len(candidate) != 1:
        return None
    hub, motifs = candidate[0]
    labels = {}
    footprints = set()
    for row, col, arm in motifs:
        cells = set(_plus(row, col))
        if footprints.intersection(cells):
            return None
        footprints.update(cells)
        labels[arm] = labels.get(arm, 0) + 1
    repeated = [color for color, number in labels.items() if number == 2]
    if len(repeated) != 1 or any(number not in (1, 2) for number in labels.values()):
        return None
    return {"fillers": fillers, "hub": hub, "motifs": motifs,
            "labels": labels, "answer": repeated[0], "counts": counts}


def _sample_dimensions(rng, count, regime):
    # Both regimes use the same permitted sizes; tests favor harder count/size combinations.
    bands = ((12, 19, 12, 20), (17, 25, 16, 27),
             (23, 30, 20, 30), (14, 23, 24, 30), (24, 30, 14, 23))
    weights = (0.24, 0.28, 0.22, 0.13, 0.13) if regime == "source_train" else (
        0.13, 0.22, 0.36, 0.14, 0.15)
    for _ in range(60):
        low_h, high_h, low_w, high_w = bands[int(rng.choice(5, p=weights))]
        height = int(rng.integers(low_h, high_h + 1))
        width = int(rng.integers(low_w, high_w + 1))
        if height * width >= 68 + 26 * count:
            return height, width
    return max(18, 3 * count), 28


def _centers(rng, height, width, count):
    places = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(12):
        chosen = []
        for index in rng.permutation(len(places)):
            row, col = places[int(index)]
            if all(max(abs(row - r), abs(col - c)) >= 3 for r, c in chosen):
                chosen.append((row, col))
                if len(chosen) == count:
                    return chosen
    return None


def _background(rng, height, width, dominant, minor, regime):
    mode = int(rng.choice(3, p=(0.36, 0.37, 0.27)))
    minor_probability = float(rng.uniform(0.17, 0.37))
    if mode == 2:
        coarse = rng.random(((height + 3) // 4, (width + 3) // 4))
        patch = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)[:height, :width]
        local_probability = np.where(patch < 0.42, minor_probability * 0.35,
                                     np.minimum(0.58, minor_probability * 1.65))
        mask = rng.random((height, width)) < local_probability
    else:
        mask = rng.random((height, width)) < minor_probability
    if mode == 1:
        clean_columns = rng.choice(width, size=int(rng.integers(2, min(7, width))),
                                   replace=False)
        clean_rows = rng.choice(height, size=int(rng.integers(1, min(5, height))),
                                replace=False)
        mask[:, clean_columns] = False
        mask[clean_rows, :] = False
    return np.where(mask, minor, dominant).astype(np.int8)


def _put_extra(grid, protected, row, col, color, fillers):
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return False
    if (row, col) in protected or int(grid[row, col]) not in fillers:
        return False
    grid[row, col] = color
    return True


def _decorate(rng, grid, protected, centers, labels, used, hub, fillers, level):
    if level == 0:
        return
    # Diagonals are free to disagree with the cardinal arms of a complete motif.
    diagonal_probability = (0, 0.16, 0.35, 0.60)[level]
    for row, col in centers:
        for dr in (-1, 1):
            for dc in (-1, 1):
                if rng.random() < diagonal_probability:
                    _put_extra(grid, protected, row + dr, col + dc,
                               int(rng.choice(used)), fillers)
    cluster_count = int(rng.integers(0, level + 1))
    for _ in range(cluster_count):
        row = int(rng.integers(grid.shape[0]))
        col = int(rng.integers(grid.shape[1]))
        color = int(rng.choice(used))
        if _put_extra(grid, protected, row, col, color, fillers):
            direction = _DIRECTIONS[int(rng.integers(4))]
            _put_extra(grid, protected, row + direction[0], col + direction[1],
                       color, fillers)
    # Each label receives an independent amount of noise, including zero.
    for color in used:
        amount = int(rng.integers(0, (0, 4, 7, 13)[level]))
        if level == 3 and rng.random() < 0.25:
            amount += int(rng.integers(4, 10))
        for _ in range(amount):
            for _ in range(25):
                row = int(rng.integers(grid.shape[0]))
                col = int(rng.integers(grid.shape[1]))
                if _put_extra(grid, protected, row, col, color, fillers):
                    break


def _make_pair(rng, regime, roles, motif_count, level):
    minor, hub, dominant = roles
    remaining = [color for color in range(10) if color not in roles]
    for _ in range(70):
        height, width = _sample_dimensions(rng, motif_count, regime)
        extra_hubs = (0, int(rng.integers(0, 2)), int(rng.integers(1, 3)),
                      int(rng.integers(2, 5)))[level]
        locations = _centers(rng, height, width, motif_count + extra_hubs)
        if locations is None:
            continue
        answer = int(rng.choice(remaining))
        other = [color for color in remaining if color != answer]
        picked = [int(color) for color in rng.choice(other, size=motif_count - 2,
                                                     replace=False)]
        labels = [answer, answer] + picked
        rng.shuffle(labels)
        used = sorted(set(labels))
        grid = _background(rng, height, width, dominant, minor, regime)
        protected = set()
        for row, col in locations:
            protected.update(_plus(row, col))
        for (row, col), color in zip(locations[:motif_count], labels):
            grid[row, col] = hub
            for dr, dc in _DIRECTIONS:
                grid[row + dr, col + dc] = color
        for row, col in locations[motif_count:]:
            grid[row, col] = hub
            if level == 3 or (level == 2 and rng.random() < 0.55):
                # Four occupied neighbors, but at least two distinct arm labels.
                colors = [int(rng.choice(used))] * 3
                colors.append(int(rng.choice([x for x in used if x != colors[0]])))
                rng.shuffle(colors)
                for (dr, dc), color in zip(_DIRECTIONS, colors):
                    grid[row + dr, col + dc] = color
            else:
                for index in rng.permutation(4)[:int(rng.integers(1, 4))]:
                    dr, dc = _DIRECTIONS[int(index)]
                    grid[row + dr, col + dc] = int(rng.choice(used))
        _decorate(rng, grid, protected, locations[:motif_count], labels, used,
                  hub, (dominant, minor), level)
        info = _inspect(grid)
        if info is None or info["hub"] != hub or info["answer"] != answer:
            continue
        if len(info["motifs"]) != motif_count or info["fillers"] != (dominant, minor):
            continue
        if int(info["counts"][minor]) < max(int(info["counts"][color]) for color in used) + 2:
            continue
        if (height, width, hashlib.sha256(grid.tobytes()).hexdigest()) in _SOURCE_HASHES:
            continue
        return {"input": grid, "output": np.array([[answer]], dtype=np.int8)}
    raise RuntimeError("could not construct a pair")


def _array_key(array):
    return (array.shape, array.astype(np.int8, copy=False).tobytes())


def _same_puzzles(a, b):
    return all(len(a[section]) == len(b[section]) and
               all(np.array_equal(pa[name], pb[name])
                   for pa, pb in zip(a[section], b[section])
                   for name in ("input", "output"))
               for section in ("train", "test"))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _count(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    convention = int(rng.choice(3))  # literal 4/0, other shared roles, or local roles
    if convention == 0:
        shared_hub, shared_minor = 4, 0
    else:
        shared_hub, shared_minor = [int(x) for x in rng.choice(10, size=2,
                                                                 replace=False)]
    total = train_count + test_count
    levels = [0, 1, 3] + [int(rng.integers(4)) for _ in range(total - 3)]
    rng.shuffle(levels)
    train_motifs = [int(rng.integers(3, 9)) for _ in range(train_count)]
    if train_motifs[1] == train_motifs[0]:
        train_motifs[1] = 3 + (train_motifs[0] - 2) % 6
    test_motifs = [int(rng.integers(3, 9)) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    roles_seen = []
    for index in range(total):
        regime = "source_train" if index < train_count else "target_test"
        motif_count = (train_motifs[index] if index < train_count else
                       test_motifs[index - train_count])
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(25):
            if convention == 2:
                hub, minor = [int(x) for x in pair_rng.choice(10, size=2,
                                                               replace=False)]
                if index == 1 and (hub, minor) == roles_seen[0][:2]:
                    continue
            else:
                hub, minor = shared_hub, shared_minor
            dominant = int(pair_rng.choice([color for color in range(10)
                                            if color not in (hub, minor)]))
            roles = (minor, hub, dominant)
            pair = _make_pair(pair_rng, regime, roles, motif_count, levels[index])
            key = _array_key(pair["input"])
            if key in seen:
                continue
            # Two different training outputs make the role rule visible even with two examples.
            if index == 1 and int(pair["output"][0, 0]) == int(puzzle["train"][0]["output"][0, 0]):
                continue
            seen.add(key)
            roles_seen.append((hub, minor, dominant))
            puzzle["train" if index < train_count else "test"].append(pair)
            break
        else:
            raise RuntimeError("could not make distinct puzzle pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair):
    grid, output = pair["input"], pair["output"]
    assert output.shape == (1, 1)
    info = _inspect(grid)
    assert info is not None
    assert 3 <= len(info["motifs"]) <= 8
    assert len(info["labels"]) == len(info["motifs"]) - 1
    assert int(output[0, 0]) == info["answer"]
    assert info["answer"] not in info["fillers"] + (info["hub"],)
    assert (grid.shape[0], grid.shape[1], hashlib.sha256(grid.tobytes()).hexdigest()) not in _SOURCE_HASHES
    counts = info["counts"]
    assert int(counts[info["fillers"][0]]) > int(counts[info["fillers"][1]])
    assert int(counts[info["fillers"][1]]) > max(int(counts[c]) for c in info["labels"])
    assert set(int(c) for c in np.unique(grid)) == set(info["fillers"]) | {info["hub"]} | set(info["labels"])
    extra_hubs = int(counts[info["hub"]]) - len(info["motifs"])
    extras = sum(int(counts[color]) - 4 * number for color, number in info["labels"].items())
    return info, extra_hubs, extras


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    features = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            features.append(_pair_features(pair))
    train_features = features[:len(puzzle["train"])]
    assert len({item[0]["answer"] for item in train_features}) >= 2
    assert len({len(item[0]["motifs"]) for item in train_features}) >= 2
    # A puzzle demonstrates both clean plus recognition and distractor rejection.
    assert any(extra_hubs == 0 and extras == 0 for _, extra_hubs, extras in features)
    assert any(extra_hubs >= 2 and extras >= 6 for _, extra_hubs, extras in features)
    role_pairs = {(item[0]["hub"], item[0]["fillers"][1]) for item in features}
    if len(role_pairs) > 1:
        assert len({(item[0]["hub"], item[0]["fillers"][1]) for item in train_features}) > 1
    # The complete motifs in each input determine its output without a hidden seed.
    for pair, (info, _, _) in zip(puzzle["train"] + puzzle["test"], features):
        assert int(pair["output"][0, 0]) == info["answer"]


def self_test() -> None:
    first = generate_puzzle(12345)
    assert _same_puzzles(first, generate_puzzle(12345))
    explicit = generate_puzzle(9876, 7, 3)
    assert _same_puzzles(explicit, generate_puzzle(9876, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None),
                                           (10, None))):
        puzzle = generate_puzzle(2000 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts, sizes, motif_counts = set(), set(), set(), set()
    clean, noisy, diagonal, mixed, equal_totals = 0, 0, 0, 0, 0
    clusters, clean_columns, tied_answer = 0, 0, 0
    new_test_answer, repeated_test_answer, smaller_test = 0, 0, 0
    same_shape, training_five, test_nonfive = 0, 0, 0
    fixed_original, variable_roles, reused_test_filler = 0, 0, 0
    varied_test_fillers, all_clutter = 0, 0
    answer_rank = set()
    layouts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_answers = {int(pair["output"][0, 0]) for pair in puzzle["train"]}
        first_test = int(puzzle["test"][0]["output"][0, 0])
        new_test_answer += first_test not in train_answers
        repeated_test_answer += first_test in train_answers
        smaller_test += min(p["input"].shape[0] for p in puzzle["test"]) <= max(
            p["input"].shape[0] for p in puzzle["train"])
        same_shape += bool({p["input"].shape for p in puzzle["train"]} &
                           {p["input"].shape for p in puzzle["test"]})
        train_infos = [_inspect(p["input"]) for p in puzzle["train"]]
        test_infos = [_inspect(p["input"]) for p in puzzle["test"]]
        training_five += any(len(info["motifs"]) == 5 for info in train_infos)
        test_nonfive += any(len(info["motifs"]) != 5 for info in test_infos)
        roles = {(info["hub"], info["fillers"][1]) for info in train_infos + test_infos}
        fixed_original += roles == {(4, 0)}
        variable_roles += len(roles) > 1
        train_fillers = {info["fillers"][0] for info in train_infos}
        test_fillers = {info["fillers"][0] for info in test_infos}
        reused_test_filler += bool(train_fillers & test_fillers)
        varied_test_fillers += len(test_fillers) > 1
        clutter_in_puzzle = set()
        for pair in puzzle["train"] + puzzle["test"]:
            grid = pair["input"]
            info, extra_hubs, extras = _pair_features(pair)
            sizes.add(grid.shape)
            motif_counts.add(len(info["motifs"]))
            layouts.add((grid.shape, tuple(sorted((r, c) for r, c, _ in info["motifs"]))))
            clean += extra_hubs == 0 and extras == 0
            noisy += extra_hubs >= 2 and extras >= 6
            clutter_in_puzzle.add("clean" if extra_hubs == 0 and extras == 0
                                  else "noisy" if extra_hubs >= 2 and extras >= 6
                                  else "moderate")
            minor = info["fillers"][1]
            clean_columns += sum(not np.any(grid[:, col] == minor)
                                 for col in range(grid.shape[1])) >= 3
            raw = {c: int(info["counts"][c]) for c in info["labels"]}
            answer = raw[info["answer"]]
            tied_answer += answer in [value for color, value in raw.items()
                                      if color != info["answer"]]
            answer_rank.add("high" if answer > max(v for c, v in raw.items() if c != info["answer"])
                            else "low" if answer < min(v for c, v in raw.items() if c != info["answer"])
                            else "middle")
            equal_totals += len(set(raw.values())) == 1
            occupied = set()
            for row, col, _ in info["motifs"]:
                occupied.update(_plus(row, col))
                diagonal += any(int(grid[row + dr, col + dc]) in info["labels"]
                                for dr in (-1, 1) for dc in (-1, 1))
            extra_cells = {(row, col) for row in range(grid.shape[0])
                           for col in range(grid.shape[1])
                           if int(grid[row, col]) in info["labels"]
                           and (row, col) not in occupied}
            clusters += any((row + dr, col + dc) in extra_cells and
                            grid[row, col] == grid[row + dr, col + dc]
                            for row, col in extra_cells for dr, dc in _DIRECTIONS
                            if 0 <= row + dr < grid.shape[0] and
                            0 <= col + dc < grid.shape[1])
            for row, col in np.argwhere(grid == info["hub"]):
                row, col = int(row), int(col)
                if 0 < row < grid.shape[0] - 1 and 0 < col < grid.shape[1] - 1:
                    arms = [int(grid[row + dr, col + dc]) for dr, dc in _DIRECTIONS]
                    mixed += all(color in info["labels"] for color in arms) and len(set(arms)) > 1
        all_clutter += len(clutter_in_puzzle) == 3
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert motif_counts == set(range(3, 9))
    assert len(sizes) >= 35 and len(layouts) >= 100
    assert clean >= 20 and noisy >= 20 and diagonal >= 20 and mixed >= 10
    assert clusters >= 20 and clean_columns >= 15 and all_clutter == 42
    assert equal_totals < 30 and tied_answer >= 5
    assert {"high", "low", "middle"}.issubset(answer_rank)
    assert new_test_answer > 0 and repeated_test_answer > 0 and smaller_test > 0
    assert same_shape > 0 and training_five > 0 and test_nonfive > 0
    assert fixed_original > 0 and variable_roles > 0
    assert reused_test_filler > 0 and varied_test_fillers > 0
