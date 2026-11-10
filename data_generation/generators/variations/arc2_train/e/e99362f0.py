"""Cross-divided panels composed with a learned quadrant priority."""

import hashlib
from itertools import permutations

import numpy as np


# Quadrants are top-left, top-right, bottom-left, bottom-right.
_SOURCE_ORDER = (3, 0, 1, 2)
_SOURCE_INPUT_HASHES = {
    "ec1b118962b53d96825471ec5afd9f8bbf8675e58891df461d3d7aa05b291421",
    "46f98cfb8fd08245d4ad9acbc18f2ff998683207556a14c8f4eb118a152db71e",
    "9c434c24dd57f9714e09f306414bd550f52d2ee8a73457d46ded168bf30bb091",
    "654c8f002fd3b89156c1fefa29433a88377a5a03a07e916f49e32801d2c87b81",
    "92514ad2d7b86d4c4b070a58de5ae0b1769e163757da2ddfa2bed121ac0543ee",
    "41929a2999d8eca7e460c5b6ed3af9b83b0a36b9ab799cb8d7f340e799f67e6d",
    "7a2346b6d41ee378b1d265b72e64dd0b36d00ca188d1c1de373c2ffe955136d0",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _check_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _check_count("num_train", num_train, 2, 10)
    num_test = _check_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return num_train, num_test


def _adjust_count(mask, goal, rng):
    flat = mask.reshape(-1)
    occupied = np.flatnonzero(flat)
    if len(occupied) > goal:
        flat[rng.choice(occupied, size=len(occupied) - goal, replace=False)] = False
    elif len(occupied) < goal:
        empty = np.flatnonzero(~flat)
        flat[rng.choice(empty, size=goal - len(occupied), replace=False)] = True
    return mask


def _random_mask(rng, height, width, dense=False, sparse=False, mixed=False):
    size = height * width
    kind = int(rng.choice(7, p=(.06, .06, .17, .17, .29, .16, .09)))
    ranges = (
        (0, 0),
        (size, size),
        (1, max(1, size // 8)),
        (max(1, size // 8), max(1, size // 3)),
        (max(1, size // 3), max(1, 2 * size // 3)),
        (max(1, 2 * size // 3), max(1, 9 * size // 10)),
        (max(1, 9 * size // 10), size - 1),
    )
    low, high = ranges[kind]
    if dense and not mixed:
        low = max(low, size // 2)
        high = max(high, low)
    if sparse:
        low = min(low, max(1, size // 5))
        high = min(high, max(low, size // 3))
    if mixed:
        low = max(1, min(low, size - 1))
        high = max(low, min(high, size - 1))
        if low == high and kind in (0, 1):
            low, high = max(1, size // 4), min(size - 1, 3 * size // 4)
    goal = int(rng.integers(low, high + 1))
    mask = np.zeros((height, width), dtype=bool)
    if goal == 0:
        return mask
    if goal == size:
        mask[:] = True
        return mask
    style = int(rng.integers(6))
    if style == 0:
        mask.reshape(-1)[rng.choice(size, goal, replace=False)] = True
    elif style == 1:
        row, col = int(rng.integers(height)), int(rng.integers(width))
        steps = 0
        while int(mask.sum()) < goal and steps < size * 30:
            mask[row, col] = True
            if rng.random() < .11:
                row, col = int(rng.integers(height)), int(rng.integers(width))
            else:
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
                row = min(height - 1, max(0, row + dr))
                col = min(width - 1, max(0, col + dc))
            steps += 1
    elif style == 2:
        for _ in range(max(1, size // 3)):
            if int(mask.sum()) >= goal:
                break
            top, left = int(rng.integers(height)), int(rng.integers(width))
            bottom = min(height, top + int(rng.integers(1, max(2, height // 2 + 1))))
            right = min(width, left + int(rng.integers(1, max(2, width // 2 + 1))))
            mask[top:bottom, left:right] = True
    elif style == 3:
        for _ in range(max(1, height + width)):
            if int(mask.sum()) >= goal:
                break
            if rng.random() < .5:
                row = int(rng.integers(height))
                a, b = sorted(rng.integers(width, size=2))
                mask[row, a:b + 1] = True
            else:
                col = int(rng.integers(width))
                a, b = sorted(rng.integers(height, size=2))
                mask[a:b + 1, col] = True
    elif style == 4:
        rows, cols = np.indices((height, width))
        modulus = int(rng.integers(2, 7))
        mask = ((int(rng.integers(1, 7)) * rows +
                 int(rng.integers(1, 7)) * cols +
                 int(rng.integers(modulus))) % modulus) < int(rng.integers(1, modulus))
    else:
        rows, cols = np.indices((height, width))
        center_r, center_c = rng.uniform(0, height), rng.uniform(0, width)
        radius = rng.uniform(1, max(height, width))
        mask = ((rows - center_r) ** 2 + (cols - center_c) ** 2 <= radius ** 2)
        if rng.random() < .5:
            mask = ~mask
    return _adjust_count(mask, goal, rng)


def _assemble(masks, background, divider, payloads):
    height, width = masks.shape[1:]
    grid = np.full((2 * height + 1, 2 * width + 1), background, dtype=np.int8)
    grid[height, :] = divider
    grid[:, width] = divider
    regions = (
        (slice(0, height), slice(0, width)),
        (slice(0, height), slice(width + 1, None)),
        (slice(height + 1, None), slice(0, width)),
        (slice(height + 1, None), slice(width + 1, None)),
    )
    for q, region in enumerate(regions):
        grid[region][masks[q]] = payloads[q]
    return grid


def _compose(masks, background, payloads, order):
    output = np.full(masks.shape[1:], background, dtype=np.int8)
    for q in reversed(order):
        if payloads[q] is not None:
            output[masks[q]] = payloads[q]
    return output


def _make_pair(rng, height, width, roles, order, anchors=(), full_union=False,
               dense=False, sparse=False, hide_lower=False):
    background, divider, *payloads = roles
    protected_pool = [q for q in range(4) if not (hide_lower and q == order[0])]
    protected = set(int(q) for q in rng.choice(protected_pool, size=2, replace=False))
    masks = np.stack([
        _random_mask(rng, height, width, dense=dense, sparse=sparse,
                     mixed=q in protected)
        for q in range(4)
    ])
    anchor_positions = rng.permutation(height * width)[:len(anchors)]
    for flat, pattern in zip(anchor_positions, anchors):
        row, col = divmod(int(flat), width)
        masks[:, row, col] = False
        for q in pattern:
            masks[q, row, col] = True
    if hide_lower:
        masks[order[0], :, :] = True
    if full_union:
        choices = [q for q in range(4) if q not in protected]
        for flat in np.flatnonzero(~np.any(masks, axis=0)):
            row, col = divmod(int(flat), width)
            masks[choices[int(rng.integers(len(choices)))], row, col] = True
    # Anchors are sparse enough that the protected panels usually stay mixed.
    # Repair only unanchored cells, preserving every required demonstration.
    reserved = set(int(x) for x in anchor_positions)
    for q in protected:
        for desired in (True, False):
            if np.any(masks[q] == desired):
                continue
            available = [flat for flat in range(height * width) if flat not in reserved]
            flat = available[int(rng.integers(len(available)))]
            row, col = divmod(flat, width)
            if not desired and full_union and int(masks[:, row, col].sum()) == 1:
                other = next(x for x in range(4) if x not in protected)
                masks[other, row, col] = True
            masks[q, row, col] = desired
    return {
        "input": _assemble(masks, background, divider, payloads),
        "output": _compose(masks, background, payloads, order),
    }


def _grid_key(grid):
    return (grid.shape, grid.astype(np.uint8, copy=False).tobytes())


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    rows, cols = inp.shape
    _require(rows % 2 == 1 and cols % 2 == 1, "input must have odd dimensions")
    height, width = (rows - 1) // 2, (cols - 1) // 2
    _require(3 <= height <= 14 and 3 <= width <= 14, "panel size outside range")
    divider = int(inp[height, width])
    _require(np.all(inp[height, :] == divider) and np.all(inp[:, width] == divider),
             "incomplete divider cross")
    panels = (
        inp[:height, :width], inp[:height, width + 1:],
        inp[height + 1:, :width], inp[height + 1:, width + 1:],
    )
    color_sets = [set(map(int, np.unique(panel))) for panel in panels]
    _require(all(1 <= len(s) <= 2 and divider not in s for s in color_sets),
             "panels must contain background and at most one payload")
    shared = [c for c in range(10) if sum(c in s for s in color_sets) >= 2]
    _require(len(shared) == 1, "background is not uniquely identified")
    background = shared[0]
    _require(sum(len(s) == 2 and background in s for s in color_sets) >= 2,
             "at least two panels must display both colors")
    payloads = []
    for colors in color_sets:
        _require(len(colors - {background}) <= 1, "multiple payloads in one panel")
        payloads.append(next(iter(colors - {background}), None))
    active = [c for c in payloads if c is not None]
    _require(len(active) == len(set(active)), "payloads must be distinct")
    masks = np.stack([
        np.zeros((height, width), dtype=bool) if payloads[q] is None
        else panels[q] == payloads[q]
        for q in range(4)
    ])
    _require(out.shape == (height, width), "output must be one panel")
    edges = set()
    patterns = set()
    visible = set()
    for row in range(height):
        for col in range(width):
            occupied = [q for q in range(4) if masks[q, row, col]]
            patterns.add(sum(1 << q for q in occupied))
            color = int(out[row, col])
            if not occupied:
                _require(color == background, "uncovered output is not background")
            else:
                winners = [q for q in occupied if payloads[q] == color]
                _require(len(winners) == 1, "output does not select an occupied layer")
                winner = winners[0]
                visible.add(winner)
                edges.update((winner, q) for q in occupied if q != winner)
    return {
        "masks": masks, "background": background, "divider": divider,
        "payloads": tuple(payloads), "edges": edges, "patterns": patterns,
        "visible": visible, "full_union": 0 not in patterns,
        "height": height, "width": width,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys must be train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair keys must be input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimensions outside ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         np.all((grid >= 0) & (grid <= 9)), "invalid grid colors")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate grid or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if pair["input"].shape == (11, 9):
                digest = hashlib.sha256(input_key[1]).hexdigest()
                _require(digest not in _SOURCE_INPUT_HASHES, "exact source input copied")
            analyses[split].append(_analyze_pair(pair))
    training = analyses["train"]
    _require(any(0 in a["patterns"] for a in training),
             "training does not show transparent background")
    _require(any(a["full_union"] for a in training),
             "training does not show complete coverage")
    _require(set.union(*(a["visible"] for a in training)) == set(range(4)),
             "training does not visibly demonstrate every quadrant")
    edges = set.union(*(a["edges"] for a in training))
    possible = [order for order in permutations(range(4))
                if all(order.index(winner) < order.index(loser)
                       for winner, loser in edges)]
    _require(len(possible) == 1, "training does not determine one priority order")
    order = possible[0]
    for split in ("train", "test"):
        for pair, analysis in zip(puzzle[split], analyses[split]):
            expected = _compose(analysis["masks"], analysis["background"],
                                analysis["payloads"], order)
            _require(np.array_equal(pair["output"], expected),
                     "pair contradicts learned quadrant order")


def _fresh_roles(rng, base):
    for _ in range(12):
        roles = tuple(map(int, rng.permutation(10)[:6]))
        if roles[0] != base[0] and roles[1] != base[1]:
            return roles
    return tuple(base[(i + 2) % 6] for i in range(6))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    n_train, n_test = _resolve_counts(rng, num_train, num_test)
    mode = int(rng.choice(3, p=(.12, .56, .32)))  # stable, varied, size/density shift
    order = _SOURCE_ORDER if rng.random() < .27 else tuple(map(int, rng.permutation(4)))
    base_roles = tuple(map(int, rng.permutation(10)[:6]))
    base_size = (int(rng.integers(4, 9)), int(rng.integers(4, 9)))
    evidence = [(order[i], order[i + 1]) for i in range(3)]
    evidence += [(q,) for q in range(4)]
    rng.shuffle(evidence)
    anchors = [[], []]
    for index, pattern in enumerate(evidence):
        anchors[index % 2].append(pattern)
    anchors[0].append(())
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n_train + n_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(n_train + n_test):
        split = "train" if index < n_train else "test"
        local_index = index if split == "train" else index - n_train
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        if mode == 0:
            height, width = base_size
            roles = base_roles
        else:
            if mode == 2:
                low, high = (3, 7) if split == "train" else (8, 15)
            else:
                low, high = 3, 15
            height = int(pair_rng.integers(low, high))
            width = int(pair_rng.integers(low, high))
            if index == 1 and (height, width) == base_size:
                width = low if width != low else high - 1
            if index == 0:
                roles = base_roles
            elif index == 1 or split == "test":
                roles = _fresh_roles(pair_rng, base_roles)
            else:
                choice = int(pair_rng.integers(3))
                if choice == 0:
                    roles = base_roles
                elif choice == 1:
                    roles = list(base_roles)
                    a, b = pair_rng.choice(4, size=2, replace=False) + 2
                    roles[int(a)], roles[int(b)] = roles[int(b)], roles[int(a)]
                    roles = tuple(roles)
                else:
                    roles = _fresh_roles(pair_rng, base_roles)
        required = anchors[local_index] if split == "train" and local_index < 2 else ()
        full_union = (local_index == 1) if split == "train" and local_index < 2 else bool(pair_rng.random() < .52)
        dense = mode == 2 and split == "test"
        sparse = mode == 2 and split == "train"
        hide_lower = split == "test" and bool(pair_rng.random() < .12)
        for _ in range(12):
            pair = _make_pair(pair_rng, height, width, roles, order,
                              anchors=required, full_union=full_union,
                              dense=dense, sparse=sparse,
                              hide_lower=hide_lower)
            key = _grid_key(pair["input"])
            if key not in seen and not (pair["input"].shape == (11, 9) and
                                        hashlib.sha256(key[1]).hexdigest() in _SOURCE_INPUT_HASHES):
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(first, second):
    return all(
        len(first[split]) == len(second[split]) and
        all(np.array_equal(a[name], b[name])
            for a, b in zip(first[split], second[split])
            for name in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "too few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "too few test counts")
    _require(_equal_puzzles(generate_puzzle(12345), generate_puzzle(12345)),
             "automatic generation is nondeterministic")
    _require(_equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3)),
             "explicit generation is nondeterministic")
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, a, b)
        _require(a is None or len(puzzle["train"]) == a, "train count ignored")
        _require(b is None or len(puzzle["test"]) == b, "test count ignored")
        validate_generated_puzzle(puzzle)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(4000, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    pairs = [pair for puzzle in puzzles for split in ("train", "test")
             for pair in puzzle[split]]
    analyses = [_analyze_pair(pair) for pair in pairs]
    sizes = {(a["height"], a["width"]) for a in analyses}
    _require(len(sizes) >= 25 and max(h for h, _ in sizes) >= 12,
             "insufficient size diversity")
    _require(any(h != w for h, w in sizes), "aspect ratio never varies")
    _require(sum(len({pair["output"].shape for split in ("train", "test")
                      for pair in puzzle[split]}) > 1 for puzzle in puzzles) >= 25,
             "too few puzzles vary size within a seed")
    _require(sum(len({a["background"] for split in ("train", "test")
                      for a in map(_analyze_pair, puzzle[split])}) > 1
                 for puzzle in puzzles) >= 25,
             "too few puzzles change background within a seed")
    orders = set()
    for puzzle in puzzles:
        edges = set.union(*(_analyze_pair(pair)["edges"] for pair in puzzle["train"]))
        orders.add(next(order for order in permutations(range(4))
                        if all(order.index(w) < order.index(l) for w, l in edges)))
    _require(len(orders) >= 10, "too few quadrant priority orders")
    _require(any(0 in a["patterns"] for a in analyses) and
             any(a["full_union"] for a in analyses), "coverage does not vary")
    _require(any(15 not in a["patterns"] for a in analyses),
             "four-way overlaps remain mandatory")
    _require(any(any(int(mask.sum()) in (0, mask.size) for mask in a["masks"])
                 for a in analyses), "no empty or full panels")
    _require(any(len(a["visible"]) < 4 for a in analyses) and
             any(len(a["visible"]) == 4 for a in analyses),
             "layer visibility does not vary")
