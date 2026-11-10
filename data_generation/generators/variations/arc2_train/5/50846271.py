"""Generate sparse-clue, equal-armed cross completion puzzles."""

import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "a6128fcda69b1f3d8e6c9c0ce0f7a29f04de3149017b94d43dd33c2e232d7ae1",
    "329b70a7128a048b3a7e2dc1fa9c0029bea99f81417495fc8e7cc26cd1e0a20a",
    "297cb660e8c871504fee0d34ea040cc7284a779a6bd95acc0beccada091848f7",
    "b21de671adfc0c1c123e2ab492abe29abc2adf595befa27fc78e3953db90888d",
    "a301d78a1173461325dd2bc17853dc91614bd23b72ee0b102e910aa475cb62fa",
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(rng, seed, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        nt = 1 + int(seed) % 4
        minimum = max(2, nt + 1)
        return minimum + (3 * int(seed) + int(seed) // 4) % (11 - minimum), nt
    if num_train is None:
        nt = int(num_test)
        return int(rng.integers(max(2, nt + 1), 11)), nt
    ntr = int(num_train)
    return ntr, int(rng.integers(1, min(4, ntr - 1) + 1))


def _cells(height, width, row, col, arm):
    result = {(row, col)}
    for d in range(1, arm + 1):
        for r, c in ((row-d, col), (row+d, col),
                     (row, col-d), (row, col+d)):
            if 0 <= r < height and 0 <= c < width:
                result.add((r, c))
    return frozenset(result)


def _tips(height, width, row, col, arm):
    return frozenset((r, c) for r, c in
                     ((row-arm, col), (row+arm, col),
                      (row, col-arm), (row, col+arm))
                     if 0 <= r < height and 0 <= c < width)


def _guards(height, width, row, col, arm):
    return _tips(height, width, row, col, arm + 1)


def _clipped(height, width, row, col, arm):
    return row < arm or col < arm or row + arm >= height or col + arm >= width


def _candidate_crosses(grid, background, clue, arm):
    height, width = grid.shape
    clues = [tuple(map(int, pos)) for pos in np.argwhere(grid == clue)]
    clue_index = {pos: i for i, pos in enumerate(clues)}
    centers = set()
    for r, c in clues:
        centers.add((r, c))
        for d in range(1, arm + 1):
            centers.update(((r-d, c), (r+d, c), (r, c-d), (r, c+d)))
    candidates = []
    for row, col in centers:
        if not (0 <= row < height and 0 <= col < width):
            continue
        cells = _cells(height, width, row, col, arm)
        if len(cells) < 7 or any(int(grid[p]) == background for p in cells):
            continue
        mask = 0
        present = []
        for pos in cells:
            if pos in clue_index:
                mask |= 1 << clue_index[pos]
                present.append(pos)
        if len(present) < 3 or len(cells) - len(present) < 1:
            continue
        tips = _tips(height, width, row, col, arm)
        tip_count = len(tips.intersection(present))
        horizontal = any(r == row and c != col for r, c in present)
        vertical = any(c == col and r != row for r, c in present)
        # A one-axis object is legible only when it spans both opposite tips.
        if not (horizontal and vertical):
            if not ((row, col-arm) in present and (row, col+arm) in present
                    or (row-arm, col) in present and (row+arm, col) in present):
                continue
        # A tipless object must have its center and a clue on each axis.
        if tip_count == 0 and not ((row, col) in present and horizontal and vertical):
            continue
        candidates.append((row, col, arm, cells, mask, tip_count))
    return candidates, len(clues)


def _solve_crosses(grid, background, clue):
    """Find the unique clue decomposition, with a common radius per grid.

    At least one cross must expose a tip to establish the grid's radius.
    Other crosses can have all tips, or even one whole axis, hidden.
    """
    solutions = []
    for arm in (2, 3):
        candidates, clue_count = _candidate_crosses(grid, background, clue, arm)
        if clue_count < 3 or not candidates:
            continue
        target = (1 << clue_count) - 1
        by_clue = [[] for _ in range(clue_count)]
        for index, candidate in enumerate(candidates):
            mask = candidate[4]
            while mask:
                bit = mask & -mask
                by_clue[bit.bit_length() - 1].append(index)
                mask ^= bit

        def search(covered, used, chosen, anchored):
            if len(solutions) >= 2:
                return
            if covered == target:
                if anchored:
                    solutions.append(tuple(candidates[i] for i in chosen))
                return
            if len(chosen) >= 4:
                return
            remaining = target & ~covered
            best = None
            while remaining:
                bit = remaining & -remaining
                options = [i for i in by_clue[bit.bit_length() - 1]
                           if not candidates[i][4] & covered
                           and candidates[i][3].isdisjoint(used)]
                if not options:
                    return
                if best is None or len(options) < len(best):
                    best = options
                remaining ^= bit
            for i in best:
                item = candidates[i]
                search(covered | item[4], used | item[3], chosen + (i,),
                       anchored or item[5] > 0)
                if len(solutions) >= 2:
                    return

        search(0, frozenset(), (), False)
        if len(solutions) >= 2:
            return None
    return solutions[0] if len(solutions) == 1 else None


def _render(grid, crosses, filler, completion):
    output = grid.copy()
    for cross in crosses:
        for pos in cross[3]:
            if grid[pos] == filler:
                output[pos] = completion
    return output


def _sample_clues(rng, height, width, row, col, arm, cells, mode):
    cells = tuple(sorted(cells))
    tips = _tips(height, width, row, col, arm)
    center = (row, col)
    if mode == "strong":
        # Opposite tips identify the midpoint and radius without a guard halo.
        axes = [((row-arm, col), (row+arm, col)),
                ((row, col-arm), (row, col+arm))]
        valid = [axis for axis in axes if all(p in cells for p in axis)]
        if not valid:
            return None
        axis = valid[int(rng.integers(len(valid)))]
        other = [p for p in cells if p not in axis and p != center
                 and ((p[0] != row) if axis[0][0] == row else (p[1] != col))]
        if not other:
            return None
        chosen = set(axis)
        chosen.add(other[int(rng.integers(len(other)))])
        pool = [p for p in cells if p not in chosen]
        maximum = min(len(cells)-2, 9 if arm == 3 else 7)
        target = int(rng.integers(3, maximum + 1))
        for p in rng.permutation(pool)[:target-len(chosen)]:
            chosen.add(tuple(p))
        return frozenset(chosen)

    if mode == "tipless":
        pool = [p for p in cells if p not in tips]
        maximum = min(len(pool), len(cells)-2)
        if maximum < 3:
            return None
        target = int(rng.integers(3, maximum + 1))
        for _ in range(25):
            chosen = frozenset(pool[int(i)] for i in
                               rng.choice(len(pool), size=target, replace=False))
            if center in chosen and any(p[0] == row and p != center for p in chosen) \
                    and any(p[1] == col and p != center for p in chosen):
                return chosen
        return None

    if mode == "hidden_axis":
        horizontal = bool(rng.integers(2))
        line = [p for p in cells if (p[0] == row if horizontal else p[1] == col)]
        ends = ((row, col-arm), (row, col+arm)) if horizontal else \
               ((row-arm, col), (row+arm, col))
        if not all(p in line for p in ends):
            return None
        target = int(rng.integers(3, min(len(line), 7 if arm == 3 else 5) + 1))
        chosen = set(ends)
        pool = [p for p in line if p not in chosen]
        for p in rng.permutation(pool)[:target-2]:
            chosen.add(tuple(p))
        return frozenset(chosen)

    raise ValueError("unknown clue mode")


def _place(rng, height, width, arm, count, clipped):
    placed = []
    for index in range(count):
        positions = []
        if clipped and index == 0:
            edge = int(rng.integers(4))
            depth = int(rng.integers(arm))
            for _ in range(80):
                if edge == 0:
                    pos = (depth, int(rng.integers(arm+1, width-arm-1)))
                elif edge == 1:
                    pos = (height-1-depth, int(rng.integers(arm+1, width-arm-1)))
                elif edge == 2:
                    pos = (int(rng.integers(arm+1, height-arm-1)), depth)
                else:
                    pos = (int(rng.integers(arm+1, height-arm-1)), width-1-depth)
                positions.append(pos)
        else:
            for _ in range(100):
                positions.append((int(rng.integers(arm, height-arm)),
                                  int(rng.integers(arm, width-arm))))
        accepted = False
        for row, col in positions:
            cells = _cells(height, width, row, col, arm)
            if any(not cells.isdisjoint(old[3]) for old in placed):
                continue
            # Clue groups should not literally run into another cross.
            if any(min(abs(r-r0)+abs(c-c0) for r, c in cells
                       for r0, c0 in old[3]) < 2 for old in placed):
                continue
            placed.append((row, col, arm, cells))
            accepted = True
            break
        if not accepted:
            return None
    return placed


def _fallback_pair(rng, regime, plan, roles):
    count, arm, clipped = plan
    height = width = 24 if regime == "source_train" else 27
    near, far = arm + 2, height - arm - 3
    if clipped:
        centers = [(0, width//2), (far, near), (far, far), (near+1, near)]
    else:
        centers = [(near, near), (far, far), (near, far), (far, near)]
    crosses = [(r, c, arm, _cells(height, width, r, c, arm))
               for r, c in centers[:count]]
    background, filler, clue, completion = roles
    for _ in range(40):
        grid = np.where(rng.random((height, width)) < .45,
                        filler, background).astype(np.int8)
        for r, c, radius, cells in crosses:
            chosen = set(cells)
            tips = _tips(height, width, r, c, radius)
            removable = [p for p in cells if p not in tips]
            maximum = min(len(cells)-2, 7 if radius == 2 else 9)
            for p in rng.permutation(removable)[:len(cells)-maximum]:
                chosen.remove(tuple(p))
            for p in cells:
                grid[p] = clue if p in chosen else filler
        recovered = _solve_crosses(grid, background, clue)
        if recovered is not None and {
            (r, c, radius, cells) for r, c, radius, cells, _, _ in recovered
        } == set(crosses):
            return grid, _render(grid, crosses, filler, completion)
    raise RuntimeError(f"fallback could not build {regime} pair {plan}")


def _generate_pair(rng, regime, plan, roles):
    count, arm, clipped = plan
    background, filler, clue, completion = roles
    for attempt in range(360):
        minimum = 9 + 2*count
        if regime == "source_train":
            height = int(rng.integers(minimum, 25))
            width = int(rng.integers(minimum, 25))
        else:
            height = int(rng.integers(minimum, 28))
            width = int(rng.integers(minimum, 28))
        density = float(rng.uniform(.34, .60))
        crosses = _place(rng, height, width, arm, count, clipped)
        if crosses is None:
            continue
        grid = np.where(rng.random((height, width)) < density,
                        filler, background).astype(np.int8)
        for cross in crosses:
            for pos in cross[3]:
                grid[pos] = filler
        # One strong object establishes this grid's scale. Additional objects
        # may have tipless or entirely hidden arms.
        modes = [str(rng.choice(["strong", "tipless", "hidden_axis"],
                                p=[.55, .24, .21])) for _ in range(count)]
        modes[int(rng.integers(count))] = "strong"
        order = rng.permutation(count)
        for index in order:
            cross = crosses[int(index)]
            selected = _sample_clues(rng, height, width, *cross, modes[int(index)])
            if selected is None:
                break
            for pos in selected:
                grid[pos] = clue
        else:
            if np.count_nonzero(grid == clue) >= np.count_nonzero(grid == background):
                continue
            if np.count_nonzero(grid == clue) >= np.count_nonzero(grid == filler):
                continue
            intended = {(r, c, a, cells) for r, c, a, cells in crosses}
            recovered = _solve_crosses(grid, background, clue)
            if recovered is None or {(r, c, a, cells) for r, c, a, cells, _, _
                                     in recovered} != intended:
                continue
            output = _render(grid, crosses, filler, completion)
            if np.array_equal(grid, output):
                continue
            return grid, output
    return _fallback_pair(rng, regime, plan, roles)


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _source_digest(grid):
    normalized = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(bytes(normalized.shape) + normalized.tobytes()).hexdigest()


def _plans(rng, train_count, test_count):
    train_counts = [1, int(rng.integers(2, 4))]
    train_counts += [int(rng.integers(1, 4)) for _ in range(train_count-2)]
    rng.shuffle(train_counts)
    train_arms = [2, 3] + [int(rng.choice([2, 3], p=[.67, .33]))
                            for _ in range(train_count-2)]
    rng.shuffle(train_arms)
    train_clip = [False] * train_count
    if rng.random() < .68:
        train_clip[int(rng.integers(train_count))] = True
        for i in range(train_count):
            if rng.random() < .13:
                train_clip[i] = True
    train = list(zip(train_counts, train_arms, train_clip))
    test = []
    for _ in range(test_count):
        count = int(rng.choice([1, 2, 3, 4], p=[.12, .24, .28, .36]))
        arm = int(rng.choice([2, 3], p=[.73, .27]))
        clip = bool(any(train_clip) and rng.random() < .20)
        test.append((count, arm, clip))
    return train, test


def _local_roles(rng, base, variation, first_training_pair):
    if variation == "fixed" or first_training_pair:
        return base
    background, filler, clue, completion = base
    available = [color for color in range(10) if color not in (filler, completion)]
    if variation == "background":
        choices = [color for color in available if color != clue and color != background]
        return int(rng.choice(choices)), filler, clue, completion
    if variation == "clue":
        choices = [color for color in available if color != background and color != clue]
        return background, filler, int(rng.choice(choices)), completion
    backgrounds = [color for color in available if color != background and color != clue]
    new_background = int(rng.choice(backgrounds))
    new_clues = [color for color in available
                 if color != clue and color != background and color != new_background]
    return new_background, filler, int(rng.choice(new_clues)), completion


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    roles = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    variation = str(rng.choice(["fixed", "background", "clue", "both"],
                               p=[.65, .12, .11, .12]))
    train_plans, test_plans = _plans(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, regime, plans in (("train", "source_train", train_plans),
                                 ("test", "target_test", test_plans)):
        for index, plan in enumerate(plans):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
            pair_roles = _local_roles(rng, roles, variation,
                                      split == "train" and index == 0)
            for _ in range(5):
                input_grid, output_grid = _generate_pair(pair_rng, regime, plan,
                                                         pair_roles)
                key = _grid_key(input_grid)
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("duplicate generated input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_convention(train):
    filler = completion = None
    palettes = []
    for pair in train:
        before, after = pair["input"], pair["output"]
        change = before != after
        _require(bool(np.any(change)), "every pair needs a completion")
        source = np.unique(before[change])
        target = np.unique(after[change])
        _require(len(source) == len(target) == 1, "completion must have one role color")
        current_palette = frozenset(map(int, np.unique(before)))
        _require(len(current_palette) == 3, "input needs three role colors")
        palettes.append(current_palette)
        if filler is None:
            filler, completion = int(source[0]), int(target[0])
        else:
            _require((filler, completion) == (int(source[0]), int(target[0])),
                     "puzzle color roles changed")
    _require(all(completion not in palette for palette in palettes),
             "completion color must be new")
    return filler, completion, palettes[0], len(set(palettes)) > 1


def _pair_roles(grid, filler, completion):
    palette = frozenset(map(int, np.unique(grid)))
    _require(len(palette) == 3 and filler in palette and completion not in palette,
             "pair palette does not expose its color roles")
    remaining = list(palette - {filler})
    amounts = {color: int(np.count_nonzero(grid == color)) for color in remaining}
    clue = min(remaining, key=amounts.get)
    background = max(remaining, key=amounts.get)
    _require(amounts[clue] < amounts[background], "clue must be the sparse role")
    _require(amounts[clue] < int(np.count_nonzero(grid == filler)),
             "clue must be sparse relative to filler")
    return background, filler, clue, completion


def _analyze_pair(pair, roles):
    grid, output = pair["input"], pair["output"]
    background, filler, clue, completion = roles
    _require(set(map(int, np.unique(grid))) == {background, filler, clue},
             "input palette is inconsistent")
    _require(set(map(int, np.unique(output))) ==
             {background, filler, clue, completion}, "output palette is inconsistent")
    crosses = _solve_crosses(grid, background, clue)
    _require(crosses is not None, "crosses are not uniquely recoverable")
    _require(np.array_equal(_render(grid, crosses, filler, completion), output),
             "output does not complete the recovered crosses")
    _require(len(crosses) in (1, 2, 3, 4), "unsupported cross count")
    arms = {cross[2] for cross in crosses}
    _require(len(arms) == 1, "arm length must be common within a grid")
    _require(any(
        ((r-arm, c) in clues and (r+arm, c) in clues)
        or ((r, c-arm) in clues and (r, c+arm) in clues)
        for r, c, arm, cells, _, _ in crosses
        for clues in ({p for p in cells if grid[p] == clue},)
    ), "one cross must expose opposite tips to establish the radius")
    union = frozenset().union(*(cross[3] for cross in crosses))
    _require(any(grid[p] == filler for p in np.ndindex(grid.shape) if p not in union),
             "unrelated filler is required")
    for cross in crosses:
        clues = [p for p in cross[3] if grid[p] == clue]
        _require(len(clues) >= 3 and len(cross[3])-len(clues) >= 1,
                 "cross needs clues and a hidden portion")
    return crosses


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have exactly input and output")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                     np.issubdtype(grid.dtype, np.integer), "grid must be a 2D integer array")
            _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions out of range")
            _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                     "invalid ARC color")
        before, after = pair["input"], pair["output"]
        _require(before.shape == after.shape, "dimensions changed")
        key = _grid_key(before)
        _require(key not in seen, "duplicate input")
        seen.add(key)
        _require(_source_digest(before) not in _SOURCE_INPUT_DIGESTS,
                 "input copies a source pair")
    filler, completion, first_palette, local = _infer_convention(train)
    analyses = []
    for pair in train + test:
        if not local:
            _require(frozenset(map(int, np.unique(pair["input"]))) == first_palette,
                     "test palette changed without training demonstration")
        roles = _pair_roles(pair["input"], filler, completion)
        analyses.append(_analyze_pair(pair, roles))
    train_crosses = analyses[:len(train)]
    test_crosses = analyses[len(train):]
    _require({cross[2] for group in train_crosses for cross in group} == {2, 3},
             "training must demonstrate both supported scales")
    _require(1 in [len(group) for group in train_crosses] and
             any(len(group) >= 2 for group in train_crosses),
             "training must show single and multiple crosses")
    train_has_clip = any(_clipped(*pair["input"].shape, *cross[:3])
                         for pair, group in zip(train, train_crosses) for cross in group)
    for pair, group in zip(test, test_crosses):
        if any(_clipped(*pair["input"].shape, *cross[:3]) for cross in group):
            _require(train_has_clip, "test clipping was not demonstrated")
    for pair, group in zip(train + test, analyses):
        h, w = pair["input"].shape
        _require(h >= 9 + 2*len(group) and w >= 9 + 2*len(group),
                 "grid too small for its cross count")


def _fingerprint(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4,
             "train count diversity is too narrow")
    _require(len({len(p["test"]) for p in puzzles}) >= 3,
             "test count diversity is too narrow")
    _require(len({_fingerprint(p) for p in puzzles}) == 42,
             "seeds must make different puzzles")
    dimensions = set()
    palettes = set()
    count_arm = set()
    densities = {"train": [], "test": []}
    tipless = hidden_axis = high_clue = guard_filler = clipped = 0
    no_horizontal = no_vertical = clipped_tipless = all_guard_filler = 0
    old_spacing_rejected = guard_positions = guard_filler_positions = 0
    local_colors = background_changes = clue_changes = 0
    train_denser = test_more = 0
    clue_counts = {2: set(), 3: set()}
    clip_edges = set()
    center_colors = set()
    for puzzle in puzzles:
        filler, completion, _, local = _infer_convention(puzzle["train"])
        train_roles = tuple(_pair_roles(pair["input"], filler, completion)
                            for pair in puzzle["train"])
        palettes.add(train_roles)
        local_colors += local
        background_changes += len({roles[0] for roles in train_roles}) > 1
        clue_changes += len({roles[2] for roles in train_roles}) > 1
        pair_densities = {"train": [], "test": []}
        pair_counts = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                roles = _pair_roles(grid, filler, completion)
                crosses = _analyze_pair(pair, roles)
                dimensions.add(grid.shape)
                count_arm.add((split, len(crosses), crosses[0][2]))
                density = float(np.mean(grid == roles[1]))
                densities[split].append(density)
                pair_densities[split].append(density)
                pair_counts[split].append(len(crosses))
                for index, (r, c, arm, cells, _, _) in enumerate(crosses):
                    clues = {p for p in cells if grid[p] == roles[2]}
                    tips = _tips(*grid.shape, r, c, arm)
                    guards = _guards(*grid.shape, r, c, arm)
                    missing_horizontal = not any(p[0] == r and p[1] != c for p in clues)
                    missing_vertical = not any(p[1] == c and p[0] != r for p in clues)
                    is_clipped = _clipped(*grid.shape, r, c, arm)
                    tipless += not bool(clues & tips)
                    hidden_axis += missing_horizontal or missing_vertical
                    no_horizontal += missing_horizontal
                    no_vertical += missing_vertical
                    high_clue += len(clues) >= 7
                    clue_counts[arm].add(len(clues))
                    center_colors.add(int(grid[r, c] == roles[2]))
                    filler_guards = sum(int(grid[p] == filler) for p in guards)
                    guard_positions += len(guards)
                    guard_filler_positions += filler_guards
                    guard_filler += filler_guards > 0
                    all_guard_filler += bool(guards) and filler_guards == len(guards)
                    clipped += is_clipped
                    clipped_tipless += is_clipped and not bool(clues & tips)
                    if is_clipped:
                        if r < arm: clip_edges.add("top")
                        if r + arm >= grid.shape[0]: clip_edges.add("bottom")
                        if c < arm: clip_edges.add("left")
                        if c + arm >= grid.shape[1]: clip_edges.add("right")
                    for other in crosses[index+1:]:
                        old_spacing_rejected += (abs(r-other[0]) <= arm+other[2]+2
                                                 and abs(c-other[1]) <= arm+other[2]+2)
        train_denser += np.mean(pair_densities["train"]) > np.mean(pair_densities["test"])
        test_more += max(pair_counts["test"]) > max(pair_counts["train"])
    _require(len(dimensions) >= 25 and len(palettes) >= 20 and len(count_arm) >= 10,
             "structural diversity is too narrow")
    _require(tipless > 0 and hidden_axis > 0 and high_clue > 0
             and guard_filler > 0 and clipped > 0,
             "a reviewed cross variation is absent")
    _require(no_horizontal > 0 and no_vertical > 0 and clipped_tipless > 0
             and all_guard_filler > 0 and old_spacing_rejected > 0,
             "clue, boundary, guard, or spacing combinations are absent")
    _require(0.2 < guard_filler_positions / guard_positions < 0.8,
             "guard cells are biased toward one field color")
    _require(clue_counts[2] >= {3, 7} and clue_counts[3] >= {3, 8, 9}
             and center_colors == {0, 1},
             "clue subsets do not span hidden and nearly complete crosses")
    _require(clip_edges == {"top", "bottom", "left", "right"},
             "clipping does not reach all four edges")
    _require(local_colors > 0 and background_changes > 0 and clue_changes > 0,
             "pair-local color variation is absent")
    _require(train_denser > 0 and test_more > 0,
             "train/test regimes lack density and object-count overlap")
    _require(min(densities["train"]) < .42 and max(densities["train"]) > .53
             and min(densities["test"]) < .42 and max(densities["test"]) > .53,
             "filler densities do not span sparse and dense fields")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        p = generate_puzzle(2000+i, *counts)
        _require((len(p["train"]), len(p["test"])) == counts,
                 "explicit counts were not honored")
    for i, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                {"num_train": 2}, {"num_train": 10})):
        p = generate_puzzle(3000+i, **kwargs)
        for key, value in kwargs.items():
            _require(len(p["train" if key == "num_train" else "test"]) == value,
                     "partial count was not honored")
    for i, kwargs in enumerate(({"num_train": 1}, {"num_train": 11},
                                {"num_test": 0}, {"num_test": 5},
                                {"num_train": 2, "num_test": 2},
                                {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(4000+i, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    _require(_fingerprint(generate_puzzle(987654, 7, 3)) ==
             _fingerprint(generate_puzzle(987654, 7, 3)),
             "same seed and counts are not deterministic")
