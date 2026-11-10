"""Generate paired-panel ARC puzzles governed by exact cellwise equality."""

import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "b95813c78165db76aeeaab041da7d801fca0660be5e8b5e54a6a15872ca07f2b",
    "72193087f81a11ae4c6f3dee05c04f002124ea7fc156afe88e046b9b3d828624",
    "f14ea1cbc99047c1878a58dd0d2e53809c6cb88adb7d70ef622c3005a80eee33",
    "f79cf60f70a9a27bf558af4cc0832b768f75755e72ab9b62546c6bf2d4d48fb6",
    "c3a77f4cba80e25143f93c881def26f2662b658ba1453fa738914002c51e78eb",
    "a338ec317bd2d8b14a8e69b757099f65c068f90ded5815e97f95d2a043d6552d",
    "a9ff2364a5f4c2edf45a6b3d1aaabd43ee72ff2f69e12022d1454ab9819f8288",
})

_SIMPLE = ("rectangle", "repeat")
_COMPLEX = ("frame", "branch", "touch", "mosaic")
_EDITS = ("single", "equal", "large", "grow", "shrink", "insert", "remove", "edge", "multi")


def _choose_counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_train = int(rng.choice(np.arange(2, 11), p=[.15, .17, .16, .14, .11, .09, .07, .06, .05]))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test) + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _box(rng, height, width, min_side=2, max_side=5, fraction=.36):
    bh = int(rng.integers(min_side, min(max_side, height) + 1))
    bw = int(rng.integers(min_side, min(max_side, width) + 1))
    while bh * bw > fraction * height * width and (bh > min_side or bw > min_side):
        if bh >= bw and bh > min_side:
            bh -= 1
        elif bw > min_side:
            bw -= 1
        else:
            break
    row = int(rng.integers(height - bh + 1))
    col = int(rng.integers(width - bw + 1))
    return row, col, bh, bw


def _make_reference(rng, height, width, colors, background, motif):
    ref = np.full((height, width), background, dtype=np.int64)
    primary = colors[0]
    if motif == "rectangle":
        row, col, bh, bw = _box(rng, height, width)
        ref[row:row + bh, col:col + bw] = primary
    elif motif == "frame":
        row, col, bh, bw = _box(rng, height, width, 3, 6, .36)
        ref[row:row + bh, col:col + bw] = primary
        ref[row + 1:row + bh - 1, col + 1:col + bw - 1] = background
        if len(colors) > 1 and bh >= 4 and bw >= 4 and rng.random() < .7:
            ref[row + 1:row + bh - 1, col + 1:col + bw - 1] = colors[1]
            if bh >= 5 and bw >= 5:
                ref[row + 2:row + bh - 2, col + 2:col + bw - 2] = background
    elif motif == "branch":
        shape = np.zeros((4, 4), dtype=bool)
        shape[1, :] = True
        shape[:, int(rng.integers(1, 3))] = True
        shape[int(rng.integers(4)), int(rng.integers(4))] = True
        shape = np.rot90(shape, int(rng.integers(4)))
        row = int(rng.integers(height - 3))
        col = int(rng.integers(width - 3))
        area = ref[row:row + 4, col:col + 4]
        area[shape] = primary
    elif motif == "repeat":
        count = int(rng.integers(2, 5))
        for _ in range(count):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            ref[row, col] = primary
            if rng.random() < .5:
                ref[row:min(row + 2, height), col] = primary
    elif motif == "touch":
        row = int(rng.integers(height - 1))
        col = int(rng.integers(width - 3))
        ref[row:row + 2, col:col + 2] = primary
        ref[row:row + 2, col + 2:col + 4] = colors[1] if len(colors) > 1 else primary
        if rng.random() < .5 and row + 2 < height:
            ref[row + 2, col + int(rng.integers(4))] = colors[-1]
    else:
        row, col, bh, bw = _box(rng, height, width, 3, 5, .45)
        for rr in range(bh):
            for cc in range(bw):
                if rng.random() < .72:
                    ref[row + rr, col + cc] = colors[(rr + cc + int(rng.integers(len(colors)))) % len(colors)]

    for color in colors:
        if not np.any(ref == color):
            free = np.argwhere(ref == background)
            rr, cc = free[int(rng.integers(len(free)))]
            ref[rr, cc] = color
    if motif != "rectangle" and rng.random() < .55 and np.count_nonzero(ref != background) < .35 * height * width:
        free = np.argwhere(ref == background)
        rr, cc = free[int(rng.integers(len(free)))]
        ref[rr, cc] = colors[int(rng.integers(len(colors)))]
    return ref


def _other_color(rng, reference, background, divider, old, prefer_new=False):
    used = set(int(x) for x in np.unique(reference))
    choices = [x for x in range(10) if x not in (background, divider, old)]
    fresh = [x for x in choices if x not in used]
    if prefer_new and fresh:
        choices = fresh
    return int(rng.choice(choices))


def _single_edit(rng, candidate, reference, background, divider):
    filled = np.argwhere(candidate != background)
    if len(filled) and rng.random() < .7:
        rr, cc = filled[int(rng.integers(len(filled)))]
        candidate[rr, cc] = _other_color(rng, reference, background, divider, int(candidate[rr, cc]), rng.random() < .6)
    else:
        free = np.argwhere(candidate == background)
        rr, cc = free[int(rng.integers(len(free)))]
        candidate[rr, cc] = int(rng.choice(np.unique(reference[reference != background])))


def _equal_histogram_edit(rng, candidate, background):
    height, width = candidate.shape
    filled = np.argwhere(candidate != background)
    options = []
    for dr in (-2, -1, 0, 1, 2):
        for dc in (-2, -1, 0, 1, 2):
            if (dr or dc) and np.all((filled[:, 0] + dr >= 0) & (filled[:, 0] + dr < height) & (filled[:, 1] + dc >= 0) & (filled[:, 1] + dc < width)):
                options.append((dr, dc))
    if options and rng.random() < .8:
        dr, dc = options[int(rng.integers(len(options)))]
        old = candidate.copy()
        candidate[:] = background
        candidate[filled[:, 0] + dr, filled[:, 1] + dc] = old[filled[:, 0], filled[:, 1]]
    else:
        first = filled[int(rng.integers(len(filled)))]
        unlike = np.argwhere(candidate != candidate[first[0], first[1]])
        second = unlike[int(rng.integers(len(unlike)))]
        candidate[first[0], first[1]], candidate[second[0], second[1]] = candidate[second[0], second[1]], candidate[first[0], first[1]]


def _edit_once(rng, candidate, reference, background, divider, kind):
    height, width = candidate.shape
    if kind == "single":
        _single_edit(rng, candidate, reference, background, divider)
    elif kind == "equal":
        _equal_histogram_edit(rng, candidate, background)
    elif kind == "large":
        filled = candidate != background
        old = int(rng.choice(np.unique(candidate[filled])))
        candidate[candidate == old] = _other_color(rng, reference, background, divider, old, True)
    elif kind == "grow":
        steps = int(rng.integers(1, max(3, min(10, height * width // 5))))
        for _ in range(steps):
            front = []
            for rr, cc in np.argwhere(candidate != background):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = rr + dr, cc + dc
                    if 0 <= r2 < height and 0 <= c2 < width and candidate[r2, c2] == background:
                        front.append((r2, c2, int(candidate[rr, cc])))
            if not front:
                break
            rr, cc, color = front[int(rng.integers(len(front)))]
            candidate[rr, cc] = color
    elif kind == "shrink":
        filled = np.argwhere(candidate != background)
        count = int(rng.integers(1, max(2, len(filled) // 2 + 1)))
        chosen = filled[rng.choice(len(filled), min(count, len(filled) - 1), replace=False)]
        candidate[chosen[:, 0], chosen[:, 1]] = background
    elif kind == "insert":
        row, col, bh, bw = _box(rng, height, width, 1, 3, .2)
        color = int(rng.choice(np.unique(reference[reference != background])))
        area = candidate[row:row + bh, col:col + bw]
        area[area == background] = color
    elif kind == "remove":
        filled = np.argwhere(candidate != background)
        color = int(rng.choice(np.unique(candidate[candidate != background])))
        points = np.argwhere(candidate == color)
        count = len(points) if len(points) < len(filled) else max(1, len(points) // 2)
        selected = points[rng.choice(len(points), count, replace=False)]
        candidate[selected[:, 0], selected[:, 1]] = background
    elif kind == "edge":
        edge = [(r, c) for r in range(height) for c in range(width) if r in (0, height - 1) or c in (0, width - 1)]
        rr, cc = edge[int(rng.integers(len(edge)))]
        candidate[rr, cc] = background if candidate[rr, cc] != background else int(rng.choice(np.unique(reference[reference != background])))
    else:
        raise ValueError("unknown edit")


def _candidate(rng, reference, background, divider, status):
    candidate = reference.copy()
    if status == "match":
        return candidate
    if status == "multi":
        for _ in range(int(rng.integers(2, 4))):
            if not np.any(candidate != background):
                _single_edit(rng, candidate, reference, background, divider)
            _edit_once(rng, candidate, reference, background, divider, str(rng.choice(("single", "grow", "shrink", "insert", "remove", "large"))))
    else:
        _edit_once(rng, candidate, reference, background, divider, status)
    if np.array_equal(candidate, reference):
        _single_edit(rng, candidate, reference, background, divider)
    return candidate


def _make_pair(rng, regime, status, complexity, roles, color_count, size_band=None):
    if size_band == "compact":
        height, width = int(rng.integers(5, 9)), int(rng.integers(5, 9))
    elif size_band == "spacious":
        height, width = int(rng.integers(9, 16)), int(rng.integers(9, 15))
    else:
        height = int(rng.integers(5, 13 if regime == "source_train" else 16))
        width = int(rng.integers(5, 15))
    background, divider = roles
    available = [x for x in range(10) if x not in roles]
    colors = tuple(int(x) for x in rng.choice(available, color_count, replace=False))
    motif = "rectangle" if complexity == "solid" else str(rng.choice(_COMPLEX if complexity == "complex" else _SIMPLE))
    reference = _make_reference(rng, height, width, colors, background, motif)
    candidate = _candidate(rng, reference, background, divider, status)
    input_grid = np.concatenate((reference, np.full((height, 1), divider, dtype=np.int64), candidate), axis=1)
    output_grid = input_grid.copy()
    if not np.array_equal(reference, candidate):
        output_grid[:, width + 1:] = background
    return {"input": input_grid, "output": output_grid}


def _sample_roles(rng, base, variable, index):
    if not variable or index == 0 or (index > 1 and rng.random() < .45):
        return base
    background = int(rng.choice([x for x in range(10) if x != base[0]]))
    divider = int(rng.choice([x for x in range(10) if x not in (background, base[1])]))
    return background, divider


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    base = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    variable_roles = bool(rng.random() < .4)

    train_statuses = ["match", str(rng.choice(_EDITS))] if num_train == 2 else ["match", "equal"]
    if num_train >= 4:
        train_statuses.extend(("large", "match"))
    while len(train_statuses) < num_train:
        train_statuses.append(str(rng.choice(("match",) + _EDITS)))
    rng.shuffle(train_statuses)
    train_complexity = ["complex" if rng.random() < .45 else "simple" for _ in train_statuses]
    if num_train >= 4:
        match_indices = [i for i, status in enumerate(train_statuses) if status == "match"]
        match_index = match_indices[0]
        reject_index = next(i for i, status in enumerate(train_statuses) if status != "match")
        train_complexity[match_index] = train_complexity[reject_index] = "complex"
        train_complexity[match_indices[1]] = "solid"
    train_colors = [1, int(rng.integers(2, 5))] + [int(rng.choice((1, 2, 3, 4), p=(.3, .32, .25, .13))) for _ in range(num_train - 2)]
    rng.shuffle(train_colors)
    if num_train >= 4:
        train_colors[match_indices[1]] = 1
        if all(count == 1 for count in train_colors):
            other = next(i for i in range(num_train) if i != match_indices[1])
            train_colors[other] = int(rng.integers(2, 5))
    train_bands = [None] * num_train
    if num_train >= 4:
        indices = rng.choice(num_train, 2, replace=False)
        train_bands[int(indices[0])] = "compact"
        train_bands[int(indices[1])] = "spacious"

    if num_test >= 2 and rng.random() < .6:
        test_statuses = ["match", str(rng.choice(_EDITS))]
    else:
        test_statuses = []
    while len(test_statuses) < num_test:
        test_statuses.append(str(rng.choice(("match",) + _EDITS)))
    rng.shuffle(test_statuses)

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, statuses in (("train", train_statuses), ("test", test_statuses)):
        for index, status in enumerate(statuses):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            roles = _sample_roles(rng, base, variable_roles, index)
            complexity = train_complexity[index] if split == "train" else str(rng.choice(("solid", "simple", "complex"), p=(.12, .28, .6)))
            count = train_colors[index] if split == "train" else (1 if complexity == "solid" else int(rng.choice((1, 2, 3, 4), p=(.22, .3, .3, .18))))
            band = train_bands[index] if split == "train" else None
            for _ in range(40):
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test", status, complexity, roles, count, band)
                grid = pair["input"]
                signature = (grid.shape, grid.tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_hash(input_grid, output_grid):
    return hashlib.sha256(str(input_grid.shape).encode() + input_grid.astype(np.int64).tobytes() + output_grid.astype(np.int64).tobytes()).hexdigest()


def _pair_features(pair):
    grid = pair["input"]
    width = grid.shape[1] // 2
    reference, candidate = grid[:, :width], grid[:, width + 1:]
    background = int(np.bincount(reference.ravel(), minlength=10).argmax())
    divider = int(grid[0, width])
    return reference, candidate, background, divider


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    train_roles, test_roles = [], []
    train_outcomes = set()
    train_reference_color_counts = set()
    equal_histogram_rejection = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert np.issubdtype(array.dtype, np.integer)
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.all((array >= 0) & (array <= 9))
            assert grid.shape == output.shape
            height, total_width = grid.shape
            assert 5 <= height <= 15 and 11 <= total_width <= 29 and total_width % 2 == 1
            reference, candidate, background, divider = _pair_features(pair)
            width = reference.shape[1]
            assert background != divider
            assert np.all(grid[:, width] == divider)
            assert not np.any(reference == divider) and not np.any(candidate == divider)
            assert np.count_nonzero(reference == background) > reference.size // 2
            assert 1 <= len(set(int(x) for x in np.unique(reference)) - {background}) <= 4
            assert np.array_equal(output[:, :width + 1], grid[:, :width + 1])
            match = np.array_equal(reference, candidate)
            if match:
                assert np.array_equal(output, grid)
            else:
                assert np.all(output[:, width + 1:] == background)
            input_signature = (grid.shape, grid.tobytes())
            pair_signature = (grid.shape, grid.tobytes(), output.tobytes())
            assert input_signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            assert _pair_hash(grid, output) not in _SOURCE_PAIR_HASHES
            (train_roles if split == "train" else test_roles).append((background, divider))
            if split == "train":
                train_outcomes.add(match)
                train_reference_color_counts.add(len(set(int(x) for x in np.unique(reference)) - {background}))
                equal_histogram_rejection |= not match and np.array_equal(np.bincount(reference.ravel(), minlength=10), np.bincount(candidate.ravel(), minlength=10))
    assert train_outcomes == {False, True}
    assert 1 in train_reference_color_counts and any(n >= 2 for n in train_reference_color_counts)
    if train_count >= 3:
        assert equal_histogram_rejection
    if any(roles != train_roles[0] for roles in test_roles):
        assert len(set(train_roles)) >= 2


def _same_puzzle(first, second):
    return all(len(first[k]) == len(second[k]) and all(np.array_equal(a[g], b[g]) for a, b in zip(first[k], second[k]) for g in ("input", "output")) for k in ("train", "test"))


def _is_rectangle(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    low, high = points.min(axis=0), points.max(axis=0)
    return bool(np.all(mask[low[0]:high[0] + 1, low[1]:high[1] + 1]))


def _component_count(mask):
    pending = set(map(tuple, np.argwhere(mask)))
    count = 0
    while pending:
        stack = [pending.pop()]
        count += 1
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
    return count


def _frame_types(reference, background):
    hollow = nested = False
    for color in set(int(x) for x in np.unique(reference)) - {background}:
        points = np.argwhere(reference == color)
        low, high = points.min(axis=0), points.max(axis=0)
        r0, c0 = low
        r1, c1 = high
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = np.concatenate((reference[r0, c0:c1 + 1], reference[r1, c0:c1 + 1], reference[r0:r1 + 1, c0], reference[r0:r1 + 1, c1]))
        if np.all(border == color):
            inside = reference[r0 + 1:r1, c0 + 1:c1]
            hollow |= bool(np.any(inside == background))
            nested |= bool(np.any((inside != background) & (inside != color)))
    return hollow, nested


def self_test() -> None:
    counts, dimensions, layouts, palettes, roles = set(), set(), set(), set(), set()
    one_color = equal_hist_train = complex_train_match = complex_train_reject = 0
    single_diffs = large_diffs = fresh_colors = mixed_tests = variable_roles = 0
    solid_train_match = solid_test = hollow_train = nested_train = touching_train = disconnected_train = 0
    background_edits = border_edits = compact_train = spacious_train = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        local_roles = set()
        outcomes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                ref, cand, bg, div = _pair_features(pair)
                width = ref.shape[1]
                match = np.array_equal(ref, cand)
                differences = int(np.count_nonzero(ref != cand))
                assert np.array_equal(pair["output"][:, :width + 1], grid[:, :width + 1])
                assert np.array_equal(pair["output"][:, width + 1:], cand if match else np.full_like(cand, bg))
                dimensions.add(grid.shape)
                layouts.add((ref.shape, (ref != bg).tobytes()))
                palettes.add(tuple(sorted(set(int(x) for x in np.unique(ref)) - {bg})))
                roles.add((bg, div))
                local_roles.add((bg, div))
                if split == "train":
                    one_color += len(np.unique(ref)) == 2
                    equal_hist_train += not match and np.array_equal(np.bincount(ref.ravel(), minlength=10), np.bincount(cand.ravel(), minlength=10))
                    complex_shape = any(not _is_rectangle(ref == color) for color in np.unique(ref) if color != bg)
                    complex_train_match += match and complex_shape
                    complex_train_reject += not match and complex_shape
                    solid_train_match += match and len(np.unique(ref)) == 2 and _is_rectangle(ref != bg)
                    hollow, nested = _frame_types(ref, bg)
                    hollow_train += hollow
                    nested_train += nested
                    touching_train += bool(np.any((ref[:, :-1] != ref[:, 1:]) & (ref[:, :-1] != bg) & (ref[:, 1:] != bg)) or np.any((ref[:-1] != ref[1:]) & (ref[:-1] != bg) & (ref[1:] != bg)))
                    disconnected_train += any(_component_count(ref == color) >= 2 for color in np.unique(ref) if color != bg)
                    single_diffs += differences == 1
                    large_diffs += differences >= 8
                    fresh_colors += bool((set(np.unique(cand)) - set(np.unique(ref))) - {bg})
                    background_edits += not match and bool(np.any((ref != cand) & (ref == bg)))
                    border_edits += not match and bool(np.any(ref[0] != cand[0]) or np.any(ref[-1] != cand[-1]) or np.any(ref[:, 0] != cand[:, 0]) or np.any(ref[:, -1] != cand[:, -1]))
                    compact_train += ref.shape[0] <= 7 and ref.shape[1] <= 7
                    spacious_train += ref.shape[0] >= 9 and ref.shape[1] >= 9
                else:
                    outcomes.append(match)
                    solid_test += len(np.unique(ref)) == 2 and _is_rectangle(ref != bg)
        variable_roles += len(local_roles) >= 2
        mixed_tests += len(set(outcomes)) == 2
    assert len({x[0] for x in counts}) >= 4 and len({x[1] for x in counts}) >= 3
    assert len(dimensions) >= 40 and len(layouts) >= 170 and len(palettes) >= 20 and len(roles) >= 15
    assert one_color >= 20 and equal_hist_train >= 15 and complex_train_match >= 15 and complex_train_reject >= 15
    assert single_diffs >= 8 and large_diffs >= 8 and fresh_colors >= 8 and mixed_tests >= 4 and variable_roles >= 8
    assert solid_train_match >= 8 and solid_test >= 3 and hollow_train >= 3 and nested_train >= 1
    assert touching_train >= 10 and disconnected_train >= 10 and background_edits >= 8 and border_edits >= 8
    assert compact_train >= 8 and spacious_train >= 8
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1234 + (ntrain or 0) + (ntest or 0), ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        validate_generated_puzzle(puzzle)
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (10, 10), (True, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
