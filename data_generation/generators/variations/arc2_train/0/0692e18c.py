import numpy as np


_SOURCE_INPUTS = (
    ((0, 0, 6), (0, 6, 0), (6, 0, 0)),
    ((0, 7, 0), (7, 7, 7), (0, 7, 0)),
    ((4, 4, 0), (0, 0, 4), (0, 0, 4)),
    ((0, 0, 3), (3, 3, 0), (0, 3, 0)),
)
_MOTIFS = ("random", "symmetric", "single", "two", "line", "bend",
           "filled", "hole", "nearfull", "disconnected", "permutation",
           "three_plus_single")
_OPS = ("identity", "left_right", "top_bottom", "half_turn",
        "transpose", "quarter_turn", "three_quarter_turn", "anti_diagonal")


def _key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _transform(array, operation):
    if operation == "identity":
        return array
    if operation == "left_right":
        return np.fliplr(array)
    if operation == "top_bottom":
        return np.flipud(array)
    if operation == "half_turn":
        return np.rot90(array, 2)
    if operation == "transpose":
        return array.T
    if operation == "quarter_turn":
        return np.rot90(array)
    if operation == "three_quarter_turn":
        return np.rot90(array, 3)
    return np.rot90(array, 2).T


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        sizes.append(size)
    return tuple(sorted(sizes))


def _symmetries(mask):
    return tuple(op for op in _OPS[1:]
                 if _transform(mask, op).shape == mask.shape
                 and np.array_equal(mask, _transform(mask, op)))


def _render(mask, background, foreground):
    return np.where(mask, foreground, background).astype(np.int8)


def _answer(mask, background, foreground):
    pixels = np.kron(mask.astype(np.int8), (~mask).astype(np.int8)) != 0
    return _render(pixels, background, foreground)


def _sample_symmetric(rng, shape, axis, density):
    h, w = shape
    result = np.zeros(shape, dtype=bool)
    visited = np.zeros(shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            marker = np.zeros(shape, dtype=bool)
            marker[r, c] = True
            rr, cc = np.argwhere(_transform(marker, axis))[0]
            rr, cc = int(rr), int(cc)
            value = rng.random() < density
            result[r, c] = result[rr, cc] = value
            visited[r, c] = visited[rr, cc] = True
    return result


def _sample_mask(rng, shape, motif):
    h, w = shape
    area = h * w
    mask = np.zeros(shape, dtype=bool)
    if motif == "single":
        mask.flat[int(rng.integers(area))] = True
    elif motif == "two":
        mask.flat[rng.choice(area, size=2, replace=False)] = True
    elif motif == "nearfull":
        mask[:] = True
        mask.flat[int(rng.integers(area))] = False
    elif motif == "line":
        if rng.integers(2):
            length = int(rng.integers(2, w + 1))
            r = int(rng.integers(h))
            c = int(rng.integers(w-length+1))
            mask[r, c:c+length] = True
        else:
            length = int(rng.integers(2, h + 1))
            r = int(rng.integers(h-length+1))
            c = int(rng.integers(w))
            mask[r:r+length, c] = True
    elif motif == "bend":
        r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
        mask[r:r+2, c:c+2] = True
        corners = ((r, c), (r, c+1), (r+1, c), (r+1, c+1))
        mask[corners[int(rng.integers(4))]] = False
    elif motif == "filled":
        rh, rw = int(rng.integers(2, h+1)), int(rng.integers(2, w+1))
        r, c = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        mask[r:r+rh, c:c+rw] = True
        if mask.all():
            mask.flat[int(rng.integers(area))] = False
    elif motif == "hole" and h >= 3 and w >= 3:
        rh, rw = int(rng.integers(3, h+1)), int(rng.integers(3, w+1))
        r, c = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        mask[r:r+rh, c:c+rw] = True
        mask[r+int(rng.integers(1, rh-1)),
             c+int(rng.integers(1, rw-1))] = False
    elif motif == "permutation":
        if h <= w:
            mask[np.arange(h), rng.permutation(w)[:h]] = True
        else:
            mask[rng.permutation(h)[:w], np.arange(w)] = True
    elif motif == "cross" and h == w == 3:
        mask[1, :] = True
        mask[:, 1] = True
    elif motif == "four_asymmetric" and h == w == 3:
        for _ in range(128):
            mask[:] = False
            mask.flat[rng.choice(area, size=4, replace=False)] = True
            if not _symmetries(mask):
                break
        else:
            mask[:] = ((1, 1, 0), (0, 1, 0), (0, 1, 1))
    elif motif == "symmetric":
        axes = ["left_right", "top_bottom", "half_turn"]
        if h == w:
            axes += ["transpose", "anti_diagonal"]
        mask = _sample_symmetric(rng, shape, axes[int(rng.integers(len(axes)))],
                                 float(rng.uniform(0.18, 0.83)))
    elif motif == "disconnected":
        for _ in range(64):
            mask = rng.random(shape) < float(rng.uniform(0.16, 0.43))
            if len(_components(mask)) >= 2:
                break
        else:
            mask[:] = False
            mask[0, 0] = mask[-1, -1] = True
    elif motif == "three_plus_single" and area >= 6:
        for _ in range(64):
            mask = np.zeros(shape, dtype=bool)
            r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
            mask[r:r+2, c:c+2] = True
            mask[r+int(rng.integers(2)), c+int(rng.integers(2))] = False
            choices = [i for i in range(area) if not mask.flat[i]]
            mask.flat[choices[int(rng.integers(len(choices)))]] = True
            if _components(mask) == (1, 3):
                break
        else:
            mask = rng.random(shape) < 0.35
    else:
        mask = rng.random(shape) < float(rng.uniform(0.14, 0.86))
    if not mask.any():
        mask.flat[int(rng.integers(area))] = True
    if mask.all():
        mask.flat[int(rng.integers(area))] = False
    operations = ["identity", "left_right", "top_bottom", "half_turn"]
    if h == w:
        operations += ["transpose", "quarter_turn", "three_quarter_turn",
                       "anti_diagonal"]
    return _transform(mask, operations[int(rng.integers(len(operations)))]).copy()


def _orientation_exposed(train, test, background):
    for operation in _OPS[1:]:
        fits = True
        for pair in train:
            inp = pair["input"]
            mask = inp != background
            inner = _transform(~mask, operation)
            if inner.shape != mask.shape:
                fits = False
                break
            fg = int(inp[mask][0])
            candidate = _render(np.kron(mask.astype(np.int8),
                                        inner.astype(np.int8)) != 0,
                                background, fg)
            if not np.array_equal(candidate, pair["output"]):
                fits = False
                break
        if fits:
            for pair in test:
                inp = pair["input"]
                mask = inp != background
                inner = _transform(~mask, operation)
                if inner.shape != mask.shape:
                    continue
                fg = int(inp[mask][0])
                candidate = _render(np.kron(mask.astype(np.int8),
                                            inner.astype(np.int8)) != 0,
                                    background, fg)
                if not np.array_equal(candidate, pair["output"]):
                    return False
    return True


def _resolve_counts(rng, seed, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test+1)
        num_train = low + (seed // 4) % (11-low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test)+1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shapes(rng, train_count, test_count, mode):
    all_shapes = [(h, w) for h in range(2, 6) for w in range(2, 6)]
    def draw():
        return all_shapes[int(rng.integers(len(all_shapes)))]
    if mode == "source_like":
        return [(3, 3)] * train_count, [(3, 3)] * test_count
    if mode == "uniform":
        shape = draw()
        if shape == (2, 2):
            shape = (3, 3)
        return [shape] * train_count, [shape] * test_count
    if mode == "larger":
        train = [(3, 3)] + [tuple(min(x, 4) for x in draw())
                            for _ in range(train_count-1)]
        test = [(5, int(rng.integers(2, 6)))]
    elif mode == "smaller":
        train = [(4, 4)] + [(int(rng.integers(3, 6)),
                             int(rng.integers(3, 6))) for _ in range(train_count-1)]
        test = [(2, int(rng.integers(2, 6)))]
    elif mode == "transpose":
        rectangle = draw()
        if rectangle[0] == rectangle[1]:
            rectangle = (3, 5)
        train = [(3, 3), rectangle] + [draw() for _ in range(train_count-2)]
        test = [(rectangle[1], rectangle[0])]
    else:
        side = int(rng.integers(3, 6))
        train = [(side, side)] + [draw() for _ in range(train_count-1)]
        test = []
    test += [draw() for _ in range(test_count-len(test))]
    if len(set(train + test)) == 1:
        test[0] = (2, 5) if train[0] != (2, 5) else (5, 2)
    return train, test


def _choose_mask(rng, shape, motif, used):
    for _ in range(128):
        mask = _sample_mask(rng, shape, motif)
        if _key(mask) not in used:
            used.add(_key(mask))
            return mask
    area = shape[0] * shape[1]
    for count in (1, 2, area-1, 3, area-2):
        if not 1 <= count < area:
            continue
        for offset in range(area):
            mask = np.zeros(shape, dtype=bool)
            mask.flat[[(offset+i) % area for i in range(count)]] = True
            if _key(mask) not in used:
                used.add(_key(mask))
                return mask
    raise RuntimeError("No unused mask could be constructed")


def _pair(mask, background, foreground):
    return {"input": _render(mask, background, foreground),
            "output": _answer(mask, background, foreground)}


def _cell_rule_holds(inp, out, background, foreground):
    h, w = inp.shape
    for r in range(h*h):
        for c in range(w*w):
            parent_is_foreground = inp[r//h, c//w] == foreground
            inner_is_background = inp[r % h, c % w] == background
            expected = foreground if parent_is_foreground and inner_is_background else background
            if out[r, c] != expected:
                return False
    return True


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    background = int(rng.integers(10))
    foregrounds = [int(x) for x in rng.permutation(
        [x for x in range(10) if x != background])]
    mode = ("source_like", "uniform", "mixed", "mixed", "larger", "smaller",
            "transpose")[int(rng.integers(7))]
    train_shapes, test_shapes = _shapes(rng, train_count, test_count, mode)
    train_colors = [foregrounds[0], foregrounds[1]] + [
        foregrounds[int(rng.integers(5))] for _ in range(train_count-2)]
    rng.shuffle(train_colors)
    test_colors = []
    for _ in range(test_count):
        if rng.random() < 0.55:
            test_colors.append(int(train_colors[int(rng.integers(train_count))]))
        else:
            unseen = [c for c in foregrounds if c not in train_colors]
            test_colors.append(int(unseen[int(rng.integers(len(unseen)))]
                                   if unseen else train_colors[int(rng.integers(train_count))]))
    used = set()
    train_masks = []
    joint = mode == "mixed" and rng.random() < 0.22
    if joint:
        train_shapes[:2] = [(4, 4), (4, 4)]
    for index, shape in enumerate(train_shapes):
        if joint and index < 2:
            axis = "left_right" if index == 0 else "top_bottom"
            for _ in range(128):
                mask = _sample_symmetric(rng, shape, axis,
                                         float(rng.uniform(0.25, 0.75)))
                if _symmetries(mask) == (axis,) and _key(mask) not in used:
                    used.add(_key(mask))
                    break
            else:
                mask = _choose_mask(rng, shape, "random", used)
        else:
            if mode == "source_like":
                if train_count == 2:
                    motif = ("four_asymmetric", "cross")[index]
                elif index < 3:
                    motif = ("permutation", "cross", "four_asymmetric")[index]
                else:
                    motif = _MOTIFS[int(rng.integers(len(_MOTIFS)))]
            elif index == 0 and mode != "uniform":
                motif = "random"
            else:
                choices = _MOTIFS if min(shape) >= 3 else tuple(
                    m for m in _MOTIFS if m != "hole")
                motif = choices[int(rng.integers(len(choices)))]
            mask = _choose_mask(rng, shape, motif, used)
        train_masks.append(mask)
    if mode == "source_like":
        test_motifs = ["three_plus_single"] + ["random"] * (test_count-1)
    else:
        test_motifs = []
        for shape in test_shapes:
            choices = _MOTIFS if min(shape) >= 3 else tuple(
                m for m in _MOTIFS if m != "hole")
            test_motifs.append(choices[int(rng.integers(len(choices)))])
    test_masks = [_choose_mask(rng, shape, motif, used)
                  for shape, motif in zip(test_shapes, test_motifs)]
    puzzle = {"train": [_pair(mask, background, color)
                        for mask, color in zip(train_masks, train_colors)],
              "test": [_pair(mask, background, color)
                       for mask, color in zip(test_masks, test_colors)]}
    if not _orientation_exposed(puzzle["train"], puzzle["test"], background):
        shape = train_shapes[0]
        for _ in range(256):
            mask = _sample_mask(rng, shape, "random")
            if not _symmetries(mask) and _key(mask) not in used:
                puzzle["train"][0] = _pair(mask, background, train_colors[0])
                break
        else:
            raise RuntimeError("Could not establish tile orientation")
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            if tuple(tuple(int(v) for v in row) for row in pair["input"]) in _SOURCE_INPUTS:
                shape = pair["input"].shape
                mask = _choose_mask(rng, shape, "random", used)
                color = int(pair["input"][pair["input"] != background][0])
                puzzle[split][index] = _pair(mask, background, color)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_color_sets = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert 2 <= inp.shape[0] <= 5 and 2 <= inp.shape[1] <= 5
            assert out.shape == (inp.shape[0]**2, inp.shape[1]**2)
            colors = set(int(x) for x in np.unique(inp))
            assert len(colors) == 2
            if split == "train":
                train_color_sets.append(colors)
            ik, pk = _key(inp), (_key(inp), _key(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            assert tuple(tuple(int(v) for v in row) for row in inp) not in _SOURCE_INPUTS
            seen_inputs.add(ik)
            seen_pairs.add(pk)
    backgrounds = set.intersection(*train_color_sets)
    assert len(backgrounds) == 1, "Training must identify one shared background"
    background = next(iter(backgrounds))
    train_foregrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            colors = set(int(x) for x in np.unique(inp))
            assert background in colors
            foreground = next(iter(colors - {background}))
            if split == "train":
                train_foregrounds.add(foreground)
            mask = inp != background
            assert 1 <= int(mask.sum()) < mask.size
            assert _cell_rule_holds(inp, out, background, foreground)
            assert set(int(x) for x in np.unique(out)) == colors
    assert len(train_foregrounds) >= 2
    assert _orientation_exposed(puzzle["train"], puzzle["test"], background)


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) >= 40
    assert _signature(generate_puzzle(216)) == _signature(generate_puzzle(216))
    for seed, counts in ((216, (None, None)), (252, (None, None)),
                         (27, (2, 1))):
        validate_generated_puzzle(generate_puzzle(seed, *counts))
    joint = generate_puzzle(17, 2, 1)
    joint_background = next(iter(set.intersection(*(
        set(int(x) for x in np.unique(p["input"])) for p in joint["train"]))))
    assert all(_symmetries(p["input"] != joint_background) for p in joint["train"])
    assert _orientation_exposed(joint["train"], joint["test"], joint_background)
    for a, b in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1),
                 (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1200 + (a or 0)*10 + (b or 0), a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4), (2.0, 1)):
        try:
            generate_puzzle(17, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid count was accepted")
    all_pairs = [pair for puzzle in puzzles for split in ("train", "test")
                 for pair in puzzle[split]]
    backgrounds, shapes, foreground_counts = set(), set(), set()
    component_profiles, symmetries = set(), set()
    familiar = unseen = mixed = uniform = sparse = nearfull = 0
    heldout = test_holes = train_holes = test_symmetric = 0
    source_mask_recolors = 0
    split_axes = {"train": set(), "test": set()}
    density_combinations = {"train": set(), "test": set()}
    for puzzle in puzzles:
        bg = next(iter(set.intersection(*(set(int(x) for x in np.unique(p["input"]))
                                             for p in puzzle["train"]))))
        backgrounds.add(bg)
        puzzle_shapes = {p["input"].shape for s in ("train", "test") for p in puzzle[s]}
        mixed += len(puzzle_shapes) > 1
        uniform += len(puzzle_shapes) == 1
        heldout += bool({p["input"].shape for p in puzzle["test"]} -
                        {p["input"].shape for p in puzzle["train"]})
        train_colors = {next(iter(set(int(x) for x in np.unique(p["input"])) - {bg}))
                        for p in puzzle["train"]}
        for pair in puzzle["test"]:
            inp = pair["input"]
            fg = next(iter(set(int(x) for x in np.unique(inp)) - {bg}))
            familiar += fg in train_colors
            unseen += fg not in train_colors
            sparse += int((inp != bg).sum()) <= 2
            nearfull += int((inp != bg).sum()) == inp.size - 1
        for s in ("train", "test"):
            for pair in puzzle[s]:
                inp = pair["input"]
                mask = inp != bg
                shapes.add(inp.shape)
                foreground_counts.add(int(mask.sum()))
                component_profiles.add(_components(mask))
                symmetries.add(_symmetries(mask))
                split_axes[s].update(_symmetries(mask))
                density = float(mask.mean())
                density_band = 0 if density <= 0.25 else 1 if density < 0.75 else 2
                density_combinations[s].add((density_band, bool(_symmetries(mask)),
                                             len(_components(mask)) == 1))
                hole = any(not mask[r, c] and mask[r-1, c] and mask[r+1, c]
                           and mask[r, c-1] and mask[r, c+1]
                           for r in range(1, mask.shape[0]-1)
                           for c in range(1, mask.shape[1]-1))
                train_holes += s == "train" and hole
                test_holes += s == "test" and hole
                test_symmetric += s == "test" and bool(_symmetries(mask))
                source_mask_recolors += any(
                    mask.shape == (len(source), len(source[0]))
                    and np.array_equal(mask, np.asarray(source) != 0)
                    for source in _SOURCE_INPUTS)
    assert len(all_pairs) >= 200
    assert len(backgrounds) >= 7 and 0 in backgrounds
    assert (2, 2) in shapes and len(shapes) >= 12
    assert len(foreground_counts) >= 15 and len(component_profiles) >= 20
    assert len(symmetries) >= 5
    assert mixed >= 15 and uniform >= 3
    assert heldout >= 8
    assert familiar >= 15 and unseen >= 10
    assert sparse >= 3 and nearfull >= 2
    assert train_holes >= 3 and test_holes >= 2 and test_symmetric >= 5
    assert {"left_right", "top_bottom", "half_turn", "transpose",
            "anti_diagonal"} <= split_axes["train"]
    assert {"left_right", "top_bottom", "transpose", "anti_diagonal"} <= split_axes["test"]
    assert len(density_combinations["train"]) >= 9
    assert len(density_combinations["test"]) >= 8
    assert source_mask_recolors >= 1
