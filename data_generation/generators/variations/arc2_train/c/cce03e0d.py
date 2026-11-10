import numpy as np

SOURCE_INPUTS = {
    ((3, 3), (0, 1, 2, 1, 1, 0, 2, 0, 0)),
    ((3, 3), (1, 0, 0, 2, 1, 0, 0, 0, 1)),
    ((3, 3), (2, 1, 2, 0, 2, 1, 2, 1, 0)),
    ((3, 3), (1, 2, 2, 2, 0, 1, 1, 2, 0)),
}
SHAPES = ((2, 3), (3, 2), (3, 3), (3, 4), (4, 3),
          (4, 4), (5, 5), (2, 4), (4, 2), (3, 5),
          (5, 3), (4, 5), (5, 4))
AUTO_COUNTS = tuple((a, b) for a in range(2, 11)
                    for b in range(1, min(4, a - 1) + 1))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def counts(seed, token, num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        return AUTO_COUNTS[seed % len(AUTO_COUNTS)]
    if num_train is None:
        lo = max(2, int(num_test) + 1)
        num_train = lo + token % (11 - lo)
    if num_test is None:
        num_test = 1 + token % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def expand(grid, background, selector):
    h, w = grid.shape
    out = np.full((h*h, w*w), background, dtype=np.int8)
    for r, c in np.argwhere(grid == selector):
        out[r*h:(r+1)*h, c*w:(c+1)*w] = grid
    return out


def adjacent(mask):
    return bool(np.any(mask[:, :-1] & mask[:, 1:]) or
                np.any(mask[:-1, :] & mask[1:, :]))


def mask_key(mask):
    return (tuple(mask.shape), tuple(int(x) for x in mask.flat))


def random_mask(rng, h, w, count, style):
    mask = np.zeros((h, w), dtype=bool)
    if style == "isolated":
        positions = [(r, c) for r in range(h) for c in range(w)]
        for i in rng.permutation(len(positions)):
            r, c = positions[int(i)]
            if not mask[max(0, r-1):min(h, r+2), c].any() and not mask[r, max(0, c-1):min(w, c+2)].any():
                mask[r, c] = True
                if int(mask.sum()) == count:
                    return mask
        return None
    if style == "connected":
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, c] = True
        while int(mask.sum()) < count:
            boundary = []
            for rr, cc in np.argwhere(mask):
                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        boundary.append((nr, nc))
            r, c = boundary[int(rng.integers(len(boundary)))]
            mask[r, c] = True
        return mask
    if style == "line" and count <= max(h, w):
        axes = [axis for axis, length in ((0, w), (1, h)) if count <= length]
        axis = axes[int(rng.integers(len(axes)))]
        if axis == 0:
            r, start = int(rng.integers(h)), int(rng.integers(w-count+1))
            mask[r, start:start+count] = True
        else:
            c, start = int(rng.integers(w)), int(rng.integers(h-count+1))
            mask[start:start+count, c] = True
        return mask
    if style == "l" and count >= 3:
        r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
        mask[r, c] = mask[r+1, c] = mask[r, c+1] = True
    if style == "symmetric":
        axis = int(rng.integers(2))
        orbits, seen = [], set()
        for r in range(h):
            for c in range(w):
                mate = (r, w-1-c) if axis == 0 else (h-1-r, c)
                orbit = frozenset(((r, c), mate))
                if orbit not in seen:
                    seen.add(orbit)
                    orbits.append(tuple(orbit))
        for i in rng.permutation(len(orbits)):
            orbit = orbits[int(i)]
            if len(orbit) <= count - int(mask.sum()):
                for r, c in orbit:
                    mask[r, c] = True
            if int(mask.sum()) == count:
                return mask
        return None
    remaining = np.flatnonzero(~mask)
    need = count - int(mask.sum())
    if need < 0:
        return None
    if need:
        mask.flat[rng.choice(remaining, size=need, replace=False)] = True
    return mask


def sample_mask(rng, shape, count, touching, used):
    h, w = shape
    styles = ("connected", "line", "l", "symmetric", "uniform") if touching else ("isolated",)
    preferred = styles[int(rng.integers(len(styles)))]
    duplicate = None
    for attempt in range(320):
        style = preferred if attempt < 60 else styles[int(rng.integers(len(styles)))]
        mask = random_mask(rng, h, w, count, style)
        if mask is None or adjacent(mask) != touching:
            continue
        if mask_key(mask) not in used:
            used.add(mask_key(mask))
            return mask
        duplicate = mask
    if duplicate is not None:
        return duplicate
    raise AssertionError("could not construct selector geometry")


def sample_input(rng, mask, roles, split, index, novel_test, used):
    background, selector, base, extra, novel = roles
    inactive = np.argwhere(~mask)
    n = len(inactive)
    for _ in range(120):
        grid = np.full(mask.shape, background, dtype=np.int8)
        grid[mask] = selector
        if split == "train" and index == 0:
            content_count = min(2, n-1)
            palette = [base]
        else:
            density = int(rng.integers(3))
            content_count = (1 if density == 0 else n-1 if density == 1 else int(rng.integers(1, n)))
            palette = [base, extra] + ([novel] if split == "test" and novel_test else [])
        if split == "test" and novel_test and index == 0 and n >= 3:
            content_count = max(2, content_count)
            colors = [base, novel]
        else:
            max_colors = min(len(palette), content_count)
            color_count = 1 if split == "train" and index == 0 else int(rng.integers(1, max_colors+1))
            colors = list(rng.choice(palette, size=color_count, replace=False))
        chosen = rng.choice(n, size=content_count, replace=False)
        for i, pos in enumerate(chosen):
            r, c = inactive[int(pos)]
            grid[r, c] = int(colors[i % len(colors)])
        if key(grid) not in used and key(grid) not in SOURCE_INPUTS:
            used.add(key(grid))
            return grid
    raise AssertionError("could not construct a unique non-source input")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    token = int(rng.integers(0, np.iinfo(np.int64).max))
    ntrain, ntest = counts(seed, token, num_train, num_test)
    roles = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    shape_mode, geometry_mode = seed % 4, (seed // 4) % 3
    novel_test = bool(rng.integers(2))
    shared_shape = SHAPES[int(rng.integers(len(SHAPES)))]
    used_inputs, used_masks, train_masks_seen, test_masks_seen = set(), set(), set(), set()
    puzzle = {"train": [], "test": []}
    for split, amount in (("train", ntrain), ("test", ntest)):
        for index in range(amount):
            if shape_mode == 0:
                shape = (3, 3)
            elif shape_mode == 1:
                shape = shared_shape
            elif split == "train" and index == 0:
                shape = (3, 3)
            elif split == "train" and index == 1:
                shape = SHAPES[int(rng.integers(len(SHAPES)))]
                if shape == (3, 3):
                    shape = (3, 4)
            else:
                shape = SHAPES[int(rng.integers(len(SHAPES)))]
            h, w = shape
            touching = ((split == "train" and index > 0 and geometry_mode == 0) or
                        (split == "test" and geometry_mode != 2))
            for _ in range(80):
                if split == "train" and index == 0:
                    count = 1
                else:
                    ceiling = h*w - 2
                    if not touching:
                        ceiling = min(ceiling, (h*w + 1)//2)
                    lower = 2 if touching or split == "train" else 1
                    count = int(rng.integers(lower, ceiling+1))
                mask = sample_mask(rng, shape, count, touching, used_masks)
                if split != "test" or (mask_key(mask) not in test_masks_seen
                                       and mask_key(mask) not in train_masks_seen):
                    break
            else:
                raise AssertionError("could not choose distinct test masks")
            if split == "test":
                test_masks_seen.add(mask_key(mask))
            else:
                train_masks_seen.add(mask_key(mask))
            grid = sample_input(rng, mask, roles, split, index, novel_test, used_inputs)
            puzzle[split].append({"input": grid, "output": expand(grid, roles[0], roles[1])})
    validate_generated_puzzle(puzzle)
    return puzzle


def analyze_pair(pair):
    grid, out = pair["input"], pair["output"]
    h, w = grid.shape
    check(2 <= h <= 5 and 2 <= w <= 5, "input shape outside supported expansion")
    check(out.shape == (h*h, w*w), "incorrect squared output dimensions")
    active, blank = np.zeros((h, w), dtype=bool), set()
    for r in range(h):
        for c in range(w):
            block = out[r*h:(r+1)*h, c*w:(c+1)*w]
            if np.array_equal(block, grid):
                active[r, c] = True
            else:
                check(bool(np.all(block == block[0, 0])), "inactive block is not uniform")
                blank.add(int(block[0, 0]))
    check(bool(active.any()) and bool((~active).any()), "need active and blank blocks")
    check(len(blank) == 1, "blank blocks disagree on background")
    background = next(iter(blank))
    selected = {int(x) for x in grid[active]}
    check(len(selected) == 1, "multiple selector colors")
    selector = next(iter(selected))
    check(selector != background and bool(np.any(grid == background)), "background or selector absent")
    check(np.array_equal(active, grid == selector), "selector mask does not match copies")
    check(len(set(int(x) for x in np.unique(grid)) - {background, selector}) >= 1,
          "ordinary content absent")
    check(np.array_equal(out, expand(grid, background, selector)), "wrong expansion")
    return background, selector, active


def learn_roles(train):
    palette = set(int(x) for pair in train for x in np.unique(pair["input"]))
    possible = []
    for background in palette:
        for selector in palette - {background}:
            if all(np.array_equal(pair["output"], expand(pair["input"], background, selector))
                   for pair in train):
                possible.append((background, selector))
    return possible


def validate_generated_puzzle(puzzle: dict) -> None:
    check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "bad pair counts")
    inputs, pairs, roles, train_masks, test_masks, train_counts = set(), set(), set(), set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            for grid in pair.values():
                check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not 2D NumPy")
                check(np.issubdtype(grid.dtype, np.integer), "grid is not integer")
                check(all(1 <= side <= 30 for side in grid.shape), "grid axis outside ARC bounds")
                check(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid ARC color")
            ikey = key(pair["input"])
            pkey = (ikey, key(pair["output"]))
            check(ikey not in inputs and pkey not in pairs, "duplicate input or pair")
            check(ikey not in SOURCE_INPUTS, "source pair copied")
            inputs.add(ikey)
            pairs.add(pkey)
            background, selector, mask = analyze_pair(pair)
            roles.add((background, selector))
            if split == "train":
                train_masks.add(mask_key(mask))
                train_counts.add(int(mask.sum()))
            else:
                test_masks.add(mask_key(mask))
    check(len(roles) == 1, "puzzle color roles drift")
    check(len(train_counts) >= 2 and len(train_masks) >= 2, "training masks and counts must vary")
    check(len(test_masks) == len(test), "test masks repeat")
    check(not train_masks.intersection(test_masks), "test reuses a training mask")
    learned = learn_roles(train)
    check(len(learned) == 1 and learned[0] == next(iter(roles)), "training does not uniquely reveal roles")
    background, selector = learned[0]
    for pair in test:
        check(np.array_equal(pair["output"], expand(pair["input"], background, selector)),
              "test output cannot be inferred from training roles")


def puzzle_key(puzzle):
    return tuple(tuple((key(p["input"]), key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    for args in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1),
                 (None, 4), (2, None), (10, None), (None, None)):
        puzzle = generate_puzzle(12345, *args)
        validate_generated_puzzle(puzzle)
        check(puzzle_key(puzzle) == puzzle_key(generate_puzzle(12345, *args)), "nondeterminism")
        for requested, split in zip(args, ("train", "test")):
            check(requested is None or len(puzzle[split]) == requested, "requested count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    train_counts, test_counts, shapes, selector_counts = set(), set(), set(), set()
    masks, grids, palettes, puzzles = set(), set(), set(), set()
    features = {name: 0 for name in ("horizontal", "vertical", "isolated", "line",
                                     "l_shape", "symmetric", "multicolor", "novel",
                                     "selector_majority", "sparse_content",
                                     "nearly_filled", "rect_multi_touch")}
    regimes = set()
    mixed_train_shapes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        b, s = learn_roles(puzzle["train"])[0]
        palettes.add((b, s))
        train_colors = set(int(x) for p in puzzle["train"] for x in np.unique(p["input"]))
        train_touching = any(adjacent(p["input"] == s) for p in puzzle["train"])
        test_touching = any(adjacent(p["input"] == s) for p in puzzle["test"])
        regimes.add((train_touching, test_touching))
        mixed_train_shapes += int(len({p["input"].shape for p in puzzle["train"]}) > 1)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                mask = grid == s
                shapes.add(grid.shape)
                selector_counts.add(int(mask.sum()))
                masks.add(mask_key(mask))
                grids.add(key(grid))
                check(np.array_equal(pair["output"], expand(grid, b, s)), "independent rule check")
                horizontal = bool(np.any(mask[:, :-1] & mask[:, 1:]))
                vertical = bool(np.any(mask[:-1, :] & mask[1:, :]))
                coords = np.argwhere(mask)
                line = len(coords) >= 3 and (len(set(int(x) for x in coords[:, 0])) == 1
                                             or len(set(int(x) for x in coords[:, 1])) == 1)
                l_shape = any(mask[r, c] and mask[r+1, c] and mask[r, c+1]
                              for r in range(grid.shape[0]-1) for c in range(grid.shape[1]-1))
                symmetric = np.array_equal(mask, mask[:, ::-1]) or np.array_equal(mask, mask[::-1, :])
                multicolor = len(set(int(x) for x in np.unique(grid)) - {b, s}) >= 2
                content = int(np.sum((grid != b) & (grid != s)))
                for name, present in (("horizontal", horizontal), ("vertical", vertical),
                                      ("isolated", not horizontal and not vertical),
                                      ("line", line), ("l_shape", l_shape),
                                      ("symmetric", symmetric), ("multicolor", multicolor),
                                      ("novel", split == "test" and bool(set(int(x) for x in np.unique(grid)) - train_colors)),
                                      ("selector_majority", len(coords) > grid.size // 2),
                                      ("sparse_content", content == 1),
                                      ("nearly_filled", int(np.sum(grid == b)) == 1),
                                      ("rect_multi_touch", grid.shape[0] != grid.shape[1]
                                       and multicolor and (horizontal or vertical))):
                    features[name] += int(present)
    check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    check(len(puzzles) >= 40 and len(grids) >= 150 and len(masks) >= 80, "structural diversity")
    check(len(shapes) >= 7 and {(2, 3), (3, 4), (4, 4), (5, 5)} <= shapes, "scale diversity")
    check({1, 2, 3, 4, 5, 6, 7} <= selector_counts, "selector count diversity")
    check(len(palettes) >= 20, "puzzle role diversity")
    check(regimes == {(True, True), (False, True), (False, False)}, "train/test adjacency regimes")
    check(mixed_train_shapes >= 10, "within-puzzle shape diversity")
    check(features["horizontal"] >= 20 and features["vertical"] >= 20 and
          features["isolated"] >= 20 and features["line"] >= 4 and
          features["l_shape"] >= 10 and features["symmetric"] >= 20,
          "selector geometry diversity")
    check(features["multicolor"] >= 20 and features["novel"] >= 4 and
          features["selector_majority"] >= 10 and features["sparse_content"] >= 20 and
          features["nearly_filled"] >= 20 and features["rect_multi_touch"] >= 3,
          "tile content and combined variation diversity")
