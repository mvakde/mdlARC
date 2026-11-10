"""Sparse corruption of a solid field or a filled rectangular region."""

from functools import lru_cache

import numpy as np


_SOURCE_INPUTS = (
    ("333334", "433333", "333433", "333333", "333333", "333433"),
    ("666686", "866366", "666666", "668636", "366666", "666666"),
    ("0500000000", "0222222250", "0222222220", "5222222220",
     "0225225220", "0222222220", "0222222220", "0222252220",
     "0222222220", "0000000000"),
    ("1111111111", "0111110111", "1111011111", "1111010101",
     "1111111111", "1101011111", "1111111011", "1111111111",
     "1101110101", "1111111111", "1111111111", "1111101111"),
    ("00099000000", "00777777790", "00777777700", "00779777700",
     "00777779709", "00000000000"),
    ("000000", "000000", "000400", "000000", "400040", "000000"),
)


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("illegal pair count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


@lru_cache(maxsize=256)
def _bounds(height, width):
    # A separate rectangular region has at least a one-cell field on all sides.
    b = np.asarray(
        [(t, d, l, r)
         for t in range(1, height - 3)
         for d in range(t + 3, height)
         for l in range(1, width - 3)
         for r in range(l + 3, width)],
        dtype=np.int16,
    )
    return b.T if len(b) else np.empty((4, 0), dtype=np.int16)


def _best_fit(inp):
    """Minimum edits and number of minimum-cost clean fields/rectangles."""
    height, width = inp.shape
    colors, totals = np.unique(inp, return_counts=True)
    colors = [int(c) for c in colors]
    total = {c: int(n) for c, n in zip(colors, totals)}
    best = inp.size - max(total.values())
    ties = sum(inp.size - n == best for n in total.values())
    top, bottom, left, right = _bounds(height, width)
    if len(top) == 0:
        return best, ties
    counts = {}
    for color in colors:
        prefix = np.pad((inp == color).astype(np.int16).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        counts[color] = (prefix[bottom, right] - prefix[top, right]
                         - prefix[bottom, left] + prefix[top, left])
    for bg in colors:
        baseline = inp.size - total[bg]
        for color in colors:
            if color == bg:
                continue
            losses = baseline + counts[bg] - counts[color]
            minimum = int(losses.min())
            count = int(np.count_nonzero(losses == minimum))
            if minimum < best:
                best, ties = minimum, count
            elif minimum == best:
                ties += count
    return best, ties


def _structure(out):
    bg = int(out[0, 0])
    ys, xs = np.where(out != bg)
    if len(ys) == 0:
        return bg, None
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    color = int(out[top, left])
    assert 1 <= top and bottom < out.shape[0]
    assert 1 <= left and right < out.shape[1]
    assert bottom - top >= 3 and right - left >= 3
    assert color != bg
    assert np.all(out[top:bottom, left:right] == color)
    assert np.count_nonzero(out != bg) == (bottom - top) * (right - left)
    return bg, (top, bottom, left, right, color)


def _supported(inp, out, rect):
    same = inp == out
    if rect is None:
        return np.count_nonzero(same) >= 0.80 * inp.size
    top, bottom, left, right, color = rect
    inside = np.zeros(inp.shape, dtype=bool)
    inside[top:bottom, left:right] = True
    if np.count_nonzero(same & inside) < 0.65 * inside.sum():
        return False
    if np.count_nonzero(same & ~inside) < 0.78 * (~inside).sum():
        return False
    for edge in (inp[top, left:right], inp[bottom - 1, left:right],
                 inp[top:bottom, left], inp[top:bottom, right - 1]):
        if np.count_nonzero(edge == color) < max(2, int(np.ceil(0.45 * len(edge)))):
            return False
    return True


def _dimensions(rng):
    height = int(rng.integers(6, 22))
    width = height if rng.random() < 0.29 else int(rng.integers(6, 22))
    return height, width


def _rectangle(rng, height, width):
    if rng.random() < 0.16:
        rh, rw = height - 2, width - 2
    else:
        rh = int(rng.integers(3, height - 1))
        rw = int(rng.integers(3, width - 1))
    top = int(rng.integers(1, height - rh))
    left = int(rng.integers(1, width - rw))
    return top, top + rh, left, left + rw


def _noise(rng, out, rect, available, regime, highlight, combined):
    height, width = out.shape
    upper = max(3, min(19, int(np.ceil(0.15 * out.size))))
    goal = int(rng.integers(3, upper + 1))
    if regime == "target_test":
        goal = max(goal, int(rng.integers(3, upper + 1)))
    ncolors = 1 if highlight else int(rng.choice((1, 2, 3), p=(0.58, 0.32, 0.10)))
    ncolors = min(ncolors, len(available), goal)
    colors = [int(c) for c in rng.choice(available, size=ncolors, replace=False)]
    selected = []
    if highlight:
        top, bottom, left, right, _ = rect
        selected.append((int(rng.integers(top, bottom)), int(rng.integers(left, right))))
        outside = [(y, x) for y in range(height) for x in range(width)
                   if not (top <= y < bottom and left <= x < right)]
        selected.append(outside[int(rng.integers(len(outside)))])
    if combined:
        top, bottom, left, right, _ = rect
        corners = ((top, left), (top, right - 1),
                   (bottom - 1, left), (bottom - 1, right - 1))
        selected.append(corners[int(rng.integers(4))])
        interior_pairs = [((y, x), (y, x + 1))
                          for y in range(top + 1, bottom - 1)
                          for x in range(left + 1, right - 2)]
        interior_pairs += [((y, x), (y + 1, x))
                           for y in range(top + 1, bottom - 2)
                           for x in range(left + 1, right - 1)]
        selected.extend(interior_pairs[int(rng.integers(len(interior_pairs)))])
    if not combined and rng.random() < 0.37 and goal - len(selected) >= 2:
        anchor = (int(rng.integers(height)), int(rng.integers(width)))
        neighbors = [(anchor[0] + dy, anchor[1] + dx)
                     for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
                     if 0 <= anchor[0] + dy < height and 0 <= anchor[1] + dx < width]
        rng.shuffle(neighbors)
        if anchor not in selected:
            selected.append(anchor)
        for neighbor in neighbors:
            if neighbor not in selected:
                selected.append(neighbor)
                break
        if rng.random() < 0.28 and goal - len(selected) >= 1:
            for neighbor in neighbors:
                if neighbor not in selected:
                    selected.append(neighbor)
                    break
    used = {y * width + x for y, x in selected}
    for flat in rng.permutation(out.size):
        if len(selected) >= goal:
            break
        flat = int(flat)
        if flat not in used:
            selected.append(divmod(flat, width))
            used.add(flat)
    inp = out.copy()
    for y, x in selected:
        inp[y, x] = colors[int(rng.integers(len(colors)))]
    return inp


def _generate_pair(rng, regime, has_rect, train_palette, highlight=False, combined=False):
    for _ in range(16):
        height, width = _dimensions(rng)
        allowed = train_palette if regime == "source_train" or (not combined and rng.random() < 0.30) else tuple(range(10))
        bg = int(rng.choice([c for c in allowed if c not in train_palette])) if combined else int(rng.choice(allowed))
        rect = None
        if has_rect:
            rect_color = int(rng.choice([c for c in allowed if c != bg]))
            top, bottom, left, right = ((1, height - 1, 1, width - 1)
                                       if combined else _rectangle(rng, height, width))
            rect = (top, bottom, left, right, rect_color)
        else:
            rect_color = None
        available = [c for c in allowed if c != bg and c != rect_color]
        if not available:
            continue
        out = np.full((height, width), bg, dtype=np.int8)
        if rect is not None:
            out[top:bottom, left:right] = rect_color
        for _ in range(30):
            inp = _noise(rng, out, rect, available, regime, highlight, combined)
            changed = int(np.count_nonzero(inp != out))
            if (_supported(inp, out, rect)
                and _best_fit(inp) == (changed, 1)
                and tuple("".join(str(int(c)) for c in row) for row in inp) not in _SOURCE_INPUTS):
                return {"input": inp, "output": out}
    raise RuntimeError("unable to construct unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    palette = tuple(int(c) for c in rng.choice(10, size=int(rng.integers(3, 8)), replace=False))
    train_modes = [False, True] + [bool(rng.integers(2)) for _ in range(ntrain - 2)]
    rng.shuffle(train_modes)
    test_modes = ([False, True] + [bool(rng.integers(2)) for _ in range(ntest - 2)]
                  if ntest >= 2 else [bool(rng.integers(2))])
    rng.shuffle(test_modes)
    highlight_index = train_modes.index(True)
    pair_seeds = rng.integers(0, 2**63 - 1, size=ntrain + ntest, dtype=np.int64)
    combined_tests = [mode and rng.random() < 0.22 for mode in test_modes]
    puzzle = {"train": [], "test": []}
    for split, modes, offset in (("train", train_modes, 0), ("test", test_modes, ntrain)):
        for i, mode in enumerate(modes):
            pair = _generate_pair(np.random.default_rng(int(pair_seeds[offset + i])),
                                  "source_train" if split == "train" else "target_test",
                                  mode, palette, split == "train" and i == highlight_index,
                                  split == "test" and combined_tests[i])
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(p[x], q[x]) for p, q in zip(a[k], b[k]) for x in ("input", "output"))
        for k in ("train", "test"))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert type(train) is list and type(test) is list
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    structures = {"train": set(), "test": set()}
    train_colors = set()
    highlighted = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            input_key = (inp.shape, inp.tobytes())
            assert input_key not in seen
            seen.add(input_key)
            assert tuple("".join(str(int(c)) for c in row) for row in inp) not in _SOURCE_INPUTS
            bg, rect = _structure(out)
            structures[split].add(rect is not None)
            diff = inp != out
            changed = int(np.count_nonzero(diff))
            assert 3 <= changed <= min(19, int(np.ceil(0.15 * inp.size)))
            assert _supported(inp, out, rect)
            assert _best_fit(inp) == (changed, 1)
            clean_colors = {bg} if rect is None else {bg, rect[4]}
            assert set(int(c) for c in np.unique(inp[diff])).isdisjoint(clean_colors)
            if split == "train":
                train_colors.update(int(c) for c in np.unique(inp))
                if rect is not None:
                    top, bottom, left, right, _ = rect
                    inside = np.zeros(inp.shape, dtype=bool)
                    inside[top:bottom, left:right] = True
                    if np.any(diff & inside) and np.any(diff & ~inside):
                        if set(int(c) for c in np.unique(inp[diff & inside])) & set(int(c) for c in np.unique(inp[diff & ~inside])):
                            highlighted = True
    assert structures["train"] == {False, True}
    if len(test) >= 2:
        assert structures["test"] == {False, True}
    assert highlighted
    assert len(train_colors) >= 3


def self_test() -> None:
    counts, shapes, rect_shapes, signatures = set(), set(), set(), set()
    tally = {name: 0 for name in (
        "square_canvas", "nonsquare_canvas", "small_canvas", "large_canvas",
        "uniform", "rectangle", "square_rect", "wide_rect", "tall_rect",
        "near_full", "off_center", "corner", "top_intact", "bottom",
        "side", "interior", "shared_noise", "adjacent", "interior_adjacent",
        "boundary_run", "no_adjacency", "one_noise", "many_noise",
        "new_test_color", "new_background", "new_rectangle", "new_noise",
        "mixed_shapes", "mixed_noise_counts", "mixed_test_modes",
        "combined", "test_combined",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_colors = {int(c) for p in puzzle["train"] for c in np.unique(p["input"])}
        test_colors = {int(c) for p in puzzle["test"] for c in np.unique(p["input"])}
        tally["new_test_color"] += bool(test_colors - train_colors)
        local_shapes, local_noise_counts = set(), set()
        test_modes = set()
        for split, pair in ((split, p) for split in ("train", "test") for p in puzzle[split]):
            inp, out = pair["input"], pair["output"]
            signatures.add((inp.shape, inp.tobytes()))
            shapes.add(inp.shape)
            local_shapes.add(inp.shape)
            height, width = inp.shape
            tally["square_canvas" if height == width else "nonsquare_canvas"] += 1
            tally["small_canvas"] += height <= 8 and width <= 8
            tally["large_canvas"] += height >= 17 or width >= 17
            bg, rect = _structure(out)
            diff = inp != out
            assert _best_fit(inp) == (int(diff.sum()), 1)
            ncolors = len(np.unique(inp[diff]))
            local_noise_counts.add(ncolors)
            tally["one_noise" if ncolors == 1 else "many_noise"] += 1
            adjacent = bool(np.any(diff[:, 1:] & diff[:, :-1]) or np.any(diff[1:, :] & diff[:-1, :]))
            tally["adjacent" if adjacent else "no_adjacency"] += 1
            interior = diff.copy()
            interior[[0, -1], :] = False
            interior[:, [0, -1]] = False
            tally["interior_adjacent"] += bool(
                np.any(interior[:, 1:] & interior[:, :-1])
                or np.any(interior[1:, :] & interior[:-1, :])
            )
            if split == "test":
                test_modes.add(rect is not None)
                tally["new_background"] += bg not in train_colors
                if rect is not None:
                    tally["new_rectangle"] += rect[4] not in train_colors
                tally["new_noise"] += any(int(c) not in train_colors for c in np.unique(inp[diff]))
            if rect is None:
                tally["uniform"] += 1
            else:
                tally["rectangle"] += 1
                top, bottom, left, right, color = rect
                rh, rw = bottom - top, right - left
                rect_shapes.add((rh, rw))
                tally["square_rect" if rh == rw else "wide_rect" if rw > rh else "tall_rect"] += 1
                near_full = rh == height - 2 and rw == width - 2
                corner = any(diff[y, x] for y, x in (
                    (top, left), (top, right - 1),
                    (bottom - 1, left), (bottom - 1, right - 1)))
                tally["near_full"] += near_full
                tally["off_center"] += top != height - bottom or left != width - right
                tally["corner"] += corner
                tally["top_intact"] += not np.any(diff[top, left:right])
                tally["bottom"] += bool(np.any(diff[bottom - 1, left:right]))
                tally["side"] += bool(np.any(diff[top:bottom, left]) or np.any(diff[top:bottom, right - 1]))
                tally["interior"] += bool(np.any(diff[top + 1:bottom - 1, left + 1:right - 1]))
                inside = np.zeros(out.shape, dtype=bool)
                inside[top:bottom, left:right] = True
                tally["shared_noise"] += bool(
                    set(int(c) for c in np.unique(inp[diff & inside]))
                    & set(int(c) for c in np.unique(inp[diff & ~inside]))
                )
                boundary = np.zeros(out.shape, dtype=bool)
                boundary[top, left:right] = True
                boundary[bottom - 1, left:right] = True
                boundary[top:bottom, left] = True
                boundary[top:bottom, right - 1] = True
                damaged_boundary = diff & boundary
                tally["boundary_run"] += bool(
                    np.any(damaged_boundary[:, 1:] & damaged_boundary[:, :-1])
                    or np.any(damaged_boundary[1:, :] & damaged_boundary[:-1, :])
                )
                tally["combined"] += near_full and corner and adjacent and bool(test_colors - train_colors)
                if split == "test" and near_full and corner and bg not in train_colors:
                    core = diff[top + 1:bottom - 1, left + 1:right - 1]
                    tally["test_combined"] += bool(
                        np.any(core[:, 1:] & core[:, :-1])
                        or np.any(core[1:, :] & core[:-1, :])
                    )
        tally["mixed_shapes"] += len(local_shapes) > 1
        tally["mixed_noise_counts"] += len(local_noise_counts) > 1
        tally["mixed_test_modes"] += len(puzzle["test"]) > 1 and len(test_modes) == 2
    assert len({n for n, _ in counts}) >= 4 and len({n for _, n in counts}) >= 3
    assert len(shapes) >= 35 and len(rect_shapes) >= 25 and len(signatures) >= 100
    assert min(tally["square_canvas"], tally["nonsquare_canvas"]) >= 20
    assert min(tally["small_canvas"], tally["large_canvas"]) >= 15
    assert min(tally["uniform"], tally["rectangle"]) >= 40
    assert min(tally["square_rect"], tally["wide_rect"], tally["tall_rect"]) >= 10
    assert min(tally["near_full"], tally["corner"], tally["top_intact"],
               tally["bottom"], tally["side"], tally["interior"],
               tally["boundary_run"]) >= 10
    assert tally["shared_noise"] >= 30 and tally["off_center"] >= 30
    assert min(tally["adjacent"], tally["no_adjacency"],
               tally["interior_adjacent"], tally["one_noise"],
               tally["many_noise"]) >= 20
    assert min(tally["new_test_color"], tally["new_background"],
               tally["new_rectangle"], tally["new_noise"]) >= 10
    assert tally["mixed_shapes"] >= 40 and tally["mixed_noise_counts"] >= 30
    assert tally["mixed_test_modes"] >= 10 and tally["combined"] >= 2
    assert tally["test_combined"] >= 3
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        for seed in (0, 7, 29):
            puzzle = generate_puzzle(seed, ntrain, ntest)
            assert _equal_puzzles(puzzle, generate_puzzle(seed, ntrain, ntest))
            if ntrain is not None:
                assert len(puzzle["train"]) == ntrain
            if ntest is not None:
                assert len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
