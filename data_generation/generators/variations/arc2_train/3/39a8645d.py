"""Generate ARC puzzles that select the most frequent exact colored motif."""

import numpy as np


_SOURCE_SIGNATURES = {
    (
        "0000000101000000400001110000000440001000000040000000040000000000000044000000202004000101000200000001110020200000001000000400000000000000440000400000040000000440000001010040000000011100000000000010",
        "400044400",
    ),
    (
        "0000000000000000808000000000000800000080800080800000080000000000008080000000000000000000000000000000080800020200000080000202000008080000200000000000000000020200000000000202000000000000200000000000",
        "808080808",
    ),
    (
        "0000000000000000000000000000000800000660000088800006600000080000000600000000000000000000000000000000000000000000000000000800000000000088800000000000080000000000000000000000000000000000000000000000",
        "080888080",
    ),
    (
        "0000000000000000000000000000000303000020000000330002220000030300022000000000000000000000000000000000000008000000020000808000002220000800000022000000000000000002000303000000222000330000002200030300",
        "020222220",
    ),
}


def _validate_count_value(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _validate_count_value(num_train, "num_train", 2, 10)
    test = None if num_test is None else _validate_count_value(num_test, "num_test", 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    code = int(seed)
    if train is None and test is None:
        test = 1 + code % 4
        train = max(2, test + 1) + (code // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = max(2, test + 1) + code % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + code % min(4, train - 1)
    return train, test


def _key(array):
    return tuple(array.shape), tuple(int(x) for x in array.flat)


def _mask_key(mask):
    return tuple(mask.shape), tuple(int(x) for x in mask.flat)


def _components(mask):
    """Eight-neighbor components, recovered only from the visible grid."""
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        cells = [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (r + dr, c + dc)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        cells.append(neighbor)
        result.append(cells)
    return result


def _four_connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    seen = {min(cells)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(cells)


def _sample_mask(rng, shape, target=None, four_preference=None):
    h, w = shape
    low = max(h, w, 2)
    high = h * w
    if target is None:
        target = int(rng.integers(low, high + 1))
    target = max(low, min(high, target))
    for _ in range(240):
        mask = np.zeros(shape, dtype=np.int8)
        index = rng.choice(h * w, size=target, replace=False)
        mask.flat[index] = 1
        if not (np.any(mask[0]) and np.any(mask[-1]) and np.any(mask[:, 0]) and np.any(mask[:, -1])):
            continue
        if len(_components(mask)) != 1:
            continue
        if four_preference is not None and _four_connected(mask) != four_preference:
            continue
        return mask
    # A connected, tight fallback for rare narrow shapes or strict targets.
    mask = np.zeros(shape, dtype=np.int8)
    mask[0, :] = 1
    mask[:, w - 1] = 1
    mask[h - 1, :] = 1
    for r in range(h):
        for c in range(w):
            if int(mask.sum()) >= target:
                break
            mask[r, c] = 1
    return mask


def _near_mask(rng, mask):
    h, w = mask.shape
    choices = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(choices)
    for r, c in choices:
        other = mask.copy()
        other[r, c] = 1 - other[r, c]
        if np.any(other[0]) and np.any(other[-1]) and np.any(other[:, 0]) and np.any(other[:, -1]) and len(_components(other)) == 1:
            return other
    if h == w:
        for transformed in (np.rot90(mask), np.fliplr(mask)):
            if not np.array_equal(transformed, mask):
                return transformed.copy()
    return None


def _profile(rng, regime):
    classes = int(rng.choice([2, 3, 4], p=[0.37, 0.42, 0.21] if regime == "source_train" else [0.21, 0.43, 0.36]))
    winner = int(rng.integers(2, 6))
    if classes == 4:
        winner = max(3, winner)
    counts = [winner]
    for i in range(classes - 1):
        if regime == "target_test" and rng.random() < 0.55:
            count = winner - 1
        else:
            count = int(rng.integers(1, winner))
        counts.append(count)
    if sum(counts) > 14:
        counts = [counts[0]] + [min(x, 2) for x in counts[1:]]
    return counts


def _shape_and_canvas(rng, total, special):
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]
    if special == "pixel_decoy":
        shapes = [s for s in shapes if s[0] * s[1] >= 9]
    shape = shapes[int(rng.integers(len(shapes)))]
    h, w = shape
    rows = int(np.ceil(np.sqrt(total)))
    columns = int(np.ceil(total / rows))
    min_h = max(10, rows * (h + 1) - 1)
    min_w = max(10, columns * (w + 1) - 1)
    if min_h > 30 or min_w > 30:
        shape = (3, 3)
        h, w = shape
        min_h = max(10, rows * 4 - 1)
        min_w = max(10, columns * 4 - 1)
    if min_h <= 14 and min_w <= 14 and rng.random() < 0.22:
        canvas = (14, 14)
    else:
        canvas = (int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))
    return shape, canvas


def _colors(rng, background, classes, pool, winner_hint=None, same_color=False):
    available = [c for c in pool if c != background]
    available += [c for c in range(10) if c != background and c not in available]
    rng.shuffle(available)
    if winner_hint is not None and winner_hint != background:
        winner = int(winner_hint)
        available.remove(winner)
        available.insert(0, winner)
    colors = available[:classes]
    if same_color:
        # Only distractor shapes share a color. Their counts remain separate.
        colors[-1] = colors[1] if classes > 2 else colors[0]
    return colors


def _make_masks(rng, shape, counts, special, history, same_color):
    n = len(counts)
    area = shape[0] * shape[1]
    masks = []
    if special == "equal_area":
        common = int(rng.integers(max(shape), area + 1))
        targets = [common] * n
    elif special == "pixel_decoy":
        counts[0] = max(3, counts[0])
        counts[1] = counts[0] - 1
        small = max(shape)
        large = min(area, max(small + 2, int(np.ceil((counts[0] * small + 1) / counts[1]))))
        targets = [small, large] + [None] * (n - 2)
    else:
        targets = [None] * n
    for i in range(n):
        reuse = [m for m in history if m.shape == shape and all(not np.array_equal(m, old) for old in masks)]
        mask = None
        if special == "ordinary" and reuse and rng.random() < 0.26:
            mask = reuse[int(rng.integers(len(reuse)))].copy()
        if mask is None and i > 0 and targets[i] is None:
            parent = masks[int(rng.integers(i))]
            if shape[0] == shape[1] and rng.random() < (0.20 if same_color else 0.08):
                turned = np.rot90(parent, int(rng.integers(1, 4)))
                if not np.array_equal(turned, parent):
                    mask = turned.copy()
            if mask is None and rng.random() < (0.48 if same_color else 0.19):
                mask = _near_mask(rng, parent)
        for _ in range(50):
            if mask is None:
                pref = bool(rng.integers(2)) if shape[0] * shape[1] >= 6 and rng.random() < 0.65 else None
                mask = _sample_mask(rng, shape, targets[i], pref)
            if all(not np.array_equal(mask, old) for old in masks) and (targets[i] is None or int(mask.sum()) == targets[i]):
                break
            mask = None
        if mask is None:
            raise ValueError("could not obtain distinct masks")
        masks.append(mask)
    if special == "pixel_decoy" and counts[1] * int(masks[1].sum()) <= counts[0] * int(masks[0].sum()):
        raise ValueError("pixel decoy not strong enough")
    return masks


def _place(rng, canvas, background, masks, colors, counts):
    height, width = canvas
    mh, mw = masks[0].shape
    instances = [i for i, count in enumerate(counts) for _ in range(count)]
    rng.shuffle(instances)
    anchors = [(r, c) for r in range(height - mh + 1) for c in range(width - mw + 1)]
    for _ in range(64):
        grid = np.full(canvas, background, dtype=np.int8)
        occupied = np.zeros(canvas, dtype=bool)
        reserved_windows = np.zeros(canvas, dtype=bool)
        forbidden = {color: np.zeros(canvas, dtype=bool) for color in set(colors)}
        placed = []
        for i in instances:
            mask = masks[i].astype(bool)
            valid = []
            for r, c in anchors:
                if not np.any(reserved_windows[r:r + mh, c:c + mw]) and not np.any(occupied[r:r + mh, c:c + mw] & mask) and not np.any(forbidden[colors[i]][r:r + mh, c:c + mw] & mask):
                    valid.append((r, c))
            if not valid:
                break
            # Full-grid choices scatter locations; occasional row/column reuse
            # and a near-neighbor bias add aligned and compact arrangements.
            aligned = [(r, c) for r, c in valid if any(r == pr or c == pc for pr, pc in placed)]
            if aligned and rng.random() < 0.17:
                r, c = aligned[int(rng.integers(len(aligned)))]
            elif placed and rng.random() < 0.32:
                pr, pc = placed[int(rng.integers(len(placed)))]
                distances = [abs(r - pr) + abs(c - pc) for r, c in valid]
                order = np.argsort(distances)
                subset = [valid[int(j)] for j in order[:max(1, len(order) // 5)]]
                r, c = subset[int(rng.integers(len(subset)))]
            else:
                r, c = valid[int(rng.integers(len(valid)))]
            grid[r:r + mh, c:c + mw][mask] = colors[i]
            reserved_windows[r:r + mh, c:c + mw] = True
            rr, cc = np.nonzero(mask)
            for dr, dc in zip(rr, cc):
                y, x = r + int(dr), c + int(dc)
                occupied[y, x] = True
                forbidden[colors[i]][max(0, y - 1):min(height, y + 2), max(0, x - 1):min(width, x + 2)] = True
            placed.append((r, c))
        if len(placed) == len(instances):
            return grid
    raise ValueError("placement failed")


def _analyze(input_grid, output_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    assert counts.max() > input_grid.size // 2, "background is unclear"
    objects = []
    for color in values:
        color = int(color)
        if color == background:
            continue
        for cells in _components(input_grid == color):
            rs, cs = zip(*cells)
            r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
            mask = input_grid[r0:r1, c0:c1] == color
            assert len(cells) == int(mask.sum())
            objects.append((r0, c0, color, mask))
    assert len(objects) >= 3
    shapes = {mask.shape for _, _, _, mask in objects}
    assert len(shapes) == 1, "motif window changes within a pair"
    groups = {}
    inks = {}
    for _, _, color, mask in objects:
        key = (color, _mask_key(mask))
        groups[key] = groups.get(key, 0) + 1
        inks[key] = int(mask.sum())
    assert 2 <= len(groups) <= 4
    maximum = max(groups.values())
    winners = [key for key, count in groups.items() if count == maximum]
    assert len(winners) == 1, "modal class is tied"
    winner = winners[0]
    shape, bits = winner[1]
    expected = np.where(np.array(bits, dtype=bool).reshape(shape), winner[0], background).astype(np.int8)
    assert np.array_equal(output_grid, expected), "incorrect modal crop"
    return {"background": background, "objects": objects, "groups": groups, "inks": inks, "winner": winner,
            "same_color": len({key[0] for key in groups}) < len(groups)}


def _make_pair(rng, regime, special, pool, history, winner_hint=None, same_color=False):
    for _ in range(90):
        counts = _profile(rng, regime)
        if special == "pixel_decoy":
            counts[0] = max(3, counts[0])
            counts[1] = counts[0] - 1
        shape, canvas = _shape_and_canvas(rng, sum(counts), special)
        try:
            masks = _make_masks(rng, shape, counts, special, history, same_color)
        except ValueError:
            continue
        background = 0 if rng.random() < 0.64 else int(rng.integers(1, 10))
        if winner_hint is not None and background == winner_hint:
            background = (background + 1) % 10
        colors = _colors(rng, background, len(counts), pool, winner_hint, same_color)
        try:
            input_grid = _place(rng, canvas, background, masks, colors, counts)
            output_grid = np.where(masks[0] != 0, colors[0], background).astype(np.int8)
            analysis = _analyze(input_grid, output_grid)
            if special == "pixel_decoy":
                totals = {k: analysis["groups"][k] * analysis["inks"][k] for k in analysis["groups"]}
                if totals[analysis["winner"]] >= max(totals.values()):
                    continue
            if special == "equal_area" and len(set(analysis["inks"].values())) != 1:
                continue
            return {"input": input_grid, "output": output_grid}, masks
        except (AssertionError, ValueError):
            continue
    raise ValueError("pair construction failed")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # The only shared convention is exact color-and-shape frequency selection.
    # The pool encourages some visual continuity without assigning semantic roles.
    pool = [int(x) for x in rng.choice(np.arange(10), size=int(rng.integers(4, 7)), replace=False)]
    demonstrate_same_color = bool(rng.integers(2))
    same_color_index = int(rng.integers(train_count)) if demonstrate_same_color else -1
    specials = ["pixel_decoy", "equal_area"] + ["ordinary"] * (train_count - 2)
    rng.shuffle(specials)
    history = []
    puzzle = {"train": [], "test": []}
    used = set()
    train_winners = []
    train_losers = []
    for split, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            regime = "source_train" if split == "train" else "target_test"
            special = specials[index] if split == "train" else ("equal_area" if rng.random() < 0.22 else "ordinary")
            same_color = (index == same_color_index if split == "train" else demonstrate_same_color and rng.random() < 0.38)
            hint = None
            if split == "test":
                mode = int(rng.integers(3))
                source = train_winners if mode == 0 else train_losers if mode == 1 else []
                if source:
                    hint = source[int(rng.integers(len(source)))]
                elif mode == 2:
                    unseen = [c for c in range(10) if c not in set(train_winners + train_losers)]
                    if unseen:
                        hint = unseen[int(rng.integers(len(unseen)))]
            for _ in range(40):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                try:
                    pair, masks = _make_pair(pair_rng, regime, special, pool, history, hint, same_color)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise ValueError("could not construct unique pair")
            used.add(key)
            puzzle[split].append(pair)
            history.extend(mask.copy() for mask in masks)
            if split == "train":
                analysis = _analyze(pair["input"], pair["output"])
                train_winners.append(analysis["winner"][0])
                train_losers.extend(k[0] for k in analysis["groups"] if k != analysis["winner"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert _key(inp) not in seen_inputs
            seen_inputs.add(_key(inp))
            assert not np.array_equal(inp, out)
            assert ("".join(str(int(v)) for v in inp.flat), "".join(str(int(v)) for v in out.flat)) not in _SOURCE_SIGNATURES
            analyses[split].append(_analyze(inp, out))
    # Two distinct training constructions show that area and raw colored-pixel
    # totals cannot replace complete-motif frequency.
    assert any(len(set(a["inks"].values())) == 1 for a in analyses["train"])
    assert any(a["groups"][a["winner"]] * a["inks"][a["winner"]] <
               max(a["groups"][k] * a["inks"][k] for k in a["groups"]) for a in analyses["train"])
    if any(a["same_color"] for a in analyses["test"]):
        assert any(a["same_color"] for a in analyses["train"])


def self_test() -> None:
    puzzles = []
    train_counts, test_counts = set(), set()
    dimensions, output_dimensions, profiles, backgrounds = set(), set(), set(), set()
    same_colors = 0
    close_masks = 0
    near_pairs = 0
    equal_area = 0
    intermediate = 0
    winner_colors = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze(pair["input"], pair["output"])
                dimensions.add(pair["input"].shape)
                output_dimensions.add(pair["output"].shape)
                profiles.add(tuple(sorted(analysis["groups"].values(), reverse=True)))
                backgrounds.add(analysis["background"])
                winner_colors.add(analysis["winner"][0])
                same_colors += analysis["same_color"]
                equal_area += len(set(analysis["inks"].values())) == 1
                areas = sorted(set(analysis["inks"].values()))
                intermediate += len(areas) > 2 and areas[0] < analysis["inks"][analysis["winner"]] < areas[-1]
                objs = analysis["objects"]
                near_pairs += any(abs(a[0] - b[0]) < 5 and abs(a[1] - b[1]) < 5
                                  for i, a in enumerate(objs) for b in objs[i + 1:])
                keys = [k[1] for k in analysis["groups"]]
                close_masks += any(x[0] == y[0] and sum(u != v for u, v in zip(x[1], y[1])) == 1
                                   for i, x in enumerate(keys) for y in keys[i + 1:])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 20 and len(output_dimensions) >= 6
    assert len(profiles) >= 12 and len(backgrounds) >= 5
    assert len(winner_colors) >= 7 and same_colors >= 10
    assert near_pairs >= 10 and equal_area >= 20 and close_masks >= 5 and intermediate >= 5
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(3000 + i, tr, te)
        assert len(p["train"]) == tr and len(p["test"]) == te
        validate_generated_puzzle(p)
    for i, (tr, te) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        validate_generated_puzzle(generate_puzzle(4000 + i, tr, te))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(5000, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for seed, tr, te in ((17, None, None), (12345, 6, 3)):
        a, b = generate_puzzle(seed, tr, te), generate_puzzle(seed, tr, te)
        assert all(_key(x[role]) == _key(y[role]) for split in ("train", "test")
                   for x, y in zip(a[split], b[split]) for role in ("input", "output"))
    assert _key(generate_puzzle(2001, 5, 2)["test"][0]["input"]) != _key(generate_puzzle(2002, 5, 2)["test"][0]["input"])
