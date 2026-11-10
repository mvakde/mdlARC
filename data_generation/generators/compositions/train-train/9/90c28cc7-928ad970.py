"""One tile, one vote: an inset district elects its elastic frame's ink.

Only NumPy and the Python standard library are used. Generation constructs from
logical tiles; solve reconstructs everything from input pixels independently.
"""

from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or array.dtype.kind not in "iu"
            or any(not 1 <= side <= 30 for side in array.shape)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array


def _key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def _mode(values, weights=None):
    """Return the unique mode, or None when the maximum is tied."""
    totals = np.bincount(np.asarray(values).ravel(),
                         weights=None if weights is None else np.asarray(weights).ravel(),
                         minlength=10)
    winners = np.flatnonzero(totals == totals.max())
    return int(winners[0]) if len(winners) == 1 else None


def _ring(shape, box):
    top, bottom, left, right = box
    mask = np.zeros(shape, dtype=bool)
    mask[top, left:right + 1] = True
    mask[bottom, left:right + 1] = True
    mask[top:bottom + 1, left] = True
    mask[top:bottom + 1, right] = True
    return mask


def _expand(tiles, rows, cols):
    return np.repeat(np.repeat(tiles, rows, axis=0), cols, axis=1)


def _boundaries_visible(tiles):
    return (np.all(np.any(tiles[1:] != tiles[:-1], axis=1))
            and np.all(np.any(tiles[:, 1:] != tiles[:, :-1], axis=0)))


def _decode(input_grid):
    """Parse the family from input alone; no sampled parameters are consulted."""
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (grid[0, -1] == grid[-1, 0] == grid[-1, -1] == background):
        raise ValueError("The corners must identify a common background")
    occupied = np.argwhere(grid != background)
    if len(occupied) == 0:
        raise ValueError("Missing mosaic")
    top, left = occupied.min(axis=0).astype(int)
    bottom, right = occupied.max(axis=0).astype(int) + 1
    if min(top, left, grid.shape[0] - bottom, grid.shape[1] - right) < 1:
        raise ValueError("The mosaic needs positive background margins")
    crop = grid[top:bottom, left:right]
    if np.any(crop == background):
        raise ValueError("The mosaic must be filled")
    row_starts = np.r_[0, np.flatnonzero(np.any(crop[1:] != crop[:-1], axis=1)) + 1]
    col_starts = np.r_[0, np.flatnonzero(np.any(crop[:, 1:] != crop[:, :-1], axis=0)) + 1]
    rows = np.diff(np.r_[row_starts, crop.shape[0]])
    cols = np.diff(np.r_[col_starts, crop.shape[1]])
    tiles = crop[np.ix_(row_starts, col_starts)].copy()
    if not np.array_equal(crop, _expand(tiles, rows, cols)):
        raise ValueError("The row/column bands must form uniform tiles")
    colors, counts = np.unique(tiles, return_counts=True)
    candidates = colors[counts == 4]
    if len(candidates) != 1:
        raise ValueError("Exactly one color must identify four pin tiles")
    pin = int(candidates[0])
    if len(colors) < 3 or np.any(counts[colors != pin] < 5):
        raise ValueError("Each of at least two content colors must occupy five or more tiles")
    pins = np.argwhere(tiles == pin)
    mt, ml = pins.min(axis=0).astype(int)
    mb, mr = pins.max(axis=0).astype(int)
    extrema = (np.flatnonzero(pins[:, 0] == mt), np.flatnonzero(pins[:, 0] == mb),
               np.flatnonzero(pins[:, 1] == ml), np.flatnonzero(pins[:, 1] == mr))
    if any(len(indices) != 1 for indices in extrema):
        raise ValueError("Pin extrema must be unique")
    owners = [int(indices[0]) for indices in extrema]
    if len(set(owners)) != 4:
        raise ValueError("Four different pins must own the four extrema")
    frame = (mt + 1, mb - 1, ml + 1, mr - 1)
    t, b, l, r = frame
    if b - t - 1 < 2 or r - l - 1 < 2:
        raise ValueError("The frame's open interior must be at least 2 by 2")
    ti, bi, li, ri = owners
    if not (l < pins[ti, 1] < r and l < pins[bi, 1] < r
            and t < pins[li, 0] < b and t < pins[ri, 0] < b):
        raise ValueError("Pins must lie opposite the interiors of their frame sides")
    for index, position in enumerate(pins):
        if any(np.max(np.abs(position - other)) <= 1
               for j, other in enumerate(pins) if j != index):
            raise ValueError("Pin tiles must be isolated from one another")
    voters = tiles[t + 1:b, l + 1:r]
    if pin in voters or len(np.unique(voters)) < 2:
        raise ValueError("Voters must contain at least two content colors and no pins")
    winner = _mode(voters)
    if winner is None:
        raise ValueError("The interior vote must have a unique plurality")
    vertical = pins[ti, 1] == pins[bi, 1]
    horizontal = pins[li, 0] == pins[ri, 0]
    alignment = "both" if vertical and horizontal else "one" if vertical or horizontal else "none"
    pixel_weights = rows[t + 1:b, None] * cols[None, l + 1:r]
    ring = _ring(tiles.shape, frame)
    return {"background": background, "pin": pin, "tiles": tiles,
            "rows": rows, "cols": cols, "crop": (top, bottom, left, right),
            "pins": pins, "pin_box": (mt, mb, ml, mr), "frame": frame,
            "voters": voters, "winner": winner, "ring": ring,
            "interior_shape": voters.shape, "alignment": alignment,
            "pixel_winner": _mode(voters, pixel_weights),
            "global_winner": _mode(tiles[tiles != pin]),
            "ring_winner": _mode(tiles[ring])}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Elect and paint the frame using only the visible input."""
    meta = _decode(input_grid)
    painted = meta["tiles"].copy()
    painted[meta["ring"]] = meta["winner"]
    output = np.asarray(input_grid).copy()
    top, bottom, left, right = meta["crop"]
    output[top:bottom, left:right] = _expand(painted, meta["rows"], meta["cols"])
    return output


def _hidden_seam(meta):
    """A boundary invisible locally but detectable in full mosaic rows/columns."""
    patch = meta["voters"]
    return (any(np.array_equal(patch[i - 1], patch[i]) for i in range(1, patch.shape[0]))
            or any(np.array_equal(patch[:, j - 1], patch[:, j])
                   for j in range(1, patch.shape[1])))


def _thick_frame(meta):
    top, bottom, left, right = meta["frame"]
    return max(meta["rows"][top], meta["rows"][bottom],
               meta["cols"][left], meta["cols"][right]) > 1


def _ballots(rng, shape, palette, striped, plurality=False):
    height, width = shape
    if striped:
        axes = [axis for axis, length in enumerate(shape) if length >= 3]
        axis = int(rng.choice(axes))
        loser_band = int(rng.integers(shape[axis]))
        patch = np.full(shape, palette[0], dtype=np.int8)
        if axis == 0:
            patch[loser_band, :] = palette[1]
        else:
            patch[:, loser_band] = palette[1]
        return patch, (axis, loser_band)
    # Favor palette[0] without assuming majority, topology, or a center cue.
    for _ in range(500):
        probability = float(rng.uniform(.36, .70)) if len(palette) > 2 else .63
        weights = np.full(len(palette), (1 - probability) / (len(palette) - 1))
        weights[0] = probability
        patch = rng.choice(palette, size=(height, width), p=weights).astype(np.int8)
        if (len(np.unique(patch)) > 1 and _mode(patch) == int(palette[0])
                and (not plurality or (
                    np.count_nonzero(patch == palette[0]) * 2 < patch.size
                    and patch[height // 2, width // 2] != palette[0]))):
            return patch, None
    raise RuntimeError("Could not sample an untied election")


def _logical_axis(rng, interior_length):
    total_margin = int(rng.integers(0, min(4, 13 - interior_length - 4) + 1))
    before = int(rng.integers(total_margin + 1))
    return interior_length + 4 + total_margin, before + 2


def _bands(rng, length, mode):
    if mode == "uniform":
        vector = np.full(length, int(rng.integers(1, min(3, 28 // length) + 1)))
    elif mode == "peak":
        vector = rng.integers(1, 3, size=length)
        vector[int(rng.integers(length))] = int(rng.integers(4, 9))
    else:
        vector = rng.integers(1, 4, size=length)
    while vector.sum() > 28:
        vector[int(rng.choice(np.flatnonzero(vector > 1)))] -= 1
    return vector


def _scales(rng, shape, interior_start, interior_shape, stripe, uniform):
    if uniform:
        return _bands(rng, shape[0], "uniform"), _bands(rng, shape[1], "uniform")
    style = int(rng.integers(5))
    styles = (("mild", "mild"), ("uniform", "mild"), ("mild", "uniform"),
              ("peak", "mild"), ("mild", "peak"))
    row_mode, col_mode = styles[style]
    vectors = [_bands(rng, shape[0], row_mode), _bands(rng, shape[1], col_mode)]
    if stripe is not None:
        axis, loser_band = stripe
        start, length = interior_start[axis], interior_shape[axis]
        vector = rng.integers(1, 3, size=shape[axis])
        vector[start:start + length] = 1
        vector[start + loser_band] = int(rng.integers(max(6, length), 10))
        while vector.sum() > 28:
            adjustable = [i for i in range(len(vector))
                          if not start <= i < start + length and vector[i] > 1]
            vector[int(rng.choice(adjustable))] -= 1
        vectors[axis] = vector
        # At least one frame side has a visibly thick logical band.
        other = 1 - axis
        vectors[other][interior_start[other] - 1] = 2
        if vectors[other].sum() > 28:
            for i in range(len(vectors[other])):
                if i != interior_start[other] - 1 and vectors[other][i] > 1:
                    vectors[other][i] -= 1
                    break
    return vectors[0], vectors[1]


def _draw_pair(rng, interior_shape, alignment, *, striped=False, uniform=False,
               witness=False, previous_roles=None, plurality=False):
    """Build directly in logical coordinates, not by calling solve."""
    for _ in range(300):
        height, ir = _logical_axis(rng, interior_shape[0])
        width, ic = _logical_axis(rng, interior_shape[1])
        if plurality:
            content_count = 4 if np.prod(interior_shape) == 8 else int(rng.integers(3, 5))
        else:
            content_count = int(rng.integers(2, 5)) if height * width >= 60 else 2
        palette_order = rng.permutation(10)
        background, pin = map(int, palette_order[:2])
        palette = palette_order[2:2 + content_count]
        if previous_roles is not None and (
                background == previous_roles[0] or pin == previous_roles[1]
                or palette[0] == previous_roles[2]):
            continue
        patch, stripe = _ballots(rng, interior_shape, palette, striped, plurality)
        winner = int(palette[0])
        t, b = ir - 1, ir + interior_shape[0]
        l, r = ic - 1, ic + interior_shape[1]
        frame = (t, b, l, r)
        ring = _ring((height, width), frame)

        top_col = int(rng.integers(ic, r))
        left_row = int(rng.integers(ir, b))
        bottom_col = top_col
        right_row = left_row
        if alignment == "none" or (alignment == "one" and int(rng.integers(2)) == 0):
            bottom_col = int(rng.choice([x for x in range(ic, r) if x != top_col]))
        if alignment == "none" or (alignment == "one" and bottom_col == top_col):
            right_row = int(rng.choice([x for x in range(ir, b) if x != left_row]))
        pins = ((t - 1, top_col), (b + 1, bottom_col),
                (left_row, l - 1), (right_row, r + 1))

        # Sometimes the surroundings support the local winner, sometimes an
        # opponent; the two learning anchors deliberately distinguish them.
        bias_color = 1 if witness else int(rng.integers(content_count))
        bias = float(rng.uniform(.52, .69)) if plurality else (
            float(rng.uniform(.58, .82)) if witness else float(rng.uniform(.28, .78)))
        weights = np.full(content_count, (1 - bias) / (content_count - 1))
        weights[bias_color] = bias
        tiles = None
        for _ in range(160):
            candidate = rng.choice(palette, size=(height, width), p=weights).astype(np.int8)
            candidate[ir:b, ic:r] = patch
            for row, col in pins:
                candidate[row, col] = pin
            if not _boundaries_visible(candidate):
                continue
            if any(np.count_nonzero(candidate == color) < 5 for color in palette):
                continue
            if witness and (_mode(candidate[candidate != pin]) in (None, winner)
                            or _mode(candidate[ring]) in (None, winner)):
                continue
            tiles = candidate
            break
        if tiles is None:
            continue
        rows, cols = _scales(rng, tiles.shape, (ir, ic), interior_shape, stripe, uniform)
        expanded = _expand(tiles, rows, cols)
        extra_h = int(rng.integers(2, min(7, 30 - expanded.shape[0]) + 1))
        extra_w = int(rng.integers(2, min(7, 30 - expanded.shape[1]) + 1))
        top_margin = int(rng.integers(1, extra_h))
        left_margin = int(rng.integers(1, extra_w))
        input_grid = np.full((expanded.shape[0] + extra_h, expanded.shape[1] + extra_w),
                             background, dtype=np.int8)
        input_grid[top_margin:top_margin + expanded.shape[0],
                   left_margin:left_margin + expanded.shape[1]] = expanded
        painted = tiles.copy()
        painted[ring] = winner
        output_grid = input_grid.copy()
        output_grid[top_margin:top_margin + expanded.shape[0],
                    left_margin:left_margin + expanded.shape[1]] = _expand(painted, rows, cols)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("Could not construct a compatible distinct election")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    alignments = list(rng.permutation(("both", "one", "none")))
    first_shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    if max(first_shape) == 2:
        first_shape = (2, int(rng.integers(3, 5)))
    second_options = [(h, w) for h in range(2, 5) for w in range(2, 5)
                      if h * w >= 8 and (h, w) != first_shape]
    second_shape = second_options[int(rng.integers(len(second_options)))]
    train_specs = [(first_shape, alignments[0], True, False, True),
                   (second_shape, alignments[1], False, True, True)]
    for _ in range(num_train - 2):
        shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        train_specs.append((shape, str(rng.choice(alignments)), False,
                            bool(rng.random() < .18), False))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_roles = None

    def add(split, shape, alignment, striped, uniform, witness, roles=None):
        for _ in range(30):
            pair = _draw_pair(rng, shape, alignment, striped=striped, uniform=uniform,
                              witness=witness, previous_roles=roles,
                              plurality=(split == "train" and witness and uniform))
            key = _key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                puzzle[split].append(pair)
                return _decode(pair["input"])
        raise RuntimeError("Unable to construct unique inputs")

    for index, spec in enumerate(train_specs):
        meta = add("train", *spec, roles=previous_roles if index == 1 else None)
        if index == 0:
            previous_roles = (meta["background"], meta["pin"], meta["winner"])
    # The anchors carry distinct evidence, not a fixed display order.
    rng.shuffle(puzzle["train"])
    extension_axis = int(rng.integers(2))
    for index in range(num_test):
        axis = extension_axis if index == 0 else int(rng.integers(2))
        shape = [int(rng.integers(2, 6)), int(rng.integers(2, 6))]
        shape[axis] = int(rng.integers(5, 7))
        add("test", tuple(shape), str(rng.choice(alignments)),
            bool(rng.random() < .35), bool(rng.random() < .15), False)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the actual grids, construction, and minimum learning witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    train_meta, test_meta = [], []
    for split, collected in (("train", train_meta), ("test", test_meta)):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            meta = _decode(inp)
            collected.append(meta)
            tiles, winner = meta["tiles"], meta["winner"]
            top, bottom, left, right = meta["crop"]
            frame_pixels = np.zeros(inp.shape, dtype=bool)
            frame_pixels[top:bottom, left:right] = _expand(meta["ring"], meta["rows"], meta["cols"])
            assert np.all(out[frame_pixels] == winner)
            assert np.array_equal(out[~frame_pixels], inp[~frame_pixels])
            assert np.any(out != inp)
            assert np.array_equal(solve(inp), out)
            # Expected logical tiles are checked at known INPUT band positions;
            # painting may merge output bands, so never recompress the output.
            rstarts = np.r_[0, np.cumsum(meta["rows"])[:-1]]
            cstarts = np.r_[0, np.cumsum(meta["cols"])[:-1]]
            sampled_output = out[top:bottom, left:right][np.ix_(rstarts, cstarts)]
            expected_tiles = tiles.copy()
            expected_tiles[meta["ring"]] = winner
            assert np.array_equal(sampled_output, expected_tiles)
            assert np.all(sampled_output[tiles == meta["pin"]] == meta["pin"])
            # All parent's A pairs shrink; B pairs have only three colors.
            assert len(np.unique(inp)) >= 4
    assert len({m["interior_shape"] for m in train_meta}) >= 2
    assert len({m["alignment"] for m in train_meta}) >= 2
    for role in ("background", "pin", "winner"):
        assert len({m[role] for m in train_meta}) >= 2
    assert any(m["pixel_winner"] is not None and m["pixel_winner"] != m["winner"]
               and _hidden_seam(m) and _thick_frame(m) for m in train_meta)
    assert any(m["pixel_winner"] == m["winner"] for m in train_meta)
    assert any(np.count_nonzero(m["voters"] == m["winner"]) * 2 < m["voters"].size
               and m["voters"][m["voters"].shape[0] // 2,
                                m["voters"].shape[1] // 2] != m["winner"]
               and m["pixel_winner"] == m["winner"] for m in train_meta)
    assert sum(m["global_winner"] not in (None, m["winner"])
               and m["ring_winner"] not in (None, m["winner"]) for m in train_meta) >= 2
    assert all(max(m["interior_shape"]) >
               max(max(t["interior_shape"]) for t in train_meta) for m in test_meta)


def _component_count(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded schema, ablation, reproducibility, and 42-seed variation checks."""
    shapes, logical_shapes, voting_shapes, profiles, palettes, positions = (set() for _ in range(6))
    alignments, winning_colors, puzzle_keys = set(), set(), set()
    structures = set()
    stats = Counter()
    split_shapes = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                meta = _decode(pair["input"])
                tiles, voters, winner = meta["tiles"], meta["voters"], meta["winner"]
                shapes.add(pair["input"].shape)
                logical_shapes.add(tiles.shape)
                voting_shapes.add(voters.shape)
                split_shapes[split].add(voters.shape)
                profiles.add((tuple(meta["rows"]), tuple(meta["cols"])))
                positions.add(meta["pin_box"])
                palettes.add((meta["background"], meta["pin"], winner,
                              tuple(sorted(set(map(int, tiles.ravel()))))))
                alignments.add(meta["alignment"])
                winning_colors.add(winner)
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in tiles.ravel())
                structures.add((tiles.shape, normalized))
                stats["pairs"] += 1
                stats["area_reversal"] += meta["pixel_winner"] not in (None, winner)
                stats["area_agreement"] += meta["pixel_winner"] == winner
                stats["global_disagrees"] += meta["global_winner"] not in (None, winner)
                stats["global_agrees"] += meta["global_winner"] == winner
                stats["hidden_local_seam"] += _hidden_seam(meta)
                stats["submajority_winner"] += np.count_nonzero(voters == winner) * 2 < voters.size
                stats["center_loses"] += voters[voters.shape[0] // 2, voters.shape[1] // 2] != winner
                stats["disconnected_winner"] += _component_count(voters == winner) > 1
                stats["uniform_both"] += len(set(meta["rows"])) == len(set(meta["cols"])) == 1
                stats["uneven_both"] += len(set(meta["rows"])) > 1 and len(set(meta["cols"])) > 1
                stats["one_pixel_band"] += min(min(meta["rows"]), min(meta["cols"])) == 1
                stats["large_band"] += max(max(meta["rows"]), max(meta["cols"])) >= 6
                stats["canvas_30"] += 30 in pair["input"].shape
                stats["test_reversal"] += split == "test" and meta["pixel_winner"] not in (None, winner)
                stats["test_agreement"] += split == "test" and meta["pixel_winner"] == winner

    assert len(puzzle_keys) == 42
    assert len(shapes) >= 50 and len(logical_shapes) >= 25
    assert len(voting_shapes) >= 20 and len(profiles) >= 120
    assert len(positions) >= 100 and len(palettes) >= 150
    assert len(structures) >= 180 and len(winning_colors) == 10
    assert alignments == {"both", "one", "none"}
    assert len(split_shapes["train"]) == 9 and len(split_shapes["test"]) >= 12
    assert all(count > 0 for count in stats.values())
    assert stats["area_reversal"] >= 42 and stats["global_disagrees"] >= 84
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(731, 7, 3)) == _puzzle_key(generate_puzzle(731, 7, 3))

    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(12000 + 10 * train + test, train, test)
            assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    # A valid alternate palette must produce the same geometric answer.
    probe = generate_puzzle(91, 2, 1)["train"][0]
    permutation = np.array([6, 8, 1, 9, 0, 4, 2, 7, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[probe["input"]]), permutation[probe["output"]])
    # Shape/orientation and input dtype are not hidden generation conventions.
    for rotation in range(4):
        turned = np.rot90(probe["input"], rotation).astype(np.int64)
        before = turned.copy()
        assert np.array_equal(solve(turned), np.rot90(probe["output"], rotation))
        assert np.array_equal(turned, before)
    assert np.array_equal(solve(probe["input"][:, ::-1].tolist()), probe["output"][:, ::-1])
    return {"seeds": 42, "canvas_shapes": len(shapes), "logical_shapes": len(logical_shapes),
            "voting_shapes": len(voting_shapes), "scale_profiles": len(profiles),
            "palettes": len(palettes), "structures": len(structures),
            "alignment_styles": sorted(alignments),
            "statistics": {key: int(value) for key, value in stats.items()}}
