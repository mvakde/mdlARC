"""Vertical bands move as complete shapes into ascending width order."""

import hashlib

import numpy as np


SOURCE_DIGESTS = frozenset((
    "2b3d0e0495bd5e1ef34142736a7161fb039aaf3296e4053709407c2423d9be33",
    "b71fdffc416da4c11f663f00daa9d28ab16a9e80fb6f46211f93f57a711766c8",
    "52879327791d9d6279676e3a293297d0a1847932327f83a69834d368adc52879",
    "925201c5cc6c09c70a3aef1b44c99dbb1c6da9fc094c16ad71b48dce17bd0d64",
    "81a41926cb7840cd1f76aba421688ffb0b53f95344565815219105c215db3e9e",
))


def _choose_counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _digest(inp, out):
    return hashlib.sha256(repr(inp.shape).encode() +
                          inp.astype(np.int8).tobytes() +
                          out.astype(np.int8).tobytes()).hexdigest()


def _widths(rng, canvas, count, special):
    if special == "anchor":
        small = int(rng.integers(1, canvas))
        pair = (small, small + 1)
    elif special == "equal":
        scale = int(rng.integers(1, canvas // 3 + 1))
        pair = (2 * scale, 3 * scale)
    else:
        return sorted(int(x) for x in rng.choice(
            np.arange(1, canvas + 1), size=count, replace=False)), None
    remaining = [x for x in range(1, canvas + 1) if x not in pair]
    extra = [int(x) for x in rng.choice(remaining, size=count - 2, replace=False)]
    return sorted(list(pair) + extra), pair


def _kinds(rng, widths, profile, forced):
    kinds = []
    for width in widths:
        options = ("solid", "hollow", "sparse") if width >= 3 else ("solid", "sparse")
        weights = (0.53, 0.25, 0.22) if width >= 3 else (0.7, 0.3)
        kinds.append(str(rng.choice(options, p=weights)))
    if profile in ("solid", "hollow", "sparse"):
        kinds = [profile] * len(widths)
    elif profile == "mixed":
        eligible = [i for i, w in enumerate(widths) if w >= 3 and i not in forced]
        if not eligible:
            # The equal-area pair occupies two widths; other widths can be 1 and 2.
            eligible = [i for i, w in enumerate(widths) if w >= 3]
        hollow = int(rng.choice(eligible))
        candidates = [i for i in range(len(widths)) if i not in forced and i != hollow]
        sparse = int(rng.choice(candidates))
        kinds[hollow], kinds[sparse] = "hollow", "sparse"
    for i in forced:
        kinds[i] = "solid"
    for i, width in enumerate(widths):
        if kinds[i] == "hollow" and width < 3:
            kinds[i] = "sparse"
    return kinds


def _heights(rng, kinds, leading, regime, special, forced, same_height, compact):
    count = len(kinds)
    if same_height:
        return [int(rng.integers(4, 6))] * count
    low = [{"solid": 1, "hollow": 3, "sparse": 4}[k] for k in kinds]
    high = [{"solid": 8, "hollow": 9, "sparse": 10}[k] +
            (regime == "target_test") for k in kinds]
    if compact:
        high = [min(cap, {"solid": 3, "hollow": 4, "sparse": 5}[k])
                for cap, k in zip(high, kinds)]
    values = [None] * count
    if special == "anchor":
        values[forced[0]], values[forced[1]] = 5, 1
    elif special == "equal":
        values[forced[0]], values[forced[1]] = 3, 2
    budget = 30 - leading - sum(v for v in values if v is not None)
    pending = [i for i, v in enumerate(values) if v is None]
    for i in rng.permutation(pending):
        i = int(i)
        reserve = sum(low[j] for j in pending if j != i)
        cap = min(high[i], budget - reserve)
        assert cap >= low[i]
        values[i] = int(rng.integers(low[i], cap + 1))
        budget -= values[i]
        pending.remove(i)
    return values


def _band(rng, color, bg, width, height, canvas, kind):
    band = np.full((height, canvas), bg, dtype=np.int8)
    if kind == "solid":
        band[:, :width] = color
    elif kind == "hollow":
        band[:, 0] = color
        interior_tip = bool(rng.integers(3) == 0)
        if interior_tip:
            band[int(rng.integers(1, height - 1)), width - 1] = color
            band[0, 1:width - 1] = color
            band[-1, 1:width - 1] = color
        else:
            band[0, :width] = color
            band[-1, :width] = color
            for row in range(1, height - 1):
                if rng.random() < 0.8:
                    band[row, width - 1] = color
        density = float(rng.uniform(0.04, 0.38))
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if rng.random() < density:
                    band[row, col] = color
        band[1, 1] = bg
    elif kind == "sparse":
        band[0, 0] = band[-1, 0] = color
        interior_tip = width > 1 and bool(rng.integers(2))
        blank_length = int(rng.integers(1, height - (2 if interior_tip else 1)))
        blank_start = int(rng.integers(1, height - blank_length))
        blanks = set(range(blank_start, blank_start + blank_length))
        last = width - 2 if interior_tip else width - 1
        edge_density = float(rng.uniform(0.15, 0.7))
        for row in (0, height - 1):
            for col in range(1, last + 1):
                if rng.random() < edge_density:
                    band[row, col] = color
        if not interior_tip:
            band[int(rng.choice((0, height - 1))), width - 1] = color
        density = float(rng.uniform(0.08, 0.42))
        for row in range(1, height - 1):
            if row in blanks:
                continue
            if rng.random() < 0.75:
                band[row, 0] = color
            for col in range(1, width - 1 if interior_tip else width):
                if rng.random() < density:
                    band[row, col] = color
        if interior_tip:
            rows = [r for r in range(1, height - 1) if r not in blanks]
            band[int(rng.choice(rows)), width - 1] = color
    else:
        raise AssertionError("unknown band kind")
    return band


def _make_pair(rng, regime, profile, special=None, avoid_colors=None):
    count = int(rng.integers(3, 6))
    if special == "equal" and profile == "mixed":
        count = 5
    canvas = int(rng.integers(6, 25 if regime == "source_train" else 31))
    if profile == "hollow":
        canvas = max(canvas, count + 2)
    widths, pair_widths = _widths(rng, canvas, count, special)
    if profile == "hollow" and min(widths) < 3:
        widths = sorted(int(x) for x in rng.choice(
            np.arange(3, canvas + 1), size=count, replace=False))
    forced = () if pair_widths is None else tuple(widths.index(w) for w in pair_widths)
    kinds = _kinds(rng, widths, profile, forced)
    leading = int(rng.integers(0, 4))
    same_height = special is None and regime == "source_train" and rng.random() < 0.16
    compact = special is None and rng.random() < 0.23
    heights = _heights(rng, kinds, leading, regime, special, forced, same_height, compact)
    bg = 0 if rng.random() < 0.78 else int(rng.integers(1, 10))
    colors = [int(x) for x in rng.choice(
        [c for c in range(1, 10) if c != bg], size=count, replace=False)]
    if avoid_colors is not None and set(colors) == avoid_colors:
        colors[-1] = next(c for c in range(1, 10) if c != bg and c not in colors)
    bands = [_band(rng, colors[i], bg, widths[i], heights[i], canvas, kinds[i])
             for i in range(count)]
    order = [int(i) for i in rng.permutation(count)]
    if order == list(range(count)):
        order[0], order[1] = order[1], order[0]
    prefix = [np.full((leading, canvas), bg, dtype=np.int8)] if leading else []
    return {"input": np.vstack(prefix + [bands[i] for i in order]),
            "output": np.vstack(prefix + bands)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    source_like = bool(rng.random() < 0.36)
    anchor = int(rng.integers(num_train))
    others = [i for i in range(num_train) if i != anchor]
    coverage = None if source_like else int(rng.choice(others))
    equal = int(rng.choice(others)) if rng.random() < 0.38 else None
    bridge = int(rng.integers(num_test)) if source_like else None
    puzzle = {"train": [], "test": []}
    seen = set()
    first_colors = None
    for regime, count, split in (("source_train", num_train, "train"),
                                 ("target_test", num_test, "test")):
        for index in range(count):
            special = ("anchor" if split == "train" and index == anchor
                       else "equal" if split == "train" and index == equal
                       else None)
            if split == "train":
                profile = "solid" if source_like else (
                    "mixed" if index == coverage else "random")
            elif index == bridge:
                profile = "mixed"
            else:
                profile = str(rng.choice(
                    ("solid", "hollow", "sparse", "mixed"),
                    p=(0.18, 0.16, 0.16, 0.50)))
            for _ in range(8):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _make_pair(pair_rng, regime, profile, special,
                                  first_colors if split == "train" and index == 1 else None)
                inp, out = pair["input"], pair["output"]
                signature = (inp.shape, inp.tobytes())
                if signature not in seen and _digest(inp, out) not in SOURCE_DIGESTS:
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen.add(signature)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                bg, _, bands = _parse_bands(inp)
                first_colors = {b[2] for b in bands}
    validate_generated_puzzle(puzzle)
    return puzzle


def _parse_bands(grid):
    valid = []
    values = [int(x) for x in np.unique(grid)]
    for bg in values:
        colors = [c for c in values if c != bg]
        if not 3 <= len(colors) <= 5:
            continue
        bands = []
        for color in colors:
            rows, cols = np.where(grid == color)
            top, bottom = int(rows.min()), int(rows.max()) + 1
            width = int(cols.max()) + 1
            band = grid[top:bottom].copy()
            if (not np.all((band == bg) | (band == color))
                    or not np.any(band[:, 0] == color)
                    or not np.all(band[:, width:] == bg)):
                break
            bands.append((top, bottom, color, width, band))
        else:
            bands.sort(key=lambda b: b[0])
            leading = bands[0][0]
            cursor = leading
            for top, bottom, _, _, _ in bands:
                if top != cursor:
                    break
                cursor = bottom
            else:
                if (cursor == grid.shape[0] and np.all(grid[:leading] == bg)
                        and len({b[3] for b in bands}) == len(bands)):
                    valid.append((bg, leading, bands))
    assert len(valid) == 1, "background and band spans must be unambiguous"
    return valid[0]


def _kind(band, color, bg, width):
    if np.all(band[:, :width] == color):
        return "solid"
    if np.any(np.all(band[1:-1] == bg, axis=1)):
        return "sparse"
    assert width >= 3 and band.shape[0] >= 3
    return "hollow"


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_kinds, test_kinds = set(), set()
    train_colors = []
    area_reversal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = (inp.shape, inp.tobytes())
            pair_signature = (signature, out.shape, out.tobytes())
            assert signature not in inputs and pair_signature not in pairs
            assert _digest(inp, out) not in SOURCE_DIGESTS
            inputs.add(signature)
            pairs.add(pair_signature)
            bg, leading, bands = _parse_bands(inp)
            assert 0 <= leading <= 3 and 3 <= len(bands) <= 5
            assert all(np.any(b[4][0] == b[2]) and np.any(b[4][-1] == b[2])
                       for b in bands)
            ordered = sorted(bands, key=lambda b: b[3])
            prefix = [np.full((leading, inp.shape[1]), bg, dtype=inp.dtype)] if leading else []
            assert np.array_equal(out, np.vstack(prefix + [b[4] for b in ordered]))
            kinds = {_kind(b[4], b[2], bg, b[3]) for b in bands}
            if split == "train":
                train_kinds.update(kinds)
                train_colors.append({b[2] for b in bands})
                for a, b in zip(ordered, ordered[1:]):
                    if np.count_nonzero(a[4] == a[2]) > np.count_nonzero(b[4] == b[2]):
                        area_reversal = True
            else:
                test_kinds.update(kinds)
    assert train_colors[0] != train_colors[1]
    assert area_reversal
    assert train_kinds == {"solid"} or {"hollow", "sparse"} <= train_kinds
    if train_kinds == {"solid"}:
        assert {"hollow", "sparse"} <= test_kinds


def _same_puzzle(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[split], b[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, structures, palettes = set(), set(), set()
    band_counts, backgrounds, leading_rows = set(), set(), set()
    pure_tests = set()
    intro = mixed = equal_area = 0
    narrow_hollow = narrow_sparse = wide_solid = interior_tip = 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_types = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, leading, bands = _parse_bands(inp)
                ordered = sorted(bands, key=lambda b: b[3])
                kinds = [_kind(b[4], b[2], bg, b[3]) for b in ordered]
                shapes.add(inp.shape)
                structures.add((inp.shape, leading, tuple(
                    (b[3], b[4].shape[0], kinds[i],
                     int(np.count_nonzero(b[4] == b[2])))
                    for i, b in enumerate(ordered))))
                palettes.add(tuple(sorted(int(c) for c in np.unique(inp))))
                band_counts.add((split, len(bands)))
                backgrounds.add(bg)
                leading_rows.add(leading)
                if split == "train":
                    train_types.update(kinds)
                    equal_area += any(
                        a[3] * a[4].shape[0] == b[3] * b[4].shape[0]
                        and kinds[i] == kinds[j] == "solid"
                        for i, a in enumerate(ordered)
                        for j, b in enumerate(ordered) if i < j)
                elif len(set(kinds)) == 1:
                    pure_tests.add(kinds[0])
                narrow_hollow += kinds[0] == "hollow"
                narrow_sparse += kinds[0] == "sparse"
                wide_solid += kinds[-1] == "solid"
                interior_tip += any(
                    b[3] > 1 and b[4].shape[0] > 2
                    and b[4][0, b[3] - 1] != b[2]
                    and b[4][-1, b[3] - 1] != b[2]
                    and np.any(b[4][1:-1, b[3] - 1] == b[2])
                    for b in bands)
        intro += train_types == {"solid"}
        mixed += {"hollow", "sparse"} <= train_types
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(structures) >= 150 and len(palettes) >= 25
    assert {("train", n) for n in (3, 4, 5)} <= band_counts
    assert {("test", n) for n in (3, 4, 5)} <= band_counts
    assert len(backgrounds) >= 3 and leading_rows == {0, 1, 2, 3}
    assert pure_tests == {"solid", "hollow", "sparse"}
    assert intro >= 5 and mixed >= 5 and equal_area >= 3
    assert min(narrow_hollow, narrow_sparse, wide_solid, interior_tip) >= 3
    assert sum(not _same_puzzle(puzzles[0], p) for p in puzzles[1:]) >= 40
    assert _same_puzzle(generate_puzzle(17), generate_puzzle(17))
    assert _same_puzzle(generate_puzzle(987, 5, 2), generate_puzzle(987, 5, 2))
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1234, tr, te)
        validate_generated_puzzle(puzzle)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
