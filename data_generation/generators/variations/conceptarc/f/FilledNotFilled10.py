"""Complete the visible checkerboard phase inside one stepped outline."""

import hashlib
import numpy as np


_SOURCE_DIGESTS = {
    "dde7e440573cecced4f606f70633ae74e623efe4c62663750079e6e305028e83",
    "3dc0eaa25f4b0c3ec4d707a5338b8ba668462a1c187e8f49ab425a4bcd6e58dc",
    "f6978585ab4fa9c5ba73ad336aa91d07f692e7037fe786abab1b92dcb193a773",
    "4b6f1e1a71a48f3c107577ff76276e0881723a4e5955fd7b74e6daaf8bc5cefb",
    "0d9962005c948bebd041cf76e0d91c66875af81976b0330e686c153c2a38a2ba",
    "e13ca6da8a9c625b9a38bea5cef1b6677ac6977bbabe66d95b1cbe223246ceed",
    "cad9cde8b4b7c3d4eaca993542595b334d470b594d466760530b0ba2da8f0301",
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("invalid pair count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return int(num_train), int(num_test)


def _profile(rng, height, amplitude):
    if amplitude == 0:
        return [0] * height
    start = int(rng.integers(0, height - 2 * amplitude))
    end = int(rng.integers(start + 2 * amplitude, height))
    return [
        min(amplitude, max(0, row - start), max(0, end - row))
        for row in range(height)
    ]


def _geometry(rng):
    """Return row endpoints in a canonical horizontal-cap orientation."""
    for _ in range(80):
        height = int(rng.integers(4, 19))
        thin = bool(rng.random() < 0.16)
        if thin:
            height = int(rng.integers(8, 19))
            width = int(rng.integers(2, 4))
        else:
            width = int(rng.integers(4, 18))
        maximum = min(5, (height - 1) // 2)
        if thin:
            left_sign = -1
            right_sign = int(rng.choice((0, 1)))
            left_amp = int(rng.integers(2, maximum + 1))
            right_amp = (
                int(rng.integers(0, min(3, maximum) + 1))
                if right_sign else 0
            )
        else:
            left_sign = int(rng.choice((-1, 0, 1)))
            right_sign = int(rng.choice((-1, 0, 1)))
            left_amp = int(rng.integers(1, maximum + 1)) if left_sign else 0
            right_amp = int(rng.integers(1, maximum + 1)) if right_sign else 0
        left_profile = (
            [min(left_amp, row, height - 1 - row) for row in range(height)]
            if thin else _profile(rng, height, left_amp)
        )
        right_profile = _profile(rng, height, right_amp)
        lefts = [left_sign * value for value in left_profile]
        rights = [width - 1 + right_sign * value for value in right_profile]
        widths = [right - left + 1 for left, right in zip(lefts, rights)]
        if (
            min(widths) < 2
            or min(widths[1:-1]) < 3
            or max(widths) > 24
        ):
            continue
        span = max(rights) - min(lefts) + 1
        if span <= 24:
            return height, lefts, rights
    raise RuntimeError("could not construct outline")


def _margins(rng, length):
    capacity = min(12, 30 - length)
    total = int(rng.integers(0, capacity + 1))
    first_low = max(0, total - 7)
    first_high = min(7, total)
    first = int(rng.integers(first_low, first_high + 1))
    return first, total - first


def _choose_mask(rng, kind, cells):
    cells = set(cells)
    rows = sorted({row for row, _ in cells})
    cols = sorted({col for _, col in cells})
    if kind == "none":
        return set()
    if kind == "single_cue":
        cue = tuple(rng.choice(sorted(cells)))
        return cells - {cue}
    if kind == "isolated":
        shuffled = sorted(cells)
        rng.shuffle(shuffled)
        selected = []
        limit = int(rng.integers(1, min(6, len(cells) - 1) + 1))
        for cell in shuffled:
            if all(
                max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) >= 3
                for other in selected
            ):
                selected.append(cell)
                if len(selected) == limit:
                    break
        return set(selected)
    if kind in ("row", "band"):
        if kind == "row":
            length = int(rng.integers(1, min(3, len(rows)) + 1))
        else:
            length = (
                int(rng.integers(3, min(9, len(rows)) + 1))
                if len(rows) >= 3 else len(rows)
            )
        start = int(rng.integers(0, len(rows) - length + 1))
        chosen_rows = set(rows[start:start + length])
        selected = {cell for cell in cells if cell[0] in chosen_rows}
        if kind == "row" and rng.random() < 0.48:
            subset = sorted({col for _, col in selected})
            width = int(rng.integers(1, len(subset) + 1))
            left = int(rng.integers(0, len(subset) - width + 1))
            chosen_cols = set(subset[left:left + width])
            selected = {cell for cell in selected if cell[1] in chosen_cols}
        return selected
    if kind == "column":
        width = int(rng.integers(1, min(3, len(cols)) + 1))
        start = int(rng.integers(0, len(cols) - width + 1))
        chosen_cols = set(cols[start:start + width])
        selected = {cell for cell in cells if cell[1] in chosen_cols}
        if rng.random() < 0.55:
            run = int(rng.integers(2, len(rows) + 1))
            first = int(rng.integers(0, len(rows) - run + 1))
            chosen_rows = set(rows[first:first + run])
            selected = {cell for cell in selected if cell[0] in chosen_rows}
        return selected
    if kind == "zigzag":
        side = int(rng.integers(0, 2))
        run = (
            len(rows) if rng.random() < 0.52
            else int(rng.integers(2, len(rows) + 1))
        )
        first = int(rng.integers(0, len(rows) - run + 1))
        selected = set()
        for row in rows[first:first + run]:
            row_cols = [col for r, col in cells if r == row]
            selected.add((row, min(row_cols) if side == 0 else max(row_cols)))
        return selected
    if kind == "patch":
        height = int(rng.integers(2, min(8, len(rows)) + 1))
        width = int(rng.integers(2, min(8, len(cols)) + 1))
        r0 = int(rng.integers(0, len(rows) - height + 1))
        c0 = int(rng.integers(0, len(cols) - width + 1))
        chosen_rows = set(rows[r0:r0 + height])
        chosen_cols = set(cols[c0:c0 + width])
        return {
            cell for cell in cells
            if cell[0] in chosen_rows and cell[1] in chosen_cols
        }
    if kind == "mixed":
        first, second = rng.choice(
            ("row", "column", "zigzag", "isolated", "patch"),
            size=2, replace=False,
        )
        return (
            _choose_mask(rng, str(first), cells)
            | _choose_mask(rng, str(second), cells)
        )
    raise ValueError("unknown mask")


def _mask_kind(rng, regime):
    kinds = (
        "none", "row", "band", "column", "zigzag", "isolated",
        "patch", "mixed", "single_cue",
    )
    weights = (
        (12, 15, 14, 12, 14, 11, 12, 8, 2)
        if regime == "source_train"
        else (10, 10, 20, 9, 12, 9, 12, 10, 8)
    )
    return str(rng.choice(kinds, p=np.array(weights) / sum(weights)))


def _make_pair(rng, colors, kind):
    background, border, accent = colors
    for _ in range(80):
        height, lefts, rights = _geometry(rng)
        object_width = max(rights) - min(lefts) + 1
        top, bottom = _margins(rng, height)
        left_pad, right_pad = _margins(rng, object_width)
        if top + bottom + left_pad + right_pad == 0:
            right_pad = 1
        shifted_lefts = [x - min(lefts) + left_pad for x in lefts]
        shifted_rights = [x - min(lefts) + left_pad for x in rights]
        output = np.full(
            (top + height + bottom, left_pad + object_width + right_pad),
            background, dtype=np.int8,
        )
        phase = int(rng.integers(0, 2))
        cells = set()
        for i in range(height):
            row = top + i
            left, right = shifted_lefts[i], shifted_rights[i]
            if i in (0, height - 1):
                output[row, left:right + 1] = border
            else:
                output[row, left] = border
                output[row, right] = border
                for col in range(left + 1, right):
                    if (row + col) % 2 == phase:
                        output[row, col] = accent
                        cells.add((row, col))
        if len(cells) < 5:
            continue
        erase = _choose_mask(rng, kind, cells)
        if kind != "none" and not erase:
            erase = {tuple(rng.choice(sorted(cells)))}
        if len(erase) == len(cells):
            erase.remove(tuple(rng.choice(sorted(erase))))
        input_grid = output.copy()
        for row, col in erase:
            input_grid[row, col] = background
        turns = int(rng.integers(0, 4))
        input_grid = np.rot90(input_grid, turns)
        output = np.rot90(output, turns)
        if rng.random() < 0.5:
            input_grid = np.fliplr(input_grid)
            output = np.fliplr(output)
        return {"input": input_grid.copy(), "output": output.copy()}
    raise RuntimeError("could not construct accented interior")


def _inspect_orientation(inp, out, background, border, accent):
    locations = np.argwhere(out != background)
    if not len(locations):
        return None
    top, bottom = int(locations[:, 0].min()), int(locations[:, 0].max())
    if (
        bottom - top + 1 < 4
        or np.any(out[:top] != background)
        or np.any(out[bottom + 1:] != background)
    ):
        return None
    lefts, rights = [], []
    for row in range(top, bottom + 1):
        occupied = np.flatnonzero(out[row] != background)
        if not len(occupied):
            return None
        left, right = int(occupied[0]), int(occupied[-1])
        if (
            right - left + 1 < 2
            or (row not in (top, bottom) and right - left + 1 < 3)
            or np.any(out[row, :left] != background)
            or np.any(out[row, right + 1:] != background)
        ):
            return None
        if row in (top, bottom):
            if np.any(out[row, left:right + 1] != border):
                return None
        elif (
            out[row, left] != border
            or out[row, right] != border
            or np.any(out[row, left + 1:right] == border)
        ):
            return None
        lefts.append(left)
        rights.append(right)
    if (
        any(abs(a - b) > 1 for a, b in zip(lefts, lefts[1:]))
        or any(abs(a - b) > 1 for a, b in zip(rights, rights[1:]))
    ):
        return None
    actual = {(int(row), int(col)) for row, col in np.argwhere(out == accent)}
    if not actual or len({(row + col) % 2 for row, col in actual}) != 1:
        return None
    row, col = next(iter(actual))
    phase = (row + col) % 2
    expected = {
        (row, col)
        for row in range(top + 1, bottom)
        for col in range(lefts[row - top] + 1, rights[row - top])
        if (row + col) % 2 == phase
    }
    if actual != expected:
        return None
    if not np.all((inp == out) | ((out == accent) & (inp == background))):
        return None
    survivors = {(int(row), int(col)) for row, col in np.argwhere(inp == accent)}
    if not survivors:
        return None
    return {
        "colors": (background, border, accent),
        "phase": phase,
        "lefts": tuple(lefts),
        "rights": tuple(rights),
        "top": top,
        "bottom": bottom,
        "accent_count": len(actual),
        "survivors": len(survivors),
        "erased": expected - survivors,
        "margins": (
            top, out.shape[0] - bottom - 1,
            min(lefts), out.shape[1] - max(rights) - 1,
        ),
    }


def _analyze_pair(inp, out):
    colors = [int(c) for c in np.unique(out)]
    assert len(colors) == 3
    for turns in range(4):
        rotated_input = np.rot90(inp, turns)
        rotated_output = np.rot90(out, turns)
        for background in colors:
            for border in colors:
                if border == background:
                    continue
                accent = next(c for c in colors if c not in (background, border))
                info = _inspect_orientation(
                    rotated_input, rotated_output, background, border, accent
                )
                if info is not None:
                    info["turns"] = turns
                    return info
    raise AssertionError("pair is not one recoverable outlined checkerboard")


def _input_candidates(inp):
    """Recover outputs from input geometry and surviving accents alone."""
    colors = [int(c) for c in np.unique(inp)]
    if len(colors) != 3:
        return []
    predictions = []
    for turns in range(4):
        grid = np.rot90(inp, turns)
        for background in colors:
            occupied = np.argwhere(grid != background)
            if not len(occupied):
                continue
            top = int(occupied[:, 0].min())
            bottom = int(occupied[:, 0].max())
            if bottom - top + 1 < 4:
                continue
            for border in colors:
                if border == background:
                    continue
                accent = next(
                    color for color in colors
                    if color not in (background, border)
                )
                lefts, rights = [], []
                valid = True
                for row in range(top, bottom + 1):
                    nonbackground = np.flatnonzero(grid[row] != background)
                    if len(nonbackground) < 2:
                        valid = False
                        break
                    left, right = int(nonbackground[0]), int(nonbackground[-1])
                    if row in (top, bottom):
                        valid = bool(np.all(grid[row, left:right + 1] == border))
                    else:
                        valid = bool(
                            right - left + 1 >= 3
                            and grid[row, left] == border
                            and grid[row, right] == border
                            and not np.any(grid[row, left + 1:right] == border)
                        )
                    if not valid:
                        break
                    lefts.append(left)
                    rights.append(right)
                if not valid or any(
                    abs(a - b) > 1 for a, b in zip(lefts, lefts[1:])
                ) or any(
                    abs(a - b) > 1 for a, b in zip(rights, rights[1:])
                ):
                    continue
                visible = np.argwhere(grid == accent)
                if not len(visible):
                    continue
                phases = {
                    (int(row) + int(col)) % 2 for row, col in visible
                }
                if len(phases) != 1 or any(
                    not lefts[int(row) - top] < col < rights[int(row) - top]
                    for row, col in visible
                ):
                    continue
                phase = phases.pop()
                result = grid.copy()
                for row in range(top + 1, bottom):
                    for col in range(lefts[row - top] + 1, rights[row - top]):
                        if (row + col) % 2 == phase:
                            result[row, col] = accent
                result = np.rot90(result, -turns)
                if not any(np.array_equal(result, old) for old in predictions):
                    predictions.append(result)
    return predictions


def _pair_digest(inp, out):
    a = inp.astype(np.uint8, copy=False)
    b = out.astype(np.uint8, copy=False)
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + b.tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, infos = set(), set(), {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (key, out.astype(np.uint8).tobytes())
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
            infos[regime].append(_analyze_pair(inp, out))
            predictions = _input_candidates(inp)
            assert len(predictions) == 1
            assert np.array_equal(predictions[0], out)
    assert any(info["erased"] for info in infos["train"])
    assert all(
        info["survivors"] >= 1
        for group in infos.values()
        for info in group
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    stable_colors = bool(rng.random() < 0.5)
    shared_colors = (
        (0, 3, 4) if rng.random() < 0.12
        else tuple(map(int, rng.choice(10, size=3, replace=False)))
    )
    puzzle = {"train": [], "test": []}
    seen = set()
    anchor = int(rng.integers(0, num_train))
    for regime, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = (
                str(rng.choice(("row", "band", "zigzag", "patch", "mixed")))
                if regime == "train" and i == anchor
                else _mask_kind(
                    rng, "source_train" if regime == "train" else "target_test"
                )
            )
            colors = (
                shared_colors if stable_colors
                else (0, 3, 4) if rng.random() < 0.07
                else tuple(map(int, rng.choice(10, size=3, replace=False)))
            )
            for _ in range(100):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, np.iinfo(np.int64).max))
                )
                pair = _make_pair(pair_rng, colors, kind)
                key = (pair["input"].shape, pair["input"].tobytes())
                predictions = _input_candidates(pair["input"])
                if (
                    key not in seen
                    and _pair_digest(pair["input"], pair["output"])
                    not in _SOURCE_DIGESTS
                    and len(predictions) == 1
                    and np.array_equal(predictions[0], pair["output"])
                ):
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct pair")
        rng.shuffle(puzzle[regime])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(
        (
            regime,
            tuple(
                (
                    pair["input"].shape,
                    pair["input"].tobytes(),
                    pair["output"].tobytes(),
                )
                for pair in puzzle[regime]
            ),
        )
        for regime in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, dimensions, palettes = set(), set(), set(), set()
    profiles, gap_sizes, survivor_counts, turns = set(), set(), set(), set()
    complete_train = complete_test = single_test = zero_margin = 0
    stable = reassigned = 0
    signatures = set()
    split_masks = {"train": set(), "test": set()}
    split_contours = {"train": set(), "test": set()}
    first_masks = {"train": set(), "test": set()}
    gap_locations = {"upper": 0, "middle": 0, "lower": 0}
    thin_caps = short_wide = wide_margin = source_palette = 0
    zigzag_with_column_change = 0
    phases = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palette_set = set()
        for regime in ("train", "test"):
            for index, pair in enumerate(puzzle[regime]):
                info = _analyze_pair(pair["input"], pair["output"])
                palette_set.add(info["colors"])
                palettes.add(info["colors"])
                phases.add(info["phase"])
                source_palette += int(info["colors"] == (0, 3, 4))
                dimensions.add(pair["input"].shape)
                profiles.add((info["lefts"], info["rights"]))
                gap_sizes.add(len(info["erased"]))
                survivor_counts.add(info["survivors"])
                turns.add(info["turns"])
                zero_margin += int(0 in info["margins"])
                wide_margin += int(max(info["margins"]) >= 5)
                row_widths = [
                    right - left + 1
                    for left, right in zip(info["lefts"], info["rights"])
                ]
                thin_caps += int(min(row_widths) == 2)
                short_wide += int(
                    len(row_widths) <= 5 and max(row_widths) >= 10
                )
                left_varies = len(set(info["lefts"])) > 1
                right_varies = len(set(info["rights"])) > 1
                contour = (
                    "both" if left_varies and right_varies
                    else "left" if left_varies
                    else "right" if right_varies
                    else "rectangle"
                )
                split_contours[regime].add(contour)
                erased = info["erased"]
                rows = {row for row, _ in erased}
                cols = {col for _, col in erased}
                if not erased:
                    mask = "complete"
                elif info["survivors"] == 1:
                    mask = "single_cue"
                elif len(rows) >= 3 and all(
                    {(row, col) for r, col in erased if r == row}
                    == {
                        (row, col)
                        for col in range(
                            info["lefts"][row - info["top"]] + 1,
                            info["rights"][row - info["top"]],
                        )
                        if (row + col) % 2 == info["phase"]
                    }
                    for row in rows
                ):
                    mask = "band"
                elif len(cols) == 1:
                    mask = "column"
                elif len(rows) >= 4 and all(
                    sum(r == row for r, _ in erased) == 1 for row in rows
                ):
                    mask = "zigzag"
                    zigzag_with_column_change += int(len(cols) >= 2)
                elif len(erased) <= 3:
                    mask = "scatter"
                else:
                    mask = "patch_or_mixed"
                split_masks[regime].add(mask)
                if index == 0:
                    first_masks[regime].add(mask)
                if erased:
                    interior = max(1, info["bottom"] - info["top"] - 1)
                    midpoint = (min(rows) + max(rows)) / 2 - info["top"]
                    location = (
                        "upper" if midpoint / interior < 1 / 3
                        else "lower" if midpoint / interior > 2 / 3
                        else "middle"
                    )
                    gap_locations[location] += 1
                if regime == "train":
                    complete_train += int(not info["erased"])
                else:
                    complete_test += int(not info["erased"])
                    single_test += int(info["survivors"] == 1)
        stable += int(len(palette_set) == 1)
        reassigned += int(len(palette_set) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(dimensions) >= 35
    assert len(profiles) >= 70 and len(palettes) >= 40
    assert len(gap_sizes) >= 12 and len(survivor_counts) >= 15
    assert complete_train and complete_test and single_test and zero_margin
    assert stable and reassigned and len(turns) >= 2
    assert phases == {0, 1}
    assert thin_caps >= 5 and short_wide >= 5 and wide_margin >= 20
    assert source_palette and zigzag_with_column_change
    assert all(gap_locations.values())
    assert all(
        {"complete", "band", "column", "zigzag", "scatter", "single_cue"}
        <= split_masks[regime]
        for regime in ("train", "test")
    )
    assert all(
        {"rectangle", "left", "right", "both"}
        <= split_contours[regime]
        for regime in ("train", "test")
    )
    assert len(first_masks["train"]) >= 4
    assert len(first_masks["test"]) >= 4
    for seed, train, test in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4),
        (104, None, 1), (105, None, 4),
        (106, 2, None), (107, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
