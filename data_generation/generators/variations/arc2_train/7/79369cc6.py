"""Solved examples of completing transformed, partly concealed 3x3 stencils."""

import hashlib
import numpy as np


# Hashes of the four original input/output pairs, using int8 row-major bytes.
_SOURCE_PAIRS = {
    ((18, 16), "992ed759004e39afc1f7adffebf30d5780f22e07dedde0bc71b2eeb6e4719a50"),
    ((19, 12), "3f39531eddb053a8fe666893e2875cbf2bcf81475b452ac9cb108c916ec3a7d2"),
    ((17, 17), "05406164621e6ad30d9b7d8a659e0c0d531a83ff987dd313af4bd38c7ab1f7eb"),
    ((16, 15), "cacaace0fe3898db17970c0ad90781352f8295227fea67f6b2c0a6d13caf2ac5"),
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(seed, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)
    ):
        if value is not None and (not _integer(value) or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _transforms(mask):
    result = []
    for base in (mask, np.fliplr(mask)):
        for turns in range(4):
            candidate = np.rot90(base, turns).copy()
            if not any(np.array_equal(candidate, old) for old in result):
                result.append(candidate)
    return result


def _visible(mask, row, column, height, width):
    locator = []
    paint = []
    for y in range(3):
        for x in range(3):
            yy, xx = row + y, column + x
            if 0 <= yy < height and 0 <= xx < width:
                (locator if mask[y, x] else paint).append((yy, xx))
    return tuple(locator), tuple(paint)


def _clip_sides(mask):
    sides = []
    # On a 6x6 canvas, these are top, bottom, left and right edge clips.
    for side, (row, col) in enumerate(((-1, 1), (4, 1), (1, -1), (1, 4))):
        loc, paint = _visible(mask, row, col, 6, 6)
        if len(loc) >= 3 and len(paint) >= 2:
            sides.append(side)
    return sides


def _intrinsic_ambiguity(mask):
    # Reject stencils for which a real copy necessarily creates a shifted
    # second match, even in a locator-free field.
    transforms = _transforms(mask)
    for actual in transforms:
        actual_loc = {(y, x) for y in range(3) for x in range(3) if actual[y, x]}
        for trial in transforms:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == dc == 0:
                        continue
                    trial_loc = {(dr + y, dc + x) for y in range(3) for x in range(3) if trial[y, x]}
                    trial_paint = {(dr + y, dc + x) for y in range(3) for x in range(3) if not trial[y, x]}
                    if trial_loc <= actual_loc and not trial_paint & actual_loc:
                        return True
    return False


def _stencil_bank():
    bank = {3: [], 4: [], 5: []}
    for bits in range(512):
        mask = np.array([(bits >> bit) & 1 for bit in range(9)], dtype=bool).reshape(3, 3)
        count = int(mask.sum())
        if count not in bank:
            continue
        transforms = _transforms(mask)
        if len(transforms) < 4 or _intrinsic_ambiguity(mask):
            continue
        if not any(_clip_sides(transform) for transform in transforms):
            continue
        bank[count].append(mask)
    return bank


_BANK = _stencil_bank()


def _sample_stencil(rng, count=None, different_from=None, asymmetric=False):
    if count is None:
        count = int(rng.choice([3, 4, 5]))
    options = [mask for mask in _BANK[count]
               if not asymmetric or len(_transforms(mask)) == 8]
    for _ in range(30):
        mask = options[int(rng.integers(len(options)))].copy()
        if different_from is None or not np.array_equal(mask, different_from):
            return mask
    return next(mask.copy() for mask in options if not np.array_equal(mask, different_from))


def _extract_reference(grid):
    """Identify the sole complete two-color patch containing all paint cells."""
    height, width = grid.shape
    matches = []
    values, counts = np.unique(grid, return_counts=True)
    for value, count in zip(values, counts):
        if not 4 <= count <= 6:
            continue
        positions = np.argwhere(grid == value)
        for row in range(max(0, int(positions[:, 0].max()) - 2), min(height - 3, int(positions[:, 0].min())) + 1):
            for col in range(max(0, int(positions[:, 1].max()) - 2), min(width - 3, int(positions[:, 1].min())) + 1):
                patch = grid[row:row + 3, col:col + 3]
                colors = np.unique(patch)
                if len(colors) != 2 or value not in colors:
                    continue
                locator = int(colors[0] if colors[1] == value else colors[1])
                mask = patch == locator
                if 3 <= int(mask.sum()) <= 5:
                    matches.append((int(value), locator, mask.copy(), (row, col)))
    if len(matches) != 1:
        raise AssertionError("input needs one identifiable complete reference")
    return matches[0]


def _candidates(grid, paint, locator, stencil):
    height, width = grid.shape
    found = {}
    transforms = _transforms(stencil)
    for row in range(-1, height - 1):
        for col in range(-1, width - 1):
            # A corner clip has only four visible cells and is too weak here.
            if (row < 0 or row > height - 3) and (col < 0 or col > width - 3):
                continue
            for orientation, mask in enumerate(transforms):
                loc, fill = _visible(mask, row, col, height, width)
                if len(loc) < 3 or len(fill) < 2:
                    continue
                if all(grid[y, x] == locator for y, x in loc) and all(
                    grid[y, x] != locator and grid[y, x] != paint for y, x in fill
                ):
                    found[(row, col, fill)] = (row, col, orientation, fill, loc)
    return list(found.values())


def _separated(first, second, gap=0):
    a, b = first
    c, d = second
    return a + 2 + gap < c or c + 2 + gap < a or b + 2 + gap < d or d + 2 + gap < b


def _palette(rng, base):
    # Base roles are shared in some pairs; other pairs visibly redefine them.
    if rng.random() < 0.48:
        return base
    length = int(rng.integers(4, 8))
    return tuple(map(int, rng.choice(10, size=length, replace=False)))


def _shape(rng):
    return int(rng.integers(12, 27)), int(rng.integers(12, 27))


def _reference_start(rng, height, width):
    if rng.random() < 0.42:
        side = int(rng.integers(4))
        if side == 0:
            return 0, int(rng.integers(width - 2))
        if side == 1:
            return height - 3, int(rng.integers(width - 2))
        if side == 2:
            return int(rng.integers(height - 2)), 0
        return int(rng.integers(height - 2)), width - 3
    return int(rng.integers(height - 2)), int(rng.integers(width - 2))


def _starts(rng, height, width, mask, clipped):
    if not clipped:
        positions = [(y, x) for y in range(height - 2) for x in range(width - 2)]
    else:
        sides = _clip_sides(mask)
        positions = []
        if 0 in sides:
            positions += [(-1, x) for x in range(width - 2)]
        if 1 in sides:
            positions += [(height - 2, x) for x in range(width - 2)]
        if 2 in sides:
            positions += [(y, -1) for y in range(height - 2)]
        if 3 in sides:
            positions += [(y, width - 2) for y in range(height - 2)]
    rng.shuffle(positions)
    return positions


def _plan(rng, regime, index, total_train, transforms, source_like, teaching_role):
    if regime == "source_train":
        count = int(rng.choice([1, 2, 3], p=[0.48, 0.42, 0.10]))
        clip_rate = 0.22
    else:
        count = int(rng.choice([1, 2, 3, 4, 5], p=[0.10, 0.24, 0.40, 0.21, 0.05]))
        clip_rate = 0.20
    if source_like and regime == "source_train":
        if index == 0:
            count = 1
            clip_rate = 0
        elif index == 1:
            count = 2
            clip_rate = 0.5 if total_train == 2 else 0
        elif index == 2:
            count = 1
            clip_rate = 1
    # Sampling is with replacement: repeated orientations are intentional.
    orientations = [int(rng.integers(len(transforms))) for _ in range(count)]
    clipped = [bool(rng.random() < clip_rate) for _ in range(count)]
    if source_like and regime == "source_train" and index == 2:
        possible = [i for i, mask in enumerate(transforms) if _clip_sides(mask)]
        orientations[0] = possible[int(rng.integers(len(possible)))]
        clipped[0] = True
    if source_like and regime == "source_train" and index == 1 and total_train == 2:
        possible = [i for i, mask in enumerate(transforms) if _clip_sides(mask)]
        orientations[-1] = possible[int(rng.integers(len(possible)))]
        clipped[-1] = True
    if source_like and regime == "source_train" and index == 0 and len(transforms) > 1:
        orientations[0] = int(rng.integers(1, len(transforms)))
    if teaching_role == "rotation":
        orientations[0] = int(rng.integers(1, 4))
        clipped[0] = False
    elif teaching_role == "reflection":
        orientations[0] = int(rng.integers(4, 8))
        clipped[0] = False
    # Replace requested clips that the particular transformed mask cannot show.
    for j in range(count):
        if clipped[j] and not _clip_sides(transforms[orientations[j]]):
            possible = [i for i, mask in enumerate(transforms) if _clip_sides(mask)]
            if possible:
                orientations[j] = possible[int(rng.integers(len(possible)))]
            else:
                clipped[j] = False
    return list(zip(orientations, clipped))


def _make_pair(rng, regime, index, total_train, palette, stencil, source_like, teaching_role):
    paint, locator = palette[:2]
    field = np.asarray(palette[2:], dtype=np.int8)
    transforms = _transforms(stencil)
    plan = _plan(rng, regime, index, total_train, transforms, source_like, teaching_role)
    for _ in range(160):
        height, width = _shape(rng)
        ref = _reference_start(rng, height, width)
        occupied = [ref]
        placements = []
        for orientation, clipped in plan:
            for start in _starts(rng, height, width, transforms[orientation], clipped):
                if all(_separated(start, old, 1) for old in occupied):
                    occupied.append(start)
                    placements.append((start[0], start[1], orientation))
                    break
            else:
                break
        if len(placements) != len(plan):
            continue
        weights = rng.uniform(0.7, 1.5, len(field))
        weights /= weights.sum()
        grid = rng.choice(field, size=(height, width), p=weights).astype(np.int8)
        noise_rate = float(rng.uniform(0.025, 0.095))
        grid[rng.random((height, width)) < noise_rate] = locator
        for row, col, orientation in placements:
            loc, fill = _visible(transforms[orientation], row, col, height, width)
            for y, x in loc:
                grid[y, x] = locator
            for y, x in fill:
                grid[y, x] = field[int(rng.integers(len(field)))]
        # A one-error locator cluster makes an exact-mask near miss.
        if rng.random() < 0.55:
            for _ in range(30):
                miss_row = int(rng.integers(height - 2))
                miss_col = int(rng.integers(width - 2))
                if not all(_separated((miss_row, miss_col), old, 1) for old in occupied):
                    continue
                mask = transforms[int(rng.integers(len(transforms)))]
                loc, fill = _visible(mask, miss_row, miss_col, height, width)
                omitted = int(rng.integers(len(loc)))
                for k, (y, x) in enumerate(loc):
                    grid[y, x] = field[int(rng.integers(len(field)))] if k == omitted else locator
                for y, x in fill:
                    grid[y, x] = field[int(rng.integers(len(field)))]
                break
        rr, cc = ref
        grid[rr:rr + 3, cc:cc + 3] = np.where(stencil, locator, paint)
        if set(map(int, np.unique(grid))) != set(palette):
            continue
        try:
            found_paint, found_locator, found_mask, found_ref = _extract_reference(grid)
        except AssertionError:
            continue
        if found_paint != paint or found_locator != locator or found_ref != ref or not np.array_equal(found_mask, stencil):
            continue
        candidates = _candidates(grid, paint, locator, stencil)
        found = {(r, c, fill) for r, c, _, fill, _ in candidates}
        intended = set()
        for r, c, orientation in placements:
            _, fill = _visible(transforms[orientation], r, c, height, width)
            intended.add((r, c, fill))
        if found != intended:
            continue
        output = grid.copy()
        for _, _, _, fill, _ in candidates:
            for y, x in fill:
                output[y, x] = paint
        if np.array_equal(grid, output):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct an unambiguous pair")


def _pair_key(pair):
    return (pair["input"].shape,
            pair["input"].astype(np.int8, copy=False).tobytes(),
            pair["output"].astype(np.int8, copy=False).tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, size=5, replace=False)))
    source_like = bool(rng.random() < 0.28)
    if source_like:
        rotation_teacher, reflection_teacher = 0, 1
    else:
        rotation_teacher, reflection_teacher = map(
            int, rng.choice(num_train, size=2, replace=False)
        )
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=num_train + num_test)
    puzzle = {"train": [], "test": []}
    stencils = []
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[len(stencils)]))
            teaching_role = None
            if split == "train":
                if index == rotation_teacher:
                    teaching_role = "rotation"
                elif index == reflection_teacher:
                    teaching_role = "reflection"
            if not stencils:
                stencil = _sample_stencil(pair_rng, asymmetric=teaching_role is not None)
            elif len(stencils) == 1:
                stencil = _sample_stencil(pair_rng, different_from=stencils[0],
                                          asymmetric=teaching_role is not None)
            elif teaching_role is not None:
                stencil = _sample_stencil(pair_rng, asymmetric=True)
            elif pair_rng.random() < (0.34 if split == "train" else 0.40):
                stencil = stencils[int(pair_rng.integers(len(stencils)))].copy()
            else:
                stencil = _sample_stencil(pair_rng, different_from=stencils[int(pair_rng.integers(len(stencils)))])
                if split == "test" and pair_rng.random() < 0.55:
                    for _ in range(20):
                        if not any(np.array_equal(stencil, old) for old in stencils):
                            break
                        stencil = _sample_stencil(pair_rng)
            stencils.append(stencil)
            palette = base_palette if len(stencils) == 1 else _palette(pair_rng, base_palette)
            for _ in range(8):
                pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  index, num_train, palette, stencil, source_like, teaching_role)
                key = _pair_key(pair)
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique puzzle pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    grid, output = pair["input"], pair["output"]
    paint, locator, mask, ref = _extract_reference(grid)
    candidates = _candidates(grid, paint, locator, mask)
    assert 1 <= len(candidates) <= 5
    for i, candidate in enumerate(candidates):
        assert _separated(ref, candidate[:2])
        for other in candidates[i + 1:]:
            assert _separated(candidate[:2], other[:2])
    expected = grid.copy()
    for _, _, _, fill, _ in candidates:
        for y, x in fill:
            expected[y, x] = paint
    assert np.array_equal(expected, output)
    assert np.any(grid != output)
    assert set(map(int, np.unique(grid))) == set(map(int, np.unique(output)))
    return {"paint": paint, "locator": locator, "mask": mask, "ref": ref,
            "candidates": candidates, "shape": grid.shape, "palette": tuple(map(int, np.unique(grid)))}


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        train, test = puzzle["train"], puzzle["test"]
        assert isinstance(train, list) and isinstance(test, list)
        assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
        seen_inputs = set()
        seen_pairs = set()
        train_info = []
        for split, pairs in (("train", train), ("test", test)):
            for pair in pairs:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in (pair["input"], pair["output"]):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
                assert pair["input"].shape == pair["output"].shape
                key = _pair_key(pair)
                input_key = (pair["input"].shape,
                             pair["input"].astype(np.int8, copy=False).tobytes())
                assert input_key not in seen_inputs and key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(key)
                source_hash = hashlib.sha256(pair["input"].astype(np.int8).tobytes() +
                                             pair["output"].astype(np.int8).tobytes()).hexdigest()
                assert (pair["input"].shape, source_hash) not in _SOURCE_PAIRS
                info = _analyze(pair)
                if split == "train":
                    train_info.append(info)
        # The input-local rule is demonstrated with changed cells in every
        # training output, including at least one complete transformed copy.
        assert any(
            0 <= c[0] <= info["shape"][0] - 3 and 0 <= c[1] <= info["shape"][1] - 3
            for info in train_info for c in info["candidates"]
        )
        assert any(1 <= c[2] <= 3 and len(_transforms(info["mask"])) == 8
                   for info in train_info for c in info["candidates"])
        assert any(4 <= c[2] <= 7 and len(_transforms(info["mask"])) == 8
                   for info in train_info for c in info["candidates"])
        assert len({info["mask"].tobytes() for info in train_info}) >= 2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed or task-inconsistent puzzle") from exc


def _equal(first, second):
    return all(np.array_equal(a[name], b[name])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for name in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(11)
    assert _equal(a, generate_puzzle(11))
    assert not _equal(a, generate_puzzle(12))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4), (2, None),
                                            (10, None), (7, 3))):
        puzzle = generate_puzzle(500 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4), (2, 4)):
        try:
            generate_puzzle(900, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    train_counts, test_counts = set(), set()
    loc_counts, transform_counts, shapes, target_counts = set(), set(), set(), set()
    stencils, layouts = set(), set()
    repeated_in_puzzle = 0
    changed_in_puzzle = 0
    new_test_stencils = 0
    varied_palettes = 0
    varied_roles = 0
    smaller_tests = 0
    repeated_orientations = 0
    identity_pairs = 0
    clipped_sides = set()
    split_clip_sides = {"train": set(), "test": set()}
    split_target_counts = {"train": set(), "test": set()}
    split_shapes = {"train": set(), "test": set()}
    reference_sides = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        masks = []
        infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                infos[split].append(info)
                mask = info["mask"]
                masks.append(mask.tobytes())
                stencils.add(mask.tobytes())
                loc_counts.add(int(mask.sum()))
                transform_counts.add(len(_transforms(mask)))
                shapes.add(info["shape"])
                target_counts.add(len(info["candidates"]))
                split_target_counts[split].add(len(info["candidates"]))
                split_shapes[split].add(info["shape"])
                height, width = info["shape"]
                rr, cc = info["ref"]
                if rr == 0:
                    reference_sides.add("top")
                if rr == height - 3:
                    reference_sides.add("bottom")
                if cc == 0:
                    reference_sides.add("left")
                if cc == width - 3:
                    reference_sides.add("right")
                orientations = [candidate[2] for candidate in info["candidates"]]
                repeated_orientations += len(orientations) > len(set(orientations))
                identity_pairs += 0 in orientations
                for row, col, orientation, _, _ in info["candidates"]:
                    layouts.add((info["shape"], info["ref"], row, col, orientation))
                    if row == -1:
                        clipped_sides.add("top")
                        split_clip_sides[split].add("top")
                    if row == info["shape"][0] - 2:
                        clipped_sides.add("bottom")
                        split_clip_sides[split].add("bottom")
                    if col == -1:
                        clipped_sides.add("left")
                        split_clip_sides[split].add("left")
                    if col == info["shape"][1] - 2:
                        clipped_sides.add("right")
                        split_clip_sides[split].add("right")
        repeated_in_puzzle += len(set(masks)) < len(masks)
        changed_in_puzzle += len(set(masks)) > 1
        train_masks = {info["mask"].tobytes() for info in infos["train"]}
        new_test_stencils += any(info["mask"].tobytes() not in train_masks
                                 for info in infos["test"])
        all_infos = infos["train"] + infos["test"]
        varied_palettes += len({info["palette"] for info in all_infos}) > 1
        varied_roles += len({(info["paint"], info["locator"]) for info in all_infos}) > 1
        largest_train = max(info["shape"][0] * info["shape"][1]
                            for info in infos["train"])
        smaller_tests += any(info["shape"][0] * info["shape"][1] < largest_train
                             for info in infos["test"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert loc_counts == {3, 4, 5} and {4, 8} <= transform_counts
    assert len(shapes) >= 35 and len(stencils) >= 25 and len(layouts) >= 150
    assert {1, 2, 3, 4} <= target_counts
    assert changed_in_puzzle == 42 and repeated_in_puzzle >= 8
    assert clipped_sides == {"top", "bottom", "left", "right"}
    assert all(sides == clipped_sides for sides in split_clip_sides.values())
    assert reference_sides == clipped_sides
    assert split_target_counts["train"] == {1, 2, 3}
    assert {1, 2, 3, 4} <= split_target_counts["test"]
    assert len(split_shapes["train"] & split_shapes["test"]) >= 10
    assert new_test_stencils >= 15 and varied_palettes >= 15 and varied_roles >= 15
    assert smaller_tests >= 15 and identity_pairs >= 15 and repeated_orientations >= 15
