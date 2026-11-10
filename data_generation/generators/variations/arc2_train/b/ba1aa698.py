"""Generate framed-panel motion puzzles.

The only rule shared by all pairs is rigid, constant translation of an
unchanged motif. Colors, geometry, and direction are visible within a pair.
"""

import hashlib

import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (3, 2), (4, 2), (5, 2), (7, 2),
    (4, 3), (5, 3), (6, 3), (8, 3),
    (5, 4), (7, 4), (10, 4),
)

# The compact shapes can be rotated and reflected. The source T is included.
_MOTIFS = (
    ("11",), ("111",), ("1111",),
    ("11", "11"), ("111", "111"),
    ("111", "010"), ("101", "111"),
    ("110", "011"), ("100", "111"),
    ("10", "11"), ("01", "11"),
    ("010", "111", "010"),
    ("100", "110", "011"),
    ("100", "111", "001"),
    ("111", "101"), ("110", "010", "011"),
    ("1", "1", "1"), ("1", "1", "1", "1"),
)

# Exact source-pair hashes, including both grid shapes and cell contents.
_SOURCE_PAIR_HASHES = {
    "3b66a604b09c9bfb5f7065010ff3d10c6fd7adfcb37770c5b89e897bad0cbee3",
    "79b00bd5d6a424fe1f5103bce1095b1469dc18fddee256a1d4b1e805bd9daa5e",
    "23fb88543011a006b7b714afd30264afee7382d728276df38940162467bd2a0a",
    "513e7d9fb8a713d68fe5b49912fea21c0195aacb9ff293354dab51de15633a88",
}


def _checked_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _checked_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_train, num_test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_hash(pair):
    a, b = pair["input"], pair["output"]
    raw = (str(a.shape) + ":").encode() + a.astype(np.int8).tobytes()
    raw += (str(b.shape) + ":").encode() + b.astype(np.int8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _shape_key(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _sample_motif(rng, panel_width):
    for _ in range(30):
        rows = ("111", "010") if rng.random() < 0.2 else _MOTIFS[int(rng.integers(len(_MOTIFS)))]
        mask = np.array([[v == "1" for v in row] for row in rows], dtype=bool)
        mask = np.rot90(mask, int(rng.integers(4)))
        if int(rng.integers(2)):
            mask = np.fliplr(mask)
        if mask.shape[1] <= panel_width - 2:
            return mask.copy()
    # A short bar always fits even the narrowest panel.
    return np.array([[True, True]], dtype=bool)


def _palette(rng, colors, shared, new_frame=False):
    frame = int(rng.choice(colors))
    if new_frame:
        unused = [c for c in range(10) if c not in colors]
        if unused:
            frame = int(rng.choice(unused))
    background = int(rng.choice([c for c in colors if c != frame]))
    if shared:
        motif_color = frame
    else:
        motif_color = int(rng.choice([c for c in colors if c not in (frame, background)]))
    return frame, background, motif_color


def _render_pair(rng, n_panels, panel_width, step, motif, palette, stacked):
    frame, background, motif_color = palette
    motif_height, motif_width = motif.shape
    minimum = motif_height + n_panels * abs(step) + 2
    assert minimum <= 28
    if minimum <= 14 and rng.random() < 0.18:
        inner_height = 14
    elif rng.random() < 0.08:
        inner_height = 28
    else:
        inner_height = int(rng.integers(minimum, 29))
    slack = inner_height - motif_height - n_panels * abs(step) - 2
    first_top = 1 + int(rng.integers(slack + 1))
    if step < 0:
        first_top += n_panels * abs(step)
    left = int(rng.integers(1, panel_width - motif_width))

    height = inner_height + 2
    width = n_panels * (panel_width + 1) + 1
    inp = np.full((height, width), background, dtype=np.int8)
    inp[[0, -1], :] = frame
    inp[:, ::panel_width + 1] = frame
    out = np.full((height, panel_width + 2), background, dtype=np.int8)
    out[[0, -1], :] = frame
    out[:, [0, -1]] = frame
    for index in range(n_panels + 1):
        target = out if index == n_panels else inp
        col = 1 + left if index == n_panels else 1 + index * (panel_width + 1) + left
        top = 1 + first_top + index * step
        patch = target[top:top + motif_height, col:col + motif_width]
        patch[motif] = motif_color
    if stacked:
        inp, out = np.rot90(inp, -1).copy(), np.rot90(out, -1).copy()
    return {"input": inp, "output": out}


def _new_pair(rng, regime, colors, shared, seen_orientations, train_infos, simple_style):
    counts = (2, 3, 3, 4, 4, 5) if regime == "source_train" else (2, 3, 4, 4, 5, 5)
    n_panels = int(rng.choice(counts))
    maximum_width = min(7, 29 // n_panels - 1)
    panel_width = int(rng.integers(4, maximum_width + 1))
    reusable = [info["mask"] for info in train_infos
                if info["mask"].shape[1] <= panel_width - 2]
    if regime == "target_test" and reusable and rng.random() < 0.4:
        motif = reusable[int(rng.integers(len(reusable)))].copy()
    else:
        motif = _sample_motif(rng, panel_width)

    possible_steps = [s for s in (-4, -3, -2, -1, 1, 2, 3, 4)
                      if motif.shape[0] + n_panels * abs(s) + 2 <= 28]
    train_steps = [info["step"] for info in train_infos if info["step"] in possible_steps]
    if regime == "target_test" and train_steps and rng.random() < 0.4:
        step = int(rng.choice(train_steps))
    else:
        step = int(rng.choice(possible_steps))

    if simple_style:
        stacked = False
    elif regime == "source_train":
        stacked = bool(rng.random() < 0.32)
    else:
        stacked = bool(rng.choice(seen_orientations))
    palette = _palette(
        rng, colors, shared,
        new_frame=(regime == "target_test" and not simple_style and rng.random() < 0.28),
    )
    return _render_pair(rng, n_panels, panel_width, step, motif, palette, stacked)


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    todo = [next(iter(cells))]
    reached = set()
    while todo:
        cell = todo.pop()
        if cell in reached:
            continue
        reached.add(cell)
        r, c = cell
        todo.extend(n for n in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                    if n in cells and n not in reached)
    return reached == cells


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    frame = int(inp[0, 0])
    assert np.all(inp[0] == frame) and np.all(inp[-1] == frame)
    assert np.all(inp[:, 0] == frame) and np.all(inp[:, -1] == frame)
    internal_cols = np.flatnonzero(np.all(inp[:, 1:-1] == frame, axis=0)) + 1
    internal_rows = np.flatnonzero(np.all(inp[1:-1, :] == frame, axis=1)) + 1
    assert (len(internal_cols) > 0) != (len(internal_rows) > 0)
    stacked = len(internal_rows) > 0
    if stacked:
        inp, out = np.rot90(inp, 1), np.rot90(out, 1)
    height, width = inp.shape
    frame = int(inp[0, 0])
    dividers = np.flatnonzero(np.all(inp == frame, axis=0))
    assert len(dividers) >= 3 and dividers[0] == 0 and dividers[-1] == width - 1
    spacing = np.diff(dividers)
    assert np.all(spacing == spacing[0])
    panel_width = int(spacing[0]) - 1
    n_panels = len(dividers) - 1
    assert 4 <= panel_width <= 7 and 2 <= n_panels <= 5
    assert width == n_panels * (panel_width + 1) + 1
    assert out.shape == (height, panel_width + 2)

    backgrounds, colors, masks, tops, lefts = [], [], [], [], []
    for i in range(n_panels):
        block = inp[1:-1, dividers[i]+1:dividers[i+1]]
        counts = np.bincount(block.ravel().astype(int), minlength=10)
        background = int(np.argmax(counts))
        assert background != frame and counts[background] > block.size // 2
        mask = block != background
        assert np.any(mask)
        motif_colors = np.unique(block[mask])
        assert len(motif_colors) == 1
        rr, cc = np.nonzero(mask)
        top, left = int(rr.min()), int(cc.min())
        bottom, right = int(rr.max()) + 1, int(cc.max()) + 1
        trimmed = mask[top:bottom, left:right]
        assert _connected(trimmed)
        assert top >= 1 and left >= 1
        assert bottom <= block.shape[0] - 1 and right <= block.shape[1] - 1
        backgrounds.append(background)
        colors.append(int(motif_colors[0]))
        masks.append(trimmed)
        tops.append(top)
        lefts.append(left)
    assert len(set(backgrounds)) == len(set(colors)) == len(set(lefts)) == 1
    assert all(np.array_equal(masks[0], m) for m in masks[1:])
    steps = np.diff(tops)
    assert len(steps) and steps[0] != 0 and np.all(steps == steps[0])
    next_top = tops[-1] + int(steps[0])
    assert 1 <= next_top <= height - 3 - masks[0].shape[0]

    expected = np.full(out.shape, backgrounds[0], dtype=np.int8)
    expected[[0, -1], :] = frame
    expected[:, [0, -1]] = frame
    mh, mw = masks[0].shape
    patch = expected[1+next_top:1+next_top+mh, 1+lefts[0]:1+lefts[0]+mw]
    patch[masks[0]] = colors[0]
    assert np.array_equal(out, expected)
    assert set(np.unique(inp)) == set(np.unique(out)) == {frame, backgrounds[0], colors[0]}
    return {
        "stacked": stacked, "height": height, "n_panels": n_panels,
        "panel_width": panel_width, "frame": frame,
        "background": backgrounds[0], "motif_color": colors[0],
        "mask": masks[0], "top": tops[0], "left": lefts[0],
        "step": int(steps[0]),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _checked_count(seed, "seed", 0, 2**63 - 1)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # This palette pool gives a puzzle visual cohesion, but has no hidden
    # color mapping: every pair displays its own frame, field, and motif.
    simple_style = bool(rng.random() < 0.22)
    palette_size = 3 if simple_style else int(rng.choice((3, 3, 4, 5, 6)))
    colors = tuple(int(c) for c in rng.permutation(10)[:palette_size])
    shared_flags = [True, False]
    shared_flags.extend(bool(rng.integers(2)) for _ in range(num_train - 2))
    rng.shuffle(shared_flags)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_infos = []
    seen_orientations = []
    for split, regime, count in (
        ("train", "source_train", num_train),
        ("test", "target_test", num_test),
    ):
        for index in range(count):
            for _ in range(60):
                shared = shared_flags[index] if split == "train" else bool(rng.integers(2))
                pair = _new_pair(
                    rng, regime, colors, shared, seen_orientations,
                    train_infos, simple_style,
                )
                key = _key(pair["input"])
                if key not in seen_inputs and _pair_hash(pair) not in _SOURCE_PAIR_HASHES:
                    break
            else:
                raise ValueError("could not construct a unique pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
            if split == "train":
                info = _analyze_pair(pair)
                train_infos.append(info)
                seen_orientations.append(info["stacked"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert _pair_hash(pair) not in _SOURCE_PAIR_HASHES
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_analyze_pair(pair))

    # There is no seed-wide role cycle or direction convention. Both legal
    # motif/frame color relationships are demonstrated by training.
    assert {i["motif_color"] == i["frame"] for i in infos["train"]} == {False, True}
    train_orientations = {i["stacked"] for i in infos["train"]}
    assert all(i["stacked"] in train_orientations for i in infos["test"])


def _same_puzzle(a, b):
    return all(
        np.array_equal(x[role], y[role])
        for split in ("train", "test")
        for x, y in zip(a[split], b[split])
        for role in ("input", "output")
    ) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(901, 7, 3)
    assert _same_puzzle(a, generate_puzzle(901, 7, 3))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(55, *counts)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
        validate_generated_puzzle(puzzle)
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")

    train_counts, test_counts = set(), set()
    shapes, structures, motifs, palettes, steps = set(), set(), set(), set(), set()
    source_layouts, heights, shared_types, orientations = set(), set(), set(), set()
    same_direction = mixed_direction = reused_motif = novel_motif = False
    reused_step = novel_step = wide_palette = False
    coverage = {
        name: 0 for name in (
            "source_like", "new_test_color", "within_left", "within_orientation",
            "two_panels", "five_panels", "symmetric_t", "combined_variation",
            "within_layout", "within_palette", "within_motif",
        )
    }
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p) for p in puzzle["train"]]
        test_info = [_analyze_pair(p) for p in puzzle["test"]]
        signs = {np.sign(i["step"]) for i in train_info}
        same_direction |= len(signs) == 1
        mixed_direction |= len(signs) > 1
        train_motifs = {_shape_key(i["mask"]) for i in train_info}
        train_steps = {i["step"] for i in train_info}
        reused_motif |= any(_shape_key(i["mask"]) in train_motifs for i in test_info)
        novel_motif |= any(_shape_key(i["mask"]) not in train_motifs for i in test_info)
        reused_step |= any(i["step"] in train_steps for i in test_info)
        novel_step |= any(i["step"] not in train_steps for i in test_info)
        all_infos = train_info + test_info
        train_colors = {
            int(c) for pair in puzzle["train"] for c in np.unique(pair["input"])
        }
        puzzle_colors = {
            int(c) for split in ("train", "test")
            for pair in puzzle[split] for c in np.unique(pair["input"])
        }
        coverage["source_like"] += (
            len(puzzle_colors) <= 3 and all(not i["stacked"] for i in all_infos)
        )
        coverage["new_test_color"] += any(
            any(int(c) not in train_colors for c in np.unique(pair["input"]))
            for pair in puzzle["test"]
        )
        coverage["within_left"] += len({i["left"] for i in all_infos}) > 1
        coverage["within_orientation"] += len({i["stacked"] for i in all_infos}) > 1
        coverage["within_layout"] += len({
            (i["n_panels"], i["panel_width"]) for i in all_infos
        }) > 1
        coverage["within_palette"] += len({
            (i["frame"], i["background"], i["motif_color"]) for i in all_infos
        }) > 1
        coverage["within_motif"] += len({
            _shape_key(i["mask"]) for i in all_infos
        }) > 1
        coverage["two_panels"] += any(i["n_panels"] == 2 for i in all_infos)
        coverage["five_panels"] += any(i["n_panels"] == 5 for i in all_infos)
        coverage["symmetric_t"] += any(
            i["mask"].astype(int).tolist() == [[1, 1, 1], [0, 1, 0]]
            for i in all_infos
        )
        coverage["combined_variation"] += any(
            i["stacked"] and i["n_panels"] >= 3 and i["panel_width"] >= 5
            and abs(i["step"]) >= 2 and i["motif_color"] != i["frame"]
            for i in all_infos
        )
        colors = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                shapes.add(pair["input"].shape)
                structures.add((info["stacked"], info["n_panels"], info["panel_width"],
                                info["height"], info["top"], info["left"], info["step"],
                                _shape_key(info["mask"])))
                motifs.add(_shape_key(info["mask"]))
                palettes.add((info["frame"], info["background"], info["motif_color"]))
                steps.add(info["step"])
                heights.add(info["height"])
                shared_types.add(info["frame"] == info["motif_color"])
                orientations.add(info["stacked"])
                colors.update(np.unique(pair["input"]).tolist())
                if split == "train" and not info["stacked"]:
                    source_layouts.add((info["n_panels"], info["panel_width"]))
        wide_palette |= len(colors) >= 4
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(structures) >= 140
    assert len(motifs) >= 15 and len(palettes) >= 35
    assert steps == set(range(-4, 0)) | set(range(1, 5))
    assert 16 in heights and 30 in heights
    assert shared_types == {True, False} and orientations == {True, False}
    assert {(4, 4), (3, 4), (3, 5)} <= source_layouts
    assert all((same_direction, mixed_direction, reused_motif, novel_motif,
                reused_step, novel_step, wide_palette))
    assert coverage["source_like"] >= 2 and coverage["new_test_color"] >= 5
    assert coverage["within_left"] >= 20 and coverage["within_orientation"] >= 10
    assert coverage["within_layout"] >= 25
    assert coverage["within_palette"] == coverage["within_motif"] == 42
    assert coverage["two_panels"] >= 10 and coverage["five_panels"] >= 10
    assert coverage["symmetric_t"] >= 2 and coverage["combined_variation"] >= 5
