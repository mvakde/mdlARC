"""Framed seed/cascade analogies with input-local generation ink.

All semantics are read from the input. No parent generator, external file,
training output, random state, or saved example is used by solve().
"""

import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _next(mask):
    following = np.zeros(len(mask), dtype=bool)
    following[1:-1] = mask[:-2] & ~mask[1:-1] & mask[2:]
    return following


def _history(mask):
    rows = [np.asarray(mask, dtype=bool).copy()]
    # The active interval contracts by at least one column at each end.
    for _ in range(len(mask)):
        following = _next(rows[-1])
        if not following.any():
            return rows
        rows.append(following)
    raise ValueError("Noncontracting recurrence")


def _seed_info(strip, background):
    if strip.shape[0] != 1:
        raise ValueError("The seed must be a single-row strip")
    colors = set(map(int, np.unique(strip))) - {background}
    if len(colors) != 1:
        raise ValueError("The seed needs exactly one active color")
    color = colors.pop()
    mask = strip[0] == color
    if not _next(mask).any():
        raise ValueError("The seed must have a productive one-cell hole")
    return color, mask


def _cascade(strip, height, background, odd):
    seed_color, mask = _seed_info(strip, background)
    if odd in (background, seed_color):
        raise ValueError("The two phases and background must be distinct")
    rows = _history(mask)
    if height < 2 * (len(rows) - 1) + 3:
        raise ValueError("Canvas must include the first empty generation")
    result = np.full((height, strip.shape[1]), background, dtype=np.int8)
    for generation, row in enumerate(rows):
        result[2 * generation, row] = odd if generation % 2 else seed_color
    return result


def _outline(grid, box, color):
    r, c, h, w = box[:4]
    grid[r, c:c + w] = grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = grid[r:r + h, c + w - 1] = color


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r + h < y + bh and c + w < x + bw


def _disjoint(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return r + h <= y or y + bh <= r or c + w <= x or x + bw <= c


def _frame_rectangles(grid):
    """Recover isolated monochrome rectangular outline components by pixels."""
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    frames = []
    for r in range(height):
        for c in range(width):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            visited[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (0 <= yy < height and 0 <= xx < width
                            and not visited[yy, xx] and grid[yy, xx] == color):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            top = min(y for y, _ in cells)
            left = min(x for _, x in cells)
            bottom = max(y for y, _ in cells)
            right = max(x for _, x in cells)
            h, w = bottom - top + 1, right - left + 1
            if (h >= 3 and w >= 3 and len(cells) == 2 * h + 2 * w - 4
                    and all(y in (top, bottom) or x in (left, right) for y, x in cells)):
                frames.append((top, left, h, w, color))
    return frames


def _crop(grid, box):
    r, c, h, w = box[:4]
    return grid[r + 1:r + h - 1, c + 1:c + w - 1].copy()


def _read_scene(input_grid):
    grid = _grid(input_grid)
    frames = _frame_rectangles(grid)
    if len(frames) != 6:
        raise ValueError("Expected two enclosures and four panel frames")
    groups = [b for b in frames if sum(_inside(a, b) for a in frames) == 2]
    if len(groups) != 2 or not _disjoint(*groups):
        raise ValueError("Expected two disjoint panel pairs")
    groups.sort(key=lambda b: (b[1], b[0]))
    records = []
    assigned = set()
    for group in groups:
        panels = [b for b in frames if _inside(b, group)]
        panels.sort(key=lambda b: b[2])
        if (len(panels) != 2 or not _disjoint(*panels)
                or panels[0][2] != 3 or panels[1][2] < 7
                or panels[0][3] != panels[1][3]):
            raise ValueError("Each pair needs a seed strip and a taller companion of equal width")
        assigned.update(panels)
        records.append(dict(group=group, panels=panels,
                            seed=_crop(grid, panels[0]), full=_crop(grid, panels[1])))
    if len(assigned) != 4:
        raise ValueError("Panel association is ambiguous")
    blank_indices = [i for i, record in enumerate(records)
                     if np.all(record["full"] == record["full"][0, 0])]
    if len(blank_indices) != 1:
        raise ValueError("Exactly one large companion must be uniformly blank")
    target_index = blank_indices[0]
    target, reference = records[target_index], records[1 - target_index]
    background = int(target["full"][0, 0])
    ref_color, ref_mask = _seed_info(reference["seed"], background)
    target_color, target_mask = _seed_info(target["seed"], background)
    odd_colors = set(map(int, np.unique(reference["full"]))) - {background, ref_color}
    if len(odd_colors) != 1:
        raise ValueError("The reference must demonstrate one additional generation ink")
    odd = odd_colors.pop()
    if odd == target_color or len(_history(ref_mask)) < 3:
        raise ValueError("Reference must expose both phases, with distinct target ink")
    expected = _cascade(reference["seed"], reference["full"].shape[0], background, odd)
    if not np.array_equal(reference["full"], expected):
        raise ValueError("Reference does not demonstrate the complete spaced triplet recurrence")
    # Check fit without using the blank panel's contents as a generated answer.
    if target["full"].shape[0] < 2 * (len(_history(target_mask)) - 1) + 3:
        raise ValueError("Target companion truncates the cascade or its terminal row")
    return dict(reference=reference, target=target, background=background,
                odd=odd, groups=groups, target_index=target_index)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the missing cascade interior, using this input's reference."""
    scene = _read_scene(input_grid)
    target = scene["target"]
    return _cascade(target["seed"], target["full"].shape[0],
                    scene["background"], scene["odd"])


def _chain(depth):
    segment = np.zeros(2 * depth + 1, dtype=bool)
    segment[::2] = True
    return segment


def _pack(rng, pieces, max_width):
    pieces = list(pieces)
    rng.shuffle(pieces)
    gaps = [2] * (len(pieces) - 1)
    minimum = sum(map(len, pieces)) + sum(gaps)
    if minimum > max_width:
        raise ValueError("Mask pieces do not fit")
    width = int(rng.integers(minimum, max_width + 1))
    slack = width - minimum
    allocation = rng.multinomial(slack, [1 / (len(gaps) + 2)] * (len(gaps) + 2))
    offset = int(allocation[0])
    mask = np.zeros(width, dtype=bool)
    for i, piece in enumerate(pieces):
        mask[offset:offset + len(piece)] = piece
        offset += len(piece)
        if i < len(gaps):
            offset += gaps[i] + int(allocation[i + 1])
    return mask


def _make_mask(rng, kind, depth, max_width):
    if kind == "dense":
        width = int(rng.integers(5, max_width + 1))
        mask = np.ones(width, dtype=bool)
        mask[int(rng.integers(1, width - 1))] = False
        return mask
    if kind == "random":
        for _ in range(32):
            width = int(rng.integers(5, max_width + 1))
            mask = rng.random(width) < rng.uniform(.3, .8)
            if 1 <= len(_history(mask)) - 1 <= depth:
                return mask
        kind = "pure"  # Bounded fallback to the same cellular family.
    pieces = [_chain(depth)]
    room = max_width - len(pieces[0]) - 2
    if kind == "mixed":
        pieces.append(np.ones(int(rng.integers(3, min(5, room) + 1)), dtype=bool))
    elif kind == "multi":
        other_depth = int(rng.integers(1, min(depth, (room - 1) // 2) + 1))
        pieces.append(_chain(other_depth))
    elif kind == "pure":
        if room >= 1 and rng.random() < .3:
            pieces.append(np.ones(int(rng.integers(1, min(4, room) + 1)), dtype=bool))
    else:
        raise ValueError("Unknown seed structure")
    result = _pack(rng, pieces, max_width)
    assert len(_history(result)) - 1 == depth
    return result


def _build_scene(rng, ref_mask, target_mask, ref_height, target_height,
                 palette, wallpaper_mode):
    background, ref_color, target_color, odd = map(int, palette[:4])
    masks = [ref_mask, target_mask]
    heights = [ref_height, target_height]
    strips = [np.where(m, color, background).astype(np.int8)[None, :]
              for m, color in zip(masks, (ref_color, target_color))]
    reference = _cascade(strips[0], ref_height, background, odd)
    answer = _cascade(strips[1], target_height, background, odd)
    companions = [reference, np.full(answer.shape, background, dtype=np.int8)]
    # Two independent vertically paired enclosures; their left/right order varies.
    order = [0, 1]
    rng.shuffle(order)
    minimum_widths = [len(masks[i]) + 4 for i in order]
    minimum_heights = [heights[i] + 8 for i in order]
    width = int(rng.integers(sum(minimum_widths) + 1, 31))
    height = int(rng.integers(max(minimum_heights), 31))
    slack = rng.multinomial(width - sum(minimum_widths) - 1, [.2] * 5)
    group_widths = [minimum_widths[i] + int(slack[i]) for i in range(2)]
    lefts = [int(slack[2]), int(slack[2]) + group_widths[0] + 1 + int(slack[3])]
    if wallpaper_mode == 0:
        grid = np.full((height, width), background, dtype=np.int8)
    else:
        rr, cc = np.indices((height, width))
        bh, bw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        grid = np.asarray(palette, dtype=np.int8)[8 + (rr // bh + cc // bw) % 2].copy()
    for slot, i in enumerate(order):
        gh = int(rng.integers(minimum_heights[slot], height + 1))
        gw = group_widths[slot]
        top, left = int(rng.integers(0, height - gh + 1)), lefts[slot]
        grid[top:top + gh, left:left + gw] = background
        _outline(grid, (top, left, gh, gw), int(palette[4 + i]))
        contents = [strips[i], companions[i]]
        if rng.integers(2):
            contents.reverse()
        gap = int(rng.integers(1, gh - heights[i] - 6))
        remaining = gh - heights[i] - 7 - gap
        row = top + 1 + int(rng.integers(0, remaining + 1))
        for interior in contents:
            ph, pw = interior.shape[0] + 2, interior.shape[1] + 2
            column = left + int(rng.integers(1, gw - pw))
            _outline(grid, (row, column, ph, pw), int(palette[6 + i]))
            grid[row + 1:row + ph - 1, column + 1:column + pw - 1] = interior
            row += ph + gap
    return {"input": grid, "output": answer}


def _make_pair(rng, kind, depth, exclude_odd=None):
    target_mask = _make_mask(rng, kind, depth, 16)
    max_ref_width = 21 - len(target_mask)
    ref_depth = int(rng.integers(2, min(3, (max_ref_width - 1) // 2) + 1))
    ref_kind = "pure"
    if max_ref_width >= 2 * ref_depth + 6 and rng.random() < .35:
        ref_kind = "mixed"
    elif max_ref_width >= 2 * ref_depth + 6 and rng.random() < .4:
        ref_kind = "multi"
    ref_mask = _make_mask(rng, ref_kind, ref_depth, max_ref_width)
    target_depth = len(_history(target_mask)) - 1
    ref_height = 2 * ref_depth + 3 + int(rng.integers(0, 5))
    target_height = 2 * target_depth + 3 + int(rng.integers(0, 4))
    palette = rng.permutation(10)
    if int(palette[3]) == exclude_odd:
        # Swapping two unused role assignments is deterministic and bounded.
        palette[[3, 9]] = palette[[9, 3]]
    if rng.random() < .3:
        palette[2] = palette[1]  # Shared seed roles are also readable.
    if rng.random() < .3:
        palette[7] = palette[6]
    return _build_scene(rng, ref_mask, target_mask, ref_height, target_height,
                        palette, int(rng.integers(2)))


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    heldout_depth = int(rng.integers(4, 8))
    train_specs = [("mixed", int(rng.integers(2, 4))),
                   ("multi", int(rng.integers(2, 4)))]
    kinds = ("pure", "mixed", "multi", "dense", "random")
    for _ in range(num_train - 2):
        train_specs.append((str(rng.choice(kinds)), int(rng.integers(1, 4))))
    # Multi with depth one is productive, but the mandatory example has depth >=2.
    test_specs = [(str(rng.choice(kinds)), int(rng.integers(1, 4))) for _ in range(num_test)]
    deep_kind = "mixed" if heldout_depth <= 5 and rng.random() < .5 else "pure"
    test_specs[int(rng.integers(num_test))] = (deep_kind, heldout_depth)
    puzzle, seen = {"train": [], "test": []}, set()
    first_odd = None
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, (kind, depth) in enumerate(specs):
            for _ in range(24):
                pair = _make_pair(rng, kind, depth,
                                  first_odd if split == "train" and index == 1 else None)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("Could not construct a distinct input after 24 attempts")
            if split == "train" and index == 0:
                first_odd = _read_scene(pair["input"])["odd"]
            seen.add(signature)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _audit_cascade(strip, full, background, odd):
    """Scalar pixel audit: checks every cell, including spacers and empty tail.

    Deliberately does not call _next, _history or _cascade. This is an internal
    invariant check, not a claimed independent semantic solver.
    """
    assert strip.shape[0] == 1
    seed_colors = {int(cell) for cell in strip[0] if int(cell) != background}
    assert len(seed_colors) == 1
    seed_color = seed_colors.pop()
    assert odd not in (background, seed_color)
    current = [int(x) != background for x in strip[0]]
    expected = np.full(full.shape, background, dtype=np.int8)
    depth = 0
    while any(current):
        row = 2 * depth
        assert row < full.shape[0]
        for c, active in enumerate(current):
            if active:
                expected[row, c] = seed_color if depth % 2 == 0 else odd
        following = [False] * len(current)
        for c in range(1, len(current) - 1):
            following[c] = current[c - 1] and not current[c] and current[c + 1]
        current = following
        depth += 1
        assert depth <= full.shape[1]
    assert 2 * depth < full.shape[0], "First empty generation must be visible"
    assert np.array_equal(full, expected), "Incorrect cascade, spacing, parity or tail"
    return depth - 1


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, depths, train_odds = set(), {"train": [], "test": []}, set()
    solid_demo = multi_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(inp)
            assert signature not in seen, "Duplicate input"
            seen.add(signature)
            scene = _read_scene(inp)
            bg, odd = scene["background"], scene["odd"]
            reference, target = scene["reference"], scene["target"]
            assert out.shape == target["full"].shape
            assert _audit_cascade(reference["seed"], reference["full"], bg, odd) >= 2
            depth = _audit_cascade(target["seed"], out, bg, odd)
            depths[split].append(depth)
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(out, target["full"])
            if split == "train":
                train_odds.add(odd)
                mask = target["seed"][0] != bg
                solid_demo |= bool(np.any(mask[:-2] & mask[1:-1] & mask[2:]))
                first = np.flatnonzero(_next(mask))
                multi_demo |= len(first) > 1 and bool(np.any(np.diff(first) > 2))
    assert len(train_odds) >= 2, "Demonstrate input-local odd ink"
    assert solid_demo and multi_demo, "Training must distinguish active centers and separate fragments"
    assert max(depths["train"]) <= 3 and max(depths["train"]) >= 2
    assert max(depths["test"]) >= 4, "Include a genuinely deeper target test"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = pair[role]
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _necessity_checks():
    # Change only the informative reference cascade's odd-generation cells.
    # All target pixels, source strips, frame geometry and other pixels agree.
    reference = _chain(2)
    target = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    palettes = [np.arange(10), np.array([0, 1, 2, 8, 4, 5, 6, 7, 3, 9])]
    twins = [_build_scene(np.random.default_rng(91), reference, target, 9, 10, p, 0)
             for p in palettes]
    a, b = [solve(p["input"]) for p in twins]
    sa, sb = [_read_scene(p["input"]) for p in twins]
    assert np.array_equal(sa["target"]["seed"], sb["target"]["seed"])
    assert sa["target"]["panels"] == sb["target"]["panels"]
    assert np.array_equal(twins[0]["input"] != twins[1]["input"],
                          twins[0]["input"] == 3)
    assert not np.array_equal(a, b), "Reference must change the answer"
    odd_activity = a == 3
    assert np.array_equal(a != b, odd_activity)
    assert np.all(b[odd_activity] == 8)
    # Parent B's empty center, exact one-cell gap, iteration, parity and stop.
    assert not _next(np.array([1, 1, 1], dtype=bool)).any()
    assert not _next(np.array([1, 0, 0, 1], dtype=bool)).any()
    assert not _next(np.array([1, 0, 0, 0, 1], dtype=bool)).any()
    strip = np.array([[1, 0, 1, 0, 1]], dtype=np.int8)
    result = _cascade(strip, 9, 0, 2)
    assert result[2].tolist() == [0, 2, 0, 2, 0]
    assert result[4].tolist() == [0, 0, 1, 0, 0]
    assert not result[6:].any()
    assert not result[1::2].any()
    assert _audit_cascade(strip, result, 0, 2) == 2
    one = _cascade(np.array([[1, 0, 1]], dtype=np.int8), 5, 0, 2)
    assert np.count_nonzero(one[2:]) == 1
    # Omitting the empty-center requirement would fire inside the solid triple.
    wrong_first = np.zeros(len(target), dtype=bool)
    wrong_first[1:-1] = target[:-2] & target[2:]
    assert not np.array_equal(wrong_first, _next(target))
    for pair in twins:
        assert np.array_equal(solve(pair["input"]), pair["output"])
    return {"reference_twins": 2, "target_held_fixed": True,
            "answers_differ": True, "changed_output_cells": int(odd_activity.sum()),
            "only_odd_activity_changes": True, "solid_center_shortcut_rejected": True}


def _mask_regression_checks():
    """Exercise every productive Boolean row of widths 5..8 in full scenes.

    This deliberately includes masks outside the generator's packing templates.
    Work is bounded by 480 candidate masks, with one scene alive at a time.
    The scalar audit checks outputs without the vectorized recurrence helpers.
    """
    checked = 0
    for width in range(5, 9):
        for bits in range(1 << width):
            active = [bool(bits & (1 << c)) for c in range(width)]
            if not any(active[c - 1] and not active[c] and active[c + 1]
                       for c in range(1, width - 1)):
                continue
            current, nonempty_rows = active, 0
            while any(current):
                nonempty_rows += 1
                current = [False] + [
                    current[c - 1] and not current[c] and current[c + 1]
                    for c in range(1, width - 1)] + [False]
            pair = _build_scene(
                np.random.default_rng((width << 10) + bits), _chain(2),
                np.array(active, dtype=bool), 9, 2 * nonempty_rows + 1,
                np.arange(10), checked % 2)
            before = _key(pair["input"])
            answer = solve(pair["input"])
            assert _key(pair["input"]) == before, "solve mutated its input"
            assert np.array_equal(answer, pair["output"])
            assert np.array_equal(solve(pair["input"].tolist()), answer)
            strip = np.where(active, 2, 0).astype(np.int8)[None, :]
            assert _audit_cascade(strip, answer, 0, 3) == nonempty_rows - 1
            checked += 1
    assert checked >= 150
    return checked


def self_test():
    """42 streaming seeds; no files, trace history or retained puzzle collection."""
    shapes, output_shapes, masks, palettes, depths = set(), set(), set(), set(), set()
    placements, widths, hashes = set(), set(), set()
    full_size = edge_seed = shared_seed = deep_composite = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _read_scene(pair["input"])
                ref, target, bg = scene["reference"], scene["target"], scene["background"]
                mask = target["seed"][0] != bg
                depth = len(_history(mask)) - 1
                depths.add(depth)
                widths.add(len(mask))
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                masks.add((len(mask), mask.tobytes()))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
                placements.add((scene["target_index"],
                                ref["panels"][0][0] < ref["panels"][1][0],
                                target["panels"][0][0] < target["panels"][1][0]))
                full_size += pair["input"].shape == (30, 30)
                edge_seed += bool(mask[0] or mask[-1])
                shared_seed += set(np.unique(ref["seed"])) == set(np.unique(target["seed"]))
                positions = np.flatnonzero(mask)
                deep_composite += split == "test" and depth >= 4 and bool(np.any(np.diff(positions) != 2))
    assert len(hashes) == 42
    assert len(shapes) >= 15 and len(output_shapes) >= 25
    assert len(masks) >= 100 and len(widths) >= 8 and len(palettes) >= 15
    assert depths >= {1, 2, 3, 4, 5, 6, 7} and len(placements) == 8
    assert min(full_size, edge_seed, shared_seed, deep_composite) > 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    assert _digest(generate_puzzle(314159, 7, 3)) == _digest(generate_puzzle(314159, 7, 3))
    assert _digest(generate_puzzle(0)) != _digest(generate_puzzle(1))
    damaged = generate_puzzle(17, 2, 1)
    damaged["test"][0]["output"][1, 0] = (int(damaged["test"][0]["output"][1, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect spacer accepted")
    necessity = _necessity_checks()
    productive_masks = _mask_regression_checks()
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "seed_masks": len(masks), "target_widths": sorted(widths),
            "depths": sorted(depths), "palette_sets": len(palettes),
            "order_configurations": len(placements), "full_30x30": full_size,
            "edge_contact_seeds": edge_seed, "shared_seed_palettes": shared_seed,
            "deep_composite_tests": deep_composite, "necessity": necessity,
            "exhaustive_productive_masks": productive_masks}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
