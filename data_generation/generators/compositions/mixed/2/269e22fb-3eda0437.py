"""Recover a binary reference from its crop, then mark its largest blank window.

The reference and the two-cell background/highlight key are visible in each input.
No puzzle state, parent module, file, seed, or training output is used by solve.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 630050123, 685825204, 3310911, 669337441, 620976641,
              29409689, 404300404, 601697779, 297922596, 212513777,
              908644620, 290206294, 34174104, 157827209, 709528172,
              431919916, 332161298, 889187255, 32373945, 230502307,
              167992581, 50540662, 861738575, 980837777, 567667866,
              211119616, 681551728, 669950498, 320889279, 61737670,
              898408537, 544216965, 177392065, 361054943, 872409591,
              354340780, 170058871, 508865498, 299689685, 508624257,
              946328844)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(grid):
    grid = np.asarray(grid)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), "invalid grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid grid color")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _transform(grid, index):
    # Four rotations, then the same rotations of a left-right reflection.
    return np.rot90(grid if index < 4 else np.fliplr(grid), index % 4).copy()


def _rectangles(mask):
    """Enumerate maximal horizontal spans for every row interval.

    Any globally maximum rectangle must be maximal horizontally. Thus this
    O(H^2 W) search loses neither an optimum nor a tie, without storing all
    O(H^2 W^2) subrectangles.
    """
    height, width = mask.shape
    result = []
    for top in range(height - 1):
        common = mask[top].copy()
        for bottom in range(top + 1, height):
            common &= mask[bottom]
            edges = np.diff(np.r_[False, common, False].astype(np.int8))
            for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                if right - left >= 2:
                    rh, rw = bottom - top + 1, int(right - left)
                    result.append((rh * rw, top, int(left), rh, rw))
    return result


def _winner(mask):
    rectangles = _rectangles(mask)
    _require(bool(rectangles), "no eligible two-dimensional blank rectangle")
    area = max(item[0] for item in rectangles)
    winners = [item[1:] for item in rectangles if item[0] == area]
    _require(len(winners) == 1, "largest blank rectangle is tied")
    return winners[0]


def _mask(shape, rectangle):
    result = np.zeros(shape, dtype=bool)
    r, c, h, w = rectangle
    result[r:r + h, c:c + w] = True
    return result


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    components = []
    while remaining:
        start = remaining.pop()
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for other in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if other in remaining:
                    remaining.remove(other)
                    stack.append(other)
                    cells.append(other)
        components.append(cells)
    return components


def _parse(input_grid):
    grid = _grid(input_grid)
    _require(grid.shape[0] >= 6 and grid.shape[1] >= 3, "input is too small")
    background, highlight = map(int, grid[0, :2])
    _require(background != highlight and background != 0 and highlight != 0, "invalid key")
    _require(not np.any(grid[0, 2:]) and not np.any(grid[1]), "key must be isolated at upper left")
    components = _components(grid[2:] != 0)
    _require(len(components) == 2, "body must contain exactly two solid panels")
    panels = []
    for cells in components:
        rows, cols = zip(*cells)
        top, left, bottom, right = min(rows) + 2, min(cols), max(rows) + 3, max(cols) + 1
        panel = grid[top:bottom, left:right].copy()
        _require(len(cells) == panel.size and min(panel.shape) >= 2, "panel is not a solid rectangle")
        _require(len(np.unique(panel)) == 2, "each panel must be binary")
        panels.append(panel)
    panels.sort(key=lambda panel: panel.size, reverse=True)
    reference, crop = panels
    _require(reference.size > crop.size, "query must be a proper crop")
    _require(background in crop and highlight not in reference and highlight not in crop, "key roles conflict with panels")
    _require(set(map(int, np.unique(reference))).isdisjoint(map(int, np.unique(crop))),
             "reference and query palettes must be disjoint")
    return reference, crop, background, highlight


def _matches(reference, crop):
    """Return all exact (orientation, role map, offset, completed board) matches."""
    ref_colors = np.unique(reference)
    colors = np.unique(crop)
    binary = reference == ref_colors[1]
    h, w = crop.shape
    result = []
    for orientation in range(8):
        transformed = _transform(binary, orientation)
        H, W = transformed.shape
        if h > H or w > W:
            continue
        for swap in (0, 1):
            full = np.where(transformed, colors[1 - swap], colors[swap]).astype(np.int8)
            windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
            rows, cols = np.nonzero(np.all(windows == crop, axis=(-1, -2)))
            for r, c in zip(rows, cols):
                result.append((orientation, swap, int(r), int(c), full))
    return result


def _recover(reference, crop):
    matches = _matches(reference, crop)
    # Generated instances satisfy the stronger unique-location condition too.
    _require(len(matches) == 1, "crop must identify a unique orientation, mapping, and location")
    return matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    reference, crop, background, highlight = _parse(input_grid)
    _, _, _, _, full = _recover(reference, crop)
    r, c, h, w = _winner(full == background)
    output = full.copy()
    output[r:r + h, c:c + w] = highlight
    return output


def _features(full, background):
    blank = full == background
    winner = _winner(blank)
    r, c, h, w = winner
    target = _mask(full.shape, winner)
    others = [item for item in _rectangles(blank)
              if not (r <= item[1] and c <= item[2]
                      and item[1] + item[3] <= r + h and item[2] + item[4] <= c + w)]
    overlap = any(bool(np.any(_mask(full.shape, item[1:]) & target)) for item in others)
    remaining = blank & ~target
    longest = 0
    for view in (remaining, remaining.T):
        for row in view:
            edges = np.diff(np.r_[False, row, False].astype(np.int8))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            longest = max(longest, int(max(lengths, default=0)))
    component_sizes = [len(cells) for cells in _components(blank)]
    containing = next(len(cells) for cells in _components(blank) if (r, c) in cells)
    return {"winner": winner, "area": h * w,
            "runner_up": max((item[0] for item in others), default=0),
            "overlap": overlap, "long_line": longest > h * w,
            "connected_extra": containing > h * w,
            "largest_component": max(component_sizes),
            "wider_loser": any(item[4] > w for item in others),
            "taller_loser": any(item[3] > h for item in others)}


def _motif(board, rng):
    """Draw one small zero-valued bend, loop, staircase, comb, or bar."""
    H, W = board.shape
    h, w = int(rng.integers(2, min(6, H) + 1)), int(rng.integers(2, min(6, W) + 1))
    r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
    tile = board[r:r + h, c:c + w]
    kind = int(rng.integers(5))
    if kind == 0:
        tile[:, 0] = 0
        tile[-1] = 0
    elif kind == 1:
        tile[0] = tile[-1] = 0
        tile[:, 0] = tile[:, -1] = 0
    elif kind == 2:
        for y in range(h):
            x = min(w - 1, y * w // h)
            tile[y, x:min(w, x + 2)] = 0
    elif kind == 3:
        tile[:, 0] = 0
        tile[::2, :int(rng.integers(2, w + 1))] = 0
    else:
        tile[int(rng.integers(h))] = 0


def _reference(rng):
    H, W = int(rng.integers(8, 14)), int(rng.integers(10, 14))
    shapes = [(2, 4), (4, 2), (3, 3), (2, 5), (5, 2), (3, 4), (4, 3)]
    shapes = [(h, w) for h, w in shapes if h * w < W and h <= H - 3]
    # Rejection is bounded; only small in-memory arrays are retained.
    for attempt in range(320):
        board = np.ones((H, W), dtype=np.int8)
        h, w = shapes[int(rng.integers(len(shapes)))]
        r, c = int(rng.integers(H - h - 2)), int(rng.integers(W - w + 1))
        target = (r, c, h, w)
        board[r:r + h, c:c + w] = 0
        extensions = []
        for rr in (r - 1, r + h):
            if 0 <= rr < H - 2:
                for length in range(2, w):
                    if (h + 1) * length < h * w:
                        for start in range(w - length + 1):
                            extensions.append((rr, c + start, 1, length))
        for cc in (c - 1, c + w):
            if 0 <= cc < W:
                for length in range(2, h):
                    if (w + 1) * length < h * w:
                        for start in range(h - length + 1):
                            extensions.append((r + start, cc, length, 1))
        if not extensions:
            continue
        for index in rng.choice(len(extensions), min(len(extensions), int(rng.integers(1, 3))), replace=False):
            y, x, hh, ww = extensions[int(index)]
            board[y:y + hh, x:x + ww] = 0
        for _ in range(int(rng.integers(1, 6))):
            _motif(board[:-2], rng)
        length = int(rng.integers(h * w + 1, W + 1))
        start = int(rng.integers(W - length + 1))
        board[-1, start:start + length] = 0
        try:
            if _winner(board == 0) != target:
                continue
            _winner(board == 1)
            features = _features(board, 0)
        except ValueError:
            continue
        if not (features["overlap"] and features["long_line"]
                and features["connected_extra"]
                and (features["wider_loser"] or features["taller_loser"])):
            continue
        orbit = {_key(_transform(board, i) ^ swap) for i in range(8) for swap in (0, 1)}
        if len(orbit) == 16:
            return _transform(board, int(rng.integers(8)))
    raise RuntimeError("bounded reference construction exhausted")


def _crop(rng, reference, full, background, test, seen):
    H, W = full.shape
    target = _mask(full.shape, _winner(full == background))
    # Tests use a small distinctive fragment, training a broader window.
    for attempt in range(300):
        if test:
            h, w = int(rng.integers(2, min(H, 6) + 1)), int(rng.integers(2, min(W, 6) + 1))
        else:
            h, w = int(rng.integers(max(3, H // 2), H + 1)), int(rng.integers(max(3, W // 2), W + 1))
        if h * w == H * W:
            continue
        r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
        patch = full[r:r + h, c:c + w].copy()
        if len(np.unique(patch)) != 2 or _key(patch) in seen:
            continue
        # Every example must recover some winning cells outside the query crop.
        if int(target[r:r + h, c:c + w].sum()) == int(target.sum()):
            continue
        if test and h * w >= .5 * full.size:
            continue
        matches = _matches(reference, patch)
        if len(matches) == 1:
            return patch
    # A finite exhaustive fallback preserves the same shift and uniqueness tests.
    # The reference is small; the list of candidate rectangles is < 8,300.
    candidates = []
    for h in range(2, H + 1):
        for w in range(2, W + 1):
            fraction = h * w / full.size
            if (test and fraction >= .5) or (not test and fraction < .2) or fraction == 1:
                continue
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    if int(target[r:r + h, c:c + w].sum()) < int(target.sum()):
                        candidates.append((h * w, r, c, h, w))
    candidates.sort(reverse=not test)
    for _, r, c, h, w in candidates:
        patch = full[r:r + h, c:c + w].copy()
        if len(np.unique(patch)) == 2 and _key(patch) not in seen and len(_matches(reference, patch)) == 1:
            return patch
    raise RuntimeError("no identifying crop with the required missing target cells")


def _pack(rng, reference, crop, background, highlight, vertical):
    panels = [reference, crop]
    if rng.integers(2):
        panels.reverse()
    a, b = panels
    gap = int(rng.integers(1, 4))
    offset_a, offset_b = int(rng.integers(3)), int(rng.integers(3))
    if vertical:
        gap = min(gap, 28 - a.shape[0] - b.shape[0])
        height = 2 + a.shape[0] + gap + b.shape[0]
        width = max(offset_a + a.shape[1], offset_b + b.shape[1])
        positions = [(2, offset_a), (2 + a.shape[0] + gap, offset_b)]
    else:
        width = a.shape[1] + gap + b.shape[1]
        height = 2 + max(offset_a + a.shape[0], offset_b + b.shape[0])
        positions = [(2 + offset_a, 0), (2 + offset_b, a.shape[1] + gap)]
    height += int(rng.integers(min(2, 30 - height) + 1))
    width += int(rng.integers(min(2, 30 - width) + 1))
    grid = np.zeros((height, width), dtype=np.int8)
    grid[0, :2] = background, highlight
    for panel, (r, c) in zip(panels, positions):
        grid[r:r + panel.shape[0], c:c + panel.shape[1]] = panel
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    binary = _reference(rng)
    colors = [int(c) for c in rng.permutation(np.arange(1, 10))]
    background, highlight, ref0, ref1 = colors[:4]
    reference = np.where(binary == 0, ref0, ref1).astype(np.int8)
    blockers = colors[4:4 + int(rng.integers(1, 6))]
    # Both background phases and both rotation/reflection operations are taught
    # even at the minimum count. Novel geometry is not required in tests.
    orientations = [int(rng.integers(1, 4)), int(rng.integers(4, 8))]
    orientations += [int(rng.integers(8)) for _ in range(nt - 2)]
    phases = [0, 1] + [int(rng.integers(2)) for _ in range(nt - 2)]
    puzzle, seen_inputs, seen_crops = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            orientation = orientations[index] if split == "train" else int(rng.choice(orientations))
            phase = phases[index] if split == "train" else int(rng.integers(2))
            blocker = blockers[index % len(blockers)] if split == "train" else int(rng.choice(blockers))
            transformed = _transform(binary, orientation)
            full = np.where(transformed == phase, background, blocker).astype(np.int8)
            crop = _crop(rng, reference, full, background, split == "test", seen_crops)
            seen_crops.add(_key(crop))
            vertical = bool(index % 2) if split == "train" and index < 2 else bool(rng.integers(2))
            inp = _pack(rng, reference, crop, background, highlight, vertical)
            _require(_key(inp) not in seen_inputs, "duplicate packed input")
            seen_inputs.add(_key(inp))
            r, c, h, w = _winner(full == background)
            output = full.copy()
            output[r:r + h, c:c + w] = highlight
            puzzle[split].append({"input": inp, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, references, keys = set(), set(), set()
    train_features, train_phases, train_orientations = [], set(), set()
    train_fractions, test_fractions = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            reference, crop, background, highlight = _parse(inp)
            references.add(_key(reference))
            keys.add((background, highlight))
            orientation, swap, r, c, full = _recover(reference, crop)
            features = _features(full, background)
            target = _mask(full.shape, features["winner"])
            changed = full != output if full.shape == output.shape else np.zeros_like(full, dtype=bool)
            _require(full.shape == output.shape, "wrong completed-board shape")
            _require(np.array_equal(changed, target), "changed cells are not exactly the winning rectangle")
            _require(bool(np.all(output[target] == highlight)), "wrong highlight color")
            _require(bool(np.array_equal(output[~target], full[~target])), "cells outside winner changed")
            _require(int(target[r:r + crop.shape[0], c:c + crop.shape[1]].sum()) < int(target.sum()),
                     "query contains whole answer rectangle")
            _require(np.array_equal(solve(inp), output), "input-only solver disagrees")
            # The reference's upper/lower literal color order defines these
            # abstract phases; generation's arbitrary ref palette is immaterial.
            phase = int(background == np.unique(crop)[1 - swap])
            if split == "train":
                train_features.append(features)
                train_phases.add(phase)
                train_orientations.add(orientation)
                train_fractions.append(crop.size / full.size)
            else:
                _require(orientation in train_orientations, "test geometry lacks training support")
                test_fractions.append(crop.size / full.size)
                _require(crop.size < .5 * full.size, "test crop is not a small fragment")
    _require(len(references) == 1 and len(keys) == 1, "reference and key must stay fixed within puzzle")
    _require(train_phases == {0, 1}, "training must demonstrate both template phases as background")
    _require(any(f["overlap"] and f["long_line"] and f["connected_extra"]
                 and (f["wider_loser"] or f["taller_loser"]) for f in train_features),
             "training lacks the area, component, and one-dimensional counterexamples")
    _require(any(f > .25 for f in train_fractions), "training lacks a broad crop")


def _signature(puzzle):
    return tuple((s, _key(pair["input"]), _key(pair["output"]))
                 for s in ("train", "test") for pair in puzzle[s])


def self_test():
    """Bounded, in-memory 42-seed variation and semantic regression checks."""
    references, dimensions, crops, targets, layouts, signatures, palettes = set(), set(), set(), set(), set(), set(), set()
    area_ratios, train_ratios = [], []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        ref, _, _, _ = _parse(puzzle["train"][0]["input"])
        binary = (ref == np.unique(ref)[1]).astype(np.int8)
        references.add(min(_key(_transform(binary, i) ^ swap) for i in range(8) for swap in (0, 1)))
        dimensions.add(tuple(sorted(ref.shape)))
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                reference, crop, bg, hi = _parse(pair["input"])
                orientation, swap, _, _, full = _recover(reference, crop)
                r, c, h, w = _winner(full == bg)
                crops.add(crop.shape)
                targets.add((r, c, h, w))
                layouts.add(pair["input"].shape)
                palettes.add((bg, hi, *map(int, np.unique(crop))))
                (area_ratios if split == "test" else train_ratios).append(crop.size / full.size)
                # Drop B: plain A completion differs in at least four cells.
                assert np.count_nonzero(full != pair["output"]) == h * w >= 4
                # Drop A: processing the query alone cannot reproduce a larger
                # completed board; at least one marked cell lies outside it.
                assert crop.size < full.size
                assert not np.array_equal(_transform(binary, orientation), _transform(binary, (orientation + 1) % 8))
                if split == "train" and index < 2:
                    # An actual ablation, using the right role map but omitting
                    # A's geometric recovery, must miss both anchor answers.
                    colors = np.unique(crop)
                    canonical = np.where(reference == np.unique(reference)[1],
                                         colors[1 - swap], colors[swap]).astype(np.int8)
                    y, x, hh, ww = _winner(canonical == bg)
                    canonical[y:y + hh, x:x + ww] = hi
                    assert not np.array_equal(canonical, pair["output"])
    assert len(references) == len(signatures) == 42
    assert len(dimensions) >= 12 and len(crops) >= 25 and len(targets) >= 60
    assert len(layouts) >= 70 and len(palettes) >= 35
    assert np.mean(area_ratios) < np.mean(train_ratios)
    assert _signature(generate_puzzle(0)) == _signature(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(0, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rectangle search regressions: area, longer one-row run, and equal maxima.
    mask = np.zeros((7, 12), dtype=bool)
    mask[:3, :3] = True
    mask[:2, 3] = True
    mask[3, :2] = True
    mask[6, :] = True
    assert _winner(mask) == (0, 0, 3, 3)
    mask[:3, 7:10] = True
    try:
        _winner(mask)
    except ValueError:
        pass
    else:
        raise AssertionError("equal-area maxima accepted")
    # A valid input outside the sampling recipe: smaller reference, isolated
    # 2x2 corner winner, no eligible rival, and no long-line witness. Expected
    # cells are specified directly rather than obtained from rectangle search.
    binary = np.array([[int(c) for c in row] for row in
                       ("0011111", "0010101", "1111110",
                        "1011011", "1101111", "1110110")], dtype=np.int8)
    for height in (6, 27):
        # Extra blocker rows also test a much larger, still visibly encoded
        # reference. Pack literally so this regression does not use _pack.
        extended = np.ones((height, 7), dtype=np.int8)
        extended[:6] = binary
        reference = np.where(extended == 0, 4, 7).astype(np.int8)
        full = np.where(extended == 0, 3, 8).astype(np.int8)
        expected = full.copy()
        expected[:2, :2] = 6
        patch = full[1:5, 1:6].copy()
        for orientation in range(8):
            query = _transform(patch, orientation)
            inp = np.zeros((height + 2, 8 + query.shape[1]), dtype=np.int8)
            inp[0, :2] = 3, 6
            inp[2:, :7] = reference
            inp[2:2 + query.shape[0], 8:] = query
            before = inp.copy()
            assert np.array_equal(solve(inp), _transform(expected, orientation))
            assert np.array_equal(inp, before), "solve mutated its input"
            assert np.array_equal(solve(inp.tolist()), _transform(expected, orientation))
    # A repeated patch cannot select an arbitrary completion or hidden offset.
    try:
        _recover(reference, np.array([[3, 8], [3, 8]], dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous crop accepted")
    # Actual returned-label corruption must fail validation.
    puzzle = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations)")
