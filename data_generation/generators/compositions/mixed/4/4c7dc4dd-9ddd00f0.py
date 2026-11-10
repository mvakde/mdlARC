"""Analogy-transferred coordinate-hole codebooks.

The complete reference pair selects a geometric coordinate map independently
for every input. Only target compartments that are entirely blank are repaired.
No evidence files, saved examples, random state, or training labels are read by
solve. Diagnostics keep summaries/hashes only and write no files.
"""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np


# (swap axes, reverse resulting row axis, reverse resulting column axis).
MAPS = tuple(product((False, True), repeat=3))
IDENTITY = (False, False, False)
SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _side(n):
    return n * n + n - 1


def _order(side):
    found = [n for n in range(2, 5) if _side(n) == side]
    _require(len(found) == 1, "not a coordinate-lattice dimension")
    return found[0]


def _maps_for(shape):
    return MAPS if shape[0] == shape[1] else tuple(m for m in MAPS if not m[0])


def _address(row, col, rows, cols, mapping):
    swap, reverse_row, reverse_col = mapping
    if swap:
        _require(rows == cols, "axis exchange requires square compartments")
        row, col = col, row
    return (rows - 1 - row if reverse_row else row,
            cols - 1 - col if reverse_col else col)


def _tile(a, row, col, rows, cols):
    return a[row * (rows + 1):row * (rows + 1) + rows,
             col * (cols + 1):col * (cols + 1) + cols]


def _complete(shape, background, foreground, mapping):
    rows, cols = shape
    a = np.full((_side(rows), _side(cols)), background, dtype=np.int8)
    for row, col in product(range(rows), range(cols)):
        block = _tile(a, row, col, rows, cols)
        block[:] = foreground
        block[_address(row, col, rows, cols, mapping)] = background
    return a


def _erase(full, mask, background):
    a = full.copy()
    rows, cols = mask.shape
    for row, col in product(range(rows), range(cols)):
        if not mask[row, col]:
            _tile(a, row, col, rows, cols)[:] = background
    return a


def _read_lattice(a, background, complete=False):
    rows, cols = _order(a.shape[0]), _order(a.shape[1])
    _require(all(np.all(a[r] == background) for r in range(rows, a.shape[0], rows + 1)),
             "damaged horizontal separator")
    _require(all(np.all(a[:, c] == background) for c in range(cols, a.shape[1], cols + 1)),
             "damaged vertical separator")
    colors = set(map(int, np.unique(a))) - {background}
    _require(len(colors) == 1, "a lattice must reveal one foreground color")
    foreground = next(iter(colors))
    mask = np.zeros((rows, cols), dtype=bool)
    holes = {}
    for row, col in product(range(rows), range(cols)):
        block = _tile(a, row, col, rows, cols)
        if np.all(block == background):
            continue
        positions = np.argwhere(block == background)
        _require(len(positions) == 1 and np.count_nonzero(block == foreground) == rows * cols - 1,
                 "visible compartment must have exactly one background hole")
        mask[row, col] = True
        holes[row, col] = tuple(map(int, positions[0]))
    _require(not complete or mask.all(), "reference lattices must be complete")
    return dict(shape=(rows, cols), foreground=foreground, mask=mask, holes=holes)


def _consistent_maps(info):
    rows, cols = info["shape"]
    return [m for m in _maps_for(info["shape"])
            if all(_address(r, c, rows, cols, m) == hole
                   for (r, c), hole in info["holes"].items())]


def _components(a, color):
    seen = np.zeros(a.shape, dtype=bool)
    height, width = a.shape
    for rr, cc in np.argwhere(a == color):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, points = [(r, c)], []
        while todo:
            y, x = todo.pop()
            points.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < height and 0 <= xx < width and
                        not seen[yy, xx] and a[yy, xx] == color):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        yield points


def _frames(a):
    """Frame role is geometric: three isolated hollow rectangular outlines."""
    candidates = []
    legal = {_side(n) + 2 for n in range(2, 5)}
    for color in map(int, np.unique(a)):
        boxes = []
        for points in _components(a, color):
            ys, xs = zip(*points)
            r, c = min(ys), min(xs)
            h, w = max(ys) - r + 1, max(xs) - c + 1
            if h not in legal or w not in legal or len(points) != 2 * h + 2 * w - 4:
                continue
            block = a[r:r + h, c:c + w]
            if (np.all(block[0] == color) and np.all(block[-1] == color) and
                    np.all(block[:, 0] == color) and np.all(block[:, -1] == color) and
                    np.all(block[1:-1, 1:-1] != color)):
                boxes.append((r, c, h, w))
        if len(boxes) == 3:
            candidates.append((color, boxes))
    _require(len(candidates) == 1, "expected exactly three unambiguous lattice frames")
    return candidates[0]


def _crop(a, box):
    r, c, h, w = box
    return a[r + 1:r + h - 1, c + 1:c + w - 1]


def _arrow_mask(height, right):
    mask = np.zeros((height, 3), dtype=bool)
    mid = height // 2
    for dr, col in ((-2, 0), (-1, 1), (0, 2), (1, 1), (2, 0)):
        mask[mid + dr, col] = True
    return mask if right else mask[:, ::-1]


def _extract(input_grid):
    a = _grid(input_grid)
    frame_color, frames = _frames(a)
    scenes = []
    for transposed in (False, True):
        board = a.T if transposed else a
        boxes = [(c, r, w, h) for r, c, h, w in frames] if transposed else frames
        ordered = sorted(boxes)
        left, right, target = ordered
        # The complete pair is the aligned pair above the single target card.
        if not (left[0] == right[0] and left[2:] == right[2:] and
                left[1] + left[3] + 3 == right[1] and
                target[0] >= left[0] + left[2] + 1):
            continue
        r, c, h, w = left
        if h != w or _order(h - 2) not in (2, 3):
            continue
        gap = board[r:r + h, c + w:c + w + 3]
        # Background is the common separator role, not a literal color.
        source_left, source_right, partial = [_crop(board, b) for b in (left, right, target)]
        n = _order(h - 2)
        background = int(source_left[n, 0])
        for rightward in (True, False):
            marker = _arrow_mask(h, rightward)
            arrow_colors = set(map(int, np.unique(gap[marker])))
            if not (len(arrow_colors) == 1 and background not in arrow_colors and
                    frame_color not in arrow_colors and np.all(gap[~marker] == background)):
                continue
            source, result = ((source_left, source_right) if rightward else (source_right, source_left))
            source_info = _read_lattice(source, background, complete=True)
            result_info = _read_lattice(result, background, complete=True)
            _require(_consistent_maps(source_info) == [IDENTITY],
                     "reference source must show canonical addresses")
            maps = _consistent_maps(result_info)
            _require(len(maps) == 1, "reference does not select a unique coordinate map")
            target_info = _read_lattice(partial, background)
            mapping = maps[0]
            _require(mapping in _maps_for(target_info["shape"]),
                     "axis exchange cannot be transferred to a rectangle")
            _require(mapping in _consistent_maps(target_info),
                     "target clues disagree with the reference map")
            _require(0 < target_info["mask"].sum() < target_info["mask"].size,
                     "target needs a visible compartment and a blank compartment")
            role_colors = {source_info["foreground"], result_info["foreground"],
                           target_info["foreground"], background}
            _require(not (role_colors & (arrow_colors | {frame_color})),
                     "marker and frame roles must be distinct from lattice roles")
            # No hidden content outside the cards and the demonstrated arrow.
            outside = np.ones(board.shape, dtype=bool)
            for y, x, hh, ww in (left, right, target):
                outside[y:y + hh, x:x + ww] = False
            outside[r:r + h, c + w:c + w + 3] = False
            _require(np.all(board[outside] == background), "unexpected exterior content")
            scenes.append(dict(board=board, transposed=transposed, background=background,
                               frame_color=frame_color, arrow_color=next(iter(arrow_colors)),
                               rightward=rightward, source=source.copy(), reference=result.copy(),
                               source_box=left if rightward else right,
                               reference_box=right if rightward else left,
                               target_box=target, target=partial.copy(), mapping=mapping,
                               reference_order=n, **target_info))
    _require(len(scenes) == 1, "input must have one directed reference pair and one target")
    return scenes[0]


def _reconstruct(scene, mapping=None):
    mapping = scene["mapping"] if mapping is None else mapping
    answer = scene["target"].copy()
    rows, cols = scene["shape"]
    for row, col in product(range(rows), range(cols)):
        if not scene["mask"][row, col]:
            block = _tile(answer, row, col, rows, cols)
            block[:] = scene["foreground"]
            block[_address(row, col, rows, cols, mapping)] = scene["background"]
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the local codebook and return only the repaired target interior."""
    scene = _extract(input_grid)
    answer = _reconstruct(scene)
    return answer.T.copy() if scene["transposed"] else answer


def _draw_frame(board, box, color, interior):
    r, c, h, w = box
    board[r:r + h, c:c + w] = color
    board[r + 1:r + h - 1, c + 1:c + w - 1] = interior


def _mask(rng, shape, dense=False, center_only=False):
    """A reflection-fixed clue set deliberately leaves a residual ambiguity."""
    rows, cols = shape
    points = []
    if rows == cols:
        axes = ["diagonal", "anti"] + (["row", "col"] if rows % 2 else [])
        axis = str(rng.choice(axes))
        if center_only:
            _require(rows % 2 == 1, "center-only mask needs an odd square")
            points = [(rows // 2, cols // 2)]
        elif axis == "diagonal":
            points = [(i, i) for i in range(rows)]
        elif axis == "anti":
            points = [(i, rows - 1 - i) for i in range(rows)]
        elif axis == "row":
            points = [(rows // 2, j) for j in range(cols)]
        else:
            points = [(i, cols // 2) for i in range(rows)]
    elif rows % 2 and (not cols % 2 or rng.integers(2)):
        points = [(rows // 2, j) for j in range(cols)]
    else:
        _require(cols % 2 == 1, "rectangle needs an odd axis for ambiguous clues")
        points = [(i, cols // 2) for i in range(rows)]
    count = 1 if center_only else int(rng.integers(2 if dense else 1, len(points) + 1))
    result = np.zeros(shape, dtype=bool)
    for i in rng.choice(len(points), count, replace=False):
        result[points[int(i)]] = True
    return result


def _render(rng, shape, mask, mapping, reference_order=None):
    rows, cols = shape
    options = [n for n in (2, 3) if
               (_side(n) + 2) + 1 + (_side(rows) + 2) <= 30 and
               2 * (_side(n) + 2) + 3 <= 30]
    n = int(rng.choice(options)) if reference_order is None else reference_order
    _require(n in options, "reference and target do not fit")
    colors = [int(x) for x in rng.permutation(10)]
    background, frame, arrow = colors[:3]
    palette_mode = int(rng.integers(3))
    source_fg = colors[3]
    result_fg = source_fg if palette_mode < 2 else colors[4]
    target_fg = source_fg if palette_mode == 0 else colors[5]
    source = _complete((n, n), background, source_fg, IDENTITY)
    reference = _complete((n, n), background, result_fg, mapping)
    full = _complete(shape, background, target_fg, mapping)
    partial = _erase(full, mask, background)
    side, th, tw = _side(n) + 2, _side(rows) + 2, _side(cols) + 2
    min_h, min_w = side + 1 + th, max(2 * side + 3, tw)
    height, width = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    top_slack = height - min_h
    top = int(rng.integers(top_slack + 1))
    extra_gap = int(rng.integers(top_slack - top + 1))
    left = int(rng.integers(width - (2 * side + 3) + 1))
    target_left = int(rng.integers(width - tw + 1))
    left_box = (top, left, side, side)
    right_box = (top, left + side + 3, side, side)
    target_box = (top + side + 1 + extra_gap, target_left, th, tw)
    rightward = bool(rng.integers(2))
    board = np.full((height, width), background, dtype=np.int8)
    _draw_frame(board, left_box, frame, source if rightward else reference)
    _draw_frame(board, right_box, frame, reference if rightward else source)
    _draw_frame(board, target_box, frame, partial)
    gap = board[top:top + side, left + side:left + side + 3]
    gap[_arrow_mask(side, rightward)] = arrow
    if rng.integers(2):
        board, full = board.T.copy(), full.T.copy()
    return {"input": board, "output": full}


def _reference_twin(pair, mapping):
    """Change only the informative reference result; keep the whole target fixed."""
    scene = _extract(pair["input"])
    board = scene["board"].copy()
    n = scene["reference_order"]
    foreground = next(int(x) for x in np.unique(scene["reference"])
                      if int(x) != scene["background"])
    reference = _complete((n, n), scene["background"], foreground, mapping)
    _draw_frame(board, scene["reference_box"], scene["frame_color"], reference)
    answer = _reconstruct(scene, mapping)
    if scene["transposed"]:
        board, answer = board.T.copy(), answer.T.copy()
    return {"input": board, "output": answer}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(4))
    base_shape = ((3, 3) if regime in (0, 2) else
                  ((3, 2), (2, 3))[int(rng.integers(2))] if regime == 1 else (4, 4))
    mask = _mask(rng, base_shape, dense=True)
    first_map = _maps_for(base_shape)[int(rng.integers(len(_maps_for(base_shape))))]
    first = _render(rng, base_shape, mask, first_map)
    info = _extract(first["input"])
    alternatives = [m for m in _consistent_maps(info) if m != first_map]
    _require(alternatives, "target unexpectedly determines its map without the reference")
    second_map = alternatives[int(rng.integers(len(alternatives)))]
    second = _reference_twin(first, second_map)
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", a - 2), ("test", b)):
        for _ in range(count):
            for attempt in range(24):
                if split == "train":
                    pool = ((2, 3), (3, 2), (3, 3)) if regime == 1 else ((2, 2), (3, 3), (4, 4))
                    shape = pool[int(rng.integers(len(pool)))]
                elif regime == 0:
                    shape = (4, 4)
                elif regime == 1:
                    shape = ((3, 4), (4, 3))[int(rng.integers(2))]
                elif regime == 2:
                    shape = (3, 3)
                else:
                    shape = ((2, 2), (3, 3))[int(rng.integers(2))]
                maps = [m for m in _maps_for(shape) if split == "train" or m not in (first_map, second_map)]
                mapping = maps[int(rng.integers(len(maps)))]
                mask = _mask(rng, shape, center_only=(split == "test" and regime == 2))
                pair = _render(rng, shape, mask, mapping)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded search could not make distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _extract(inp)
            normalized_out = out.T if scene["transposed"] else out
            _require(normalized_out.shape == scene["target"].shape, "output must be target interior only")
            expected = _complete(scene["shape"], scene["background"], scene["foreground"], scene["mapping"])
            _require(np.array_equal(normalized_out, expected), "wrong coordinate-hole completion")
            _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
            rows, cols = scene["shape"]
            for r, c in product(range(rows), range(cols)):
                if scene["mask"][r, c]:
                    _require(np.array_equal(_tile(scene["target"], r, c, rows, cols),
                                            _tile(normalized_out, r, c, rows, cols)),
                             "a completed compartment was changed")
            # Parent A is necessary: a target-only map has multiple answers.
            candidates = _consistent_maps(scene)
            predictions = {_key(_reconstruct(scene, m)) for m in candidates}
            _require(len(predictions) >= 2, "reference codebook is redundant")
            # Parent B is necessary: solid filling, or repeating any one hole,
            # cannot agree with all of the differently addressed missing tiles.
            missing_holes = {_address(r, c, rows, cols, scene["mapping"])
                             for r, c in product(range(rows), range(cols)) if not scene["mask"][r, c]}
            _require(len(missing_holes) >= 2, "missing compartments do not expose distinct addresses")
            _require(not np.array_equal(scene["target"], normalized_out), "no completion required")
            if split == "train":
                training.append(scene)
    # The first two demonstrations are controlled reference-dependence twins.
    s, t = training[:2]
    _require(s["transposed"] == t["transposed"] and s["reference_box"] == t["reference_box"],
             "training twins must share their layout")
    _require(np.array_equal(s["target"], t["target"]) and s["mapping"] != t["mapping"],
             "training twins must keep the target and change its codebook")
    r, c, h, w = s["reference_box"]
    exterior = np.ones(s["board"].shape, dtype=bool)
    exterior[r + 1:r + h - 1, c + 1:c + w - 1] = False
    _require(s["board"].shape == t["board"].shape and
             np.array_equal(s["board"][exterior], t["board"][exterior]),
             "twins differ outside the informative reference interior")
    _require(not np.array_equal(_reconstruct(s), _reconstruct(t)), "twins must change the answer")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed checks, small mutation/metamorphic probes, no disk writes."""
    shapes, maps, masks, palettes, inputs, outputs, hashes = (set() for _ in range(7))
    reference_orders, directions, orientations = set(), set(), set()
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                scene = _extract(pair["input"])
                shapes.add(scene["shape"])
                maps.add(scene["mapping"])
                masks.add((scene["shape"], scene["mask"].tobytes()))
                palettes.add((scene["background"], scene["foreground"], scene["frame_color"]))
                inputs.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                reference_orders.add(scene["reference_order"])
                directions.add(scene["rightward"])
                orientations.add(scene["transposed"])
    _require(len(hashes) == 42 and len(masks) >= 35 and len(palettes) >= 60,
             "insufficient structural/palette variation across 42 seeds")
    _require(shapes == set(SHAPES) and maps == set(MAPS) and len(inputs) >= 20,
             "shape or coordinate-map variation missing")
    _require(reference_orders == {2, 3} and directions == {False, True}
             and orientations == {False, True}, "reference/layout variation missing")
    _require(_digest(generate_puzzle(817)) == _digest(generate_puzzle(817)), "seed is not reproducible")
    # Support every legal explicit count; retain only one bounded puzzle at once.
    for a in range(2, 11):
        for b in range(1, min(a - 1, 4) + 1):
            puzzle = generate_puzzle(9000 + a * 10 + b, a, b)
            validate_generated_puzzle(puzzle)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    pair = generate_puzzle(217)["test"][0]
    inp, out = pair["input"], pair["output"]
    before = inp.copy()
    _require(np.array_equal(solve(inp), out) and np.array_equal(inp, before), "solve mutates input")
    _require(np.array_equal(solve(inp.T), out.T), "transpose equivariance failed")
    # Literal colors have no special semantic role.
    permutation = np.random.default_rng(121).permutation(10).astype(np.int8)
    _require(np.array_equal(solve(permutation[inp]), permutation[out]), "palette equivariance failed")
    _require(np.array_equal(solve(inp.tolist()), out), "nested-list input failed")
    bad = generate_puzzle(3, 2, 1)
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    # Solve permits other valid masks, beyond the deliberately ambiguous sampler.
    scene = _extract(inp)
    full = _complete(scene["shape"], scene["background"], scene["foreground"], scene["mapping"])
    for mask_mode in ("near_full", "one_clue", "empty_row"):
        mask = np.ones(scene["shape"], dtype=bool)
        if mask_mode == "near_full":
            mask[-1, -1] = False
        elif mask_mode == "one_clue":
            mask[:] = False
            mask[0, 0] = True
        else:
            mask[0] = False
        board = scene["board"].copy()
        _draw_frame(board, scene["target_box"], scene["frame_color"], _erase(full, mask, scene["background"]))
        probe = board.T if scene["transposed"] else board
        expected = full.T if scene["transposed"] else full
        _require(np.array_equal(solve(probe), expected), "input-only solver overfits generated masks")
    return {"seeds": 42, "pairs": total_pairs, "distinct_puzzles": len(hashes),
            "target_shapes": len(shapes), "coordinate_maps": len(maps),
            "clue_masks": len(masks), "role_palettes": len(palettes),
            "input_shapes": len(inputs), "output_shapes": len(outputs),
            "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
