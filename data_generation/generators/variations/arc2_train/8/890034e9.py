"""Synthetic instances of the copy-a-rectangular-frame ARC rule."""

import hashlib
import json

import numpy as np


_SOURCE_PAIR_HASHES = frozenset({
    "a73e79d50739e39d7048d78afae8399e2932ec29e99da631f39444c509e93223",
    "57ecf9222ea886e576d5470b2c2cd680084777e3235d36127280fd26a7cf6295",
    "eb733973164d02db8da1c60d50779b13646ea8854dec23c88fdd659326d99a4e",
    "657afba8a3f26aa46b809557bfbde7b8845f36ce1627f543b6846695eaf80c00",
})
_TRAIN_SHAPES = (
    (2, 3), (3, 2), (3, 4), (4, 3), (3, 3), (4, 4),
    (2, 4), (4, 2), (5, 3), (3, 5), (5, 4), (4, 5), (5, 5),
)
_DIRECTIONS = ("above", "below", "left", "right", "ul", "ur", "dl", "dr")


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train, test = _count(num_train, "num_train"), _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train outside 2..10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test outside 1..4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _border_cells(box):
    r, c, h, w = box
    return ([(r, x) for x in range(c, c + w)]
            + [(r + h - 1, x) for x in range(c, c + w)]
            + [(y, c) for y in range(r + 1, r + h - 1)]
            + [(y, c + w - 1) for y in range(r + 1, r + h - 1)])


def _border_mask(shape, box):
    mask = np.zeros(shape, dtype=bool)
    for cell in _border_cells(box):
        mask[cell] = True
    return mask


def _paint_border(grid, box, color):
    for cell in _border_cells(box):
        grid[cell] = color


def _separated(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac)


def _direction(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    vertical = "above" if br + bh < ar else "below" if ar + ah < br else ""
    horizontal = "left" if bc + bw < ac else "right" if ac + aw < bc else ""
    if vertical and horizontal:
        return {("above", "left"): "ul", ("above", "right"): "ur",
                ("below", "left"): "dl", ("below", "right"): "dr"}[(vertical, horizontal)]
    return vertical or horizontal


def _canvas(rng, outer_h, outer_w, direction, preferred):
    diagonal = direction in ("ul", "ur", "dl", "dr")
    vertical = diagonal or direction in ("above", "below")
    horizontal = diagonal or direction in ("left", "right")
    min_h = max(10, 2 * outer_h + 1 if vertical else outer_h + 2)
    min_w = max(10, 2 * outer_w + 1 if horizontal else outer_w + 2)
    if preferred is not None and preferred[0] >= min_h and preferred[1] >= min_w and rng.random() < 0.38:
        return preferred
    if min_h <= 21 and min_w <= 21 and rng.random() < 0.09:
        return 21, 21
    return (int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31)))


def _place_boxes(rng, shape, outer_h, outer_w, direction):
    height, width = shape
    positions = [(r, c, outer_h, outer_w)
                 for r in range(height - outer_h + 1)
                 for c in range(width - outer_w + 1)]
    border_positions = [p for p in positions if p[0] == 0 or p[1] == 0
                        or p[0] + outer_h == height or p[1] + outer_w == width]
    border_set = set(border_positions)
    interior_positions = [p for p in positions if p not in border_set]
    interior_set = set(interior_positions)
    for _ in range(32):
        choice = rng.random()
        source_pool = (border_positions if choice < 0.13 else
                       interior_positions if choice < 0.63 and interior_positions else
                       positions)
        source = source_pool[int(rng.integers(0, len(source_pool)))]
        candidates = [p for p in positions if _direction(source, p) == direction]
        if candidates:
            placement = rng.random()
            if placement < 0.15:
                edge = [p for p in candidates if p in border_set]
                if edge:
                    candidates = edge
            elif placement < 0.65:
                interior = [p for p in candidates if p in interior_set]
                if interior:
                    candidates = interior
            distances = np.asarray([abs(p[0] - source[0]) + abs(p[1] - source[1])
                                    for p in candidates])
            order = np.argsort(distances)
            span = max(1, len(order) // 3)
            mode = int(rng.integers(0, 3))
            choices = order[:span] if mode == 0 else order[-span:] if mode == 1 else order
            target = candidates[int(choices[int(rng.integers(0, len(choices)))])]
            return source, target
    for source in positions:
        candidates = [p for p in positions if _direction(source, p) == direction]
        if candidates:
            return source, candidates[int(rng.integers(0, len(candidates)))]
    raise RuntimeError("canvas cannot fit requested placement")


def _solid_origins(grid, color, h, w):
    binary = (grid == color).astype(np.int16)
    sums = np.pad(binary, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    windows = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return {tuple(map(int, pos)) for pos in np.argwhere(windows == h * w)}


def _rect_cells(box):
    r, c, h, w = box
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _add_competitor(rng, grid, occupied, color, h, w):
    gh, gw = grid.shape
    if h > gh or w > gw:
        return set()
    positions = [(r, c, h, w) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    for index in rng.permutation(len(positions)):
        box = positions[int(index)]
        if not (_rect_cells(box) & occupied):
            r, c, _, _ = box
            grid[r:r + h, c:c + w] = color
            return _rect_cells(box)
    return set()


def _remove_extra_cores(grid, core, h, w, allowed, protected, alternatives):
    for _ in range(grid.size + 1):
        extra = _solid_origins(grid, core, h, w) - allowed
        if not extra:
            return
        r, c = min(extra)
        for y in range(r, r + h):
            for x in range(c, c + w):
                if (y, x) not in protected:
                    grid[y, x] = alternatives[(y + x) % 2]
                    break
            else:
                continue
            break
        else:
            raise RuntimeError("competing core cannot be removed")
    raise RuntimeError("core cleanup did not converge")


def _pair_palette(rng, base, shared, index):
    if shared or index == 0:
        return tuple(base)
    if rng.random() < 0.55:
        return tuple(int(x) for x in rng.permutation(base))
    retained = list(rng.choice(base, size=int(rng.integers(1, 4)), replace=False))
    remaining = [int(x) for x in rng.permutation(10) if x not in retained]
    return tuple(int(x) for x in rng.permutation(retained + remaining[:4 - len(retained)]))


def _shape_for_pair(rng, section, train_shapes):
    if section == "train":
        return _TRAIN_SHAPES[int(rng.integers(0, len(_TRAIN_SHAPES)))]
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return train_shapes[int(rng.integers(0, len(train_shapes)))]
    if mode == 1:
        return (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
    short = int(rng.integers(2, 5))
    long = int(rng.integers(6, 9))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _generate_pair(rng, core_shape, palette, preferred_shape):
    core_h, core_w = core_shape
    frame, core, noise_a, noise_b = palette
    oh, ow = core_h + 2, core_w + 2
    direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]
    shape = _canvas(rng, oh, ow, direction, preferred_shape)
    exemplar, target = _place_boxes(rng, shape, oh, ow, direction)
    core_fraction = float(rng.uniform(0.11, 0.46))
    split = float(rng.uniform(0.24, 0.76))
    grid = rng.choice(np.array([core, noise_a, noise_b], dtype=np.int8), size=shape,
                      p=[core_fraction, (1 - core_fraction) * split,
                         (1 - core_fraction) * (1 - split)]).astype(np.int8)
    er, ec, _, _ = exemplar
    tr, tc, _, _ = target
    _paint_border(grid, exemplar, frame)
    grid[er + 1:er + 1 + core_h, ec + 1:ec + 1 + core_w] = core
    grid[tr + 1:tr + 1 + core_h, tc + 1:tc + 1 + core_w] = core
    allowed = {(er + 1, ec + 1), (tr + 1, tc + 1)}
    protected = _rect_cells((er + 1, ec + 1, core_h, core_w))
    protected |= _rect_cells((tr + 1, tc + 1, core_h, core_w))
    occupied = _rect_cells(exemplar) | _rect_cells(target)
    if rng.random() < 0.62:
        if core_h != core_w and rng.random() < 0.7:
            distractor_h, distractor_w = core_w, core_h
        elif rng.random() < 0.5:
            distractor_h, distractor_w = max(1, core_h - 1), core_w
        else:
            distractor_h, distractor_w = core_h, max(1, core_w - 1)
        added = _add_competitor(rng, grid, occupied, core, distractor_h, distractor_w)
        protected |= added
        occupied |= added
    if rng.random() < 0.62:
        added = _add_competitor(rng, grid, occupied, noise_a if rng.random() < 0.5 else noise_b,
                                core_h, core_w)
        occupied |= added
    _remove_extra_cores(grid, core, core_h, core_w, allowed, protected, (noise_a, noise_b))
    # Extremely small or unusually skewed fields can omit an ambient color.
    # Resample the entire pair instead of writing a positional palette marker.
    if not {noise_a, noise_b}.issubset(set(np.unique(grid))):
        return None
    output = grid.copy()
    _paint_border(output, target, frame)
    return grid, output


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "shape changed")
    changed = input_grid != output_grid
    _require(np.any(changed), "unchanged pair")
    colors = np.unique(output_grid[changed])
    _require(len(colors) == 1, "multiple new frame colors")
    frame = int(colors[0])
    _require(not np.any(input_grid[changed] == frame), "target partly framed")
    locations = np.argwhere(changed)
    top, left = locations.min(0)
    bottom, right = locations.max(0)
    target = (int(top), int(left), int(bottom - top + 1), int(right - left + 1))
    _require(target[2] >= 3 and target[3] >= 3, "empty core")
    _require(np.array_equal(changed, _border_mask(input_grid.shape, target)), "change is not a complete border")
    frame_cells = np.argwhere(input_grid == frame)
    _require(frame_cells.size > 0, "no exemplar")
    etop, eleft = frame_cells.min(0)
    ebottom, eright = frame_cells.max(0)
    exemplar = (int(etop), int(eleft), int(ebottom - etop + 1), int(eright - eleft + 1))
    _require(exemplar[2:] == target[2:], "mismatched frame sizes")
    _require(np.array_equal(input_grid == frame, _border_mask(input_grid.shape, exemplar)), "frame is not unique rectangle")
    _require(_separated(exemplar, target), "frames overlap or adjoin")
    eh, ew = exemplar[2] - 2, exemplar[3] - 2
    exemplar_core = input_grid[etop + 1:etop + 1 + eh, eleft + 1:eleft + 1 + ew]
    target_core = input_grid[top + 1:top + 1 + eh, left + 1:left + 1 + ew]
    core = int(exemplar_core[0, 0])
    _require(core != frame and np.all(exemplar_core == core) and np.all(target_core == core), "cores disagree")
    _require(_solid_origins(input_grid, core, eh, ew) == {(int(etop + 1), int(eleft + 1)),
                                                           (int(top + 1), int(left + 1))}, "ambiguous core")
    _require(np.array_equal(output_grid == frame,
                            _border_mask(input_grid.shape, exemplar) | changed), "wrong output frame")
    _require(len(np.unique(input_grid)) == 4 and np.array_equal(np.unique(input_grid), np.unique(output_grid)), "palette error")
    return {"frame": frame, "core": core, "palette": tuple(map(int, np.unique(input_grid))),
            "core_shape": (eh, ew), "exemplar": exemplar, "target": target,
            "direction": _direction(exemplar, target),
            "perimeter_core_cells": sum(input_grid[cell] == core for cell in _border_cells(target))}


def _pair_hash(pair):
    data = {key: pair[key].tolist() for key in ("input", "output")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "bad pair counts")
    seen = set()
    information = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "bad grid")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "bad dimensions")
                _require(np.all((grid >= 0) & (grid <= 9)), "illegal color")
            key = _grid_key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_pair_hash(pair) not in _SOURCE_PAIR_HASHES, "copied source pair")
            information[section].append(_analyze_pair(pair["input"], pair["output"]))
    _require(any(i["core_shape"][0] < i["core_shape"][1] for i in information["train"]), "training lacks wide core")
    _require(any(i["core_shape"][0] > i["core_shape"][1] for i in information["train"]), "training lacks tall core")
    _require(all(i["core_shape"] in _TRAIN_SHAPES for i in information["train"]), "invalid training shape")
    _require(all(2 <= min(i["core_shape"]) and max(i["core_shape"]) <= 8 for i in information["test"]), "invalid test shape")
    # The invariant is local: every input carries its own complete stencil.
    # Thus pair palettes can change without introducing a hidden convention.


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(int(x) for x in rng.permutation(10)[:4])
    shared_palette = bool(rng.random() < 0.18)
    first_is_wide = bool(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    train_shapes = []
    previous_canvas = None
    pair_index = 0
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index < 2:
                wide = first_is_wide if index == 0 else not first_is_wide
                candidates = [s for s in _TRAIN_SHAPES if (s[0] < s[1]) == wide and s[0] != s[1]]
                core_shape = candidates[int(rng.integers(0, len(candidates)))]
            else:
                core_shape = _shape_for_pair(rng, section, train_shapes)
            palette = _pair_palette(rng, base, shared_palette, pair_index)
            for _ in range(12):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                result = _generate_pair(child_rng, core_shape, palette, previous_canvas)
                if result is not None:
                    break
            else:
                raise RuntimeError("failed to generate a palette-complete pair")
            input_grid, output_grid = result
            puzzle[section].append({"input": input_grid, "output": output_grid})
            previous_canvas = input_grid.shape
            if section == "train":
                train_shapes.append(core_shape)
            pair_index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[section]) == len(b[section]) and all(
        np.array_equal(pa[key], pb[key])
        for pa, pb in zip(a[section], b[section]) for key in ("input", "output"))
        for section in ("train", "test"))


def _visible_competitors(grid, info):
    h, w = info["core_shape"]
    occupied = _rect_cells(info["exemplar"]) | _rect_cells(info["target"])

    def separate_window(origin, height, width):
        return not (_rect_cells((origin[0], origin[1], height, width)) & occupied)

    transposed = (h != w and any(
        separate_window(origin, w, h)
        for origin in _solid_origins(grid, info["core"], w, h)))
    other_color = any(
        separate_window(origin, h, w)
        for color in info["palette"] if color not in (info["core"], info["frame"])
        for origin in _solid_origins(grid, color, h, w))
    return transposed, other_color


def _predict_from_demonstrations(train, test_input):
    # Derive the operation from changed training cells, then inspect only test input.
    for pair in train:
        _analyze_pair(pair["input"], pair["output"])
    candidates = []
    for color in np.unique(test_input):
        cells = np.argwhere(test_input == color)
        top, left = cells.min(0)
        bottom, right = cells.max(0)
        box = (int(top), int(left), int(bottom - top + 1), int(right - left + 1))
        if box[2] < 3 or box[3] < 3 or not np.array_equal(test_input == color, _border_mask(test_input.shape, box)):
            continue
        h, w = box[2] - 2, box[3] - 2
        core = int(test_input[top + 1, left + 1])
        if not np.all(test_input[top + 1:bottom, left + 1:right] == core):
            continue
        origins = _solid_origins(test_input, core, h, w)
        exemplar_origin = (int(top + 1), int(left + 1))
        if exemplar_origin not in origins or len(origins) != 2:
            continue
        target_origin = next(iter(origins - {exemplar_origin}))
        output = test_input.copy()
        _paint_border(output, (target_origin[0] - 1, target_origin[1] - 1, h + 2, w + 2), int(color))
        candidates.append(output)
    _require(len(candidates) == 1, "test input has no unique visible stencil")
    return candidates[0]


def self_test() -> None:
    example = generate_puzzle(12345)
    validate_generated_puzzle(example)
    explicit = generate_puzzle(12345, 6, 3)
    _require(len(explicit["train"]) == 6 and len(explicit["test"]) == 3, "explicit counts ignored")
    _require(_same_puzzle(explicit, generate_puzzle(12345, 6, 3)), "nondeterministic")
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + offset, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "partial train ignored")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "partial test ignored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, roles, sizes, layouts, shapes, directions = (set() for _ in range(7))
    mixed_roles = mixed_palettes = has_moat_core = has_21 = edge = repeated_size = 0
    marker_equal = transposed = other_color = low_core = high_core = 0
    repeated_test_shape = larger_test_shape = square_test_shape = 0
    close_layout = far_layout = 0
    all_inputs = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = []
        pair_sizes = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                infos.append(info)
                roles.add((info["frame"], info["core"]))
                sizes.add(pair["input"].shape)
                pair_sizes.append(pair["input"].shape)
                shapes.add(info["core_shape"])
                directions.add(info["direction"])
                layouts.add((pair["input"].shape, info["core_shape"], info["exemplar"], info["target"]))
                all_inputs.add(_grid_key(pair["input"]))
                has_moat_core += info["perimeter_core_cells"] > 0
                has_21 += pair["input"].shape == (21, 21)
                target_row, target_col, _, _ = info["target"]
                marker_equal += [int(pair["input"][target_row, target_col]),
                                 int(pair["input"][target_row, target_col + 1])] == [
                                     int(pair["input"][0, 0]), int(pair["input"][0, 1])]
                has_wrong, has_other = _visible_competitors(pair["input"], info)
                transposed += has_wrong
                other_color += has_other
                outside = np.ones(pair["input"].shape, dtype=bool)
                for box in (info["exemplar"], info["target"]):
                    r, c, h, w = box
                    outside[r:r + h, c:c + w] = False
                ambient_fraction = np.mean(pair["input"][outside] == info["core"])
                low_core += ambient_fraction < 0.20
                high_core += ambient_fraction > 0.35
                distance = abs(target_row - info["exemplar"][0]) + abs(target_col - info["exemplar"][1])
                close_layout += distance <= 10
                far_layout += distance >= 18
                edge += any(box[0] == 0 or box[1] == 0 or box[0] + box[2] == pair["input"].shape[0]
                            or box[1] + box[3] == pair["input"].shape[1]
                            for box in (info["exemplar"], info["target"]))
        mixed_roles += len({(i["frame"], i["core"]) for i in infos}) > 1
        mixed_palettes += len({i["palette"] for i in infos}) > 1
        repeated_size += len(set(pair_sizes)) < len(pair_sizes)
        training_shapes = {i["core_shape"] for i in infos[:len(puzzle["train"])]}
        longest_training_side = max(max(shape) for shape in training_shapes)
        for info in infos[len(puzzle["train"]):]:
            repeated_test_shape += info["core_shape"] in training_shapes
            larger_test_shape += max(info["core_shape"]) > longest_training_side
            square_test_shape += info["core_shape"][0] == info["core_shape"][1]
        for pair in puzzle["test"]:
            predicted = _predict_from_demonstrations(
                [{"input": p["input"].copy(), "output": p["output"].copy()} for p in puzzle["train"]],
                pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]), "test is not inferred from visible evidence")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(roles) >= 30 and len(sizes) >= 45 and len(layouts) >= 200, "structural diversity")
    _require(len(shapes) >= 15 and len(directions) == 8, "shape or direction diversity")
    _require(len(all_inputs) >= 300, "input diversity")
    _require(mixed_roles >= 15 and mixed_palettes >= 15, "insufficient within-puzzle color variation")
    _require(has_moat_core >= 100 and has_21 >= 5 and edge >= 70 and repeated_size >= 8,
             "perimeter, size, or edge variation collapsed")
    _require(marker_equal < len(all_inputs) // 2 and transposed >= 30 and other_color >= 60,
             "markers or visual competitors collapsed")
    _require(low_core >= 20 and high_core >= 20 and close_layout >= 20 and far_layout >= 20,
             "ambient density or layout distance collapsed")
    _require(repeated_test_shape >= 5 and larger_test_shape >= 5 and square_test_shape >= 2,
             "test shape regimes collapsed")
