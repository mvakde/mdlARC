"""Erase unmatched additions from independently mirrored monochrome figures.

The source is vertical. Horizontal puzzles are the review's explicit extension,
made by quarter-turning a whole puzzle. No mixed-orientation or tie-break rule is
introduced. In particular, source test 1's green figure is unresolved under
maximum support (axes 18 and 18.5 tie); generated figures never have such ties.
"""
import hashlib
import numpy as np


# Exact source-input content fingerprints, including dimensions. These exclude
# copies without excluding any source size, palette, color role, or silhouette.
_SOURCE_INPUTS = frozenset((
    "65a4d21be2e9796277b84e00d5fe3b4f7903a1d3146c749db75ed8700cc6be15",
    "8df7a91d967601bcb3ba137d14497238a77cfc07464dd422862ae11975ec5fbc",
    "e637070517b9ee2679eeb2379d980d95d6bfd0270058c84471f516cc37247f29",
    "cc2fb9f42c3386b22a7030d9ffaf6704d5f0cc2db114af488ee3306978aa909b",
))


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, grid.astype(np.uint8).tobytes())


def _source_copy(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUTS


def _mode_color(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(counts.argmax())
    assert counts[background] > grid.size / 2, "background must occupy a majority"
    return background


def _cells(grid, color):
    return set(map(tuple, np.argwhere(grid == color).tolist()))


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _best_vertical_axis(cells, width):
    """Construction check, using explicit reflected coordinate sets."""
    candidates = [(axis, {p for p in cells if (p[0], axis - p[1]) in cells})
                  for axis in range(2 * width - 1)]
    maximum = max(len(supported) for _, supported in candidates)
    winners = [(axis, supported) for axis, supported in candidates if len(supported) == maximum]
    assert len(winners) == 1, "ambiguous reflection axis"
    return winners[0]


def _erase_vertical(grid):
    """Input-only prediction, using row convolutions independently of construction."""
    bg = _mode_color(grid)
    output = np.full_like(grid, bg)
    axes = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = (grid == color).astype(np.int16)
        scores = sum((np.convolve(row, row) for row in mask), np.zeros(2 * grid.shape[1] - 1, dtype=int))
        winners = np.flatnonzero(scores == scores.max())
        assert len(winners) == 1, "ambiguous maximum support"
        axis = int(winners[0])
        reflected = axis - np.arange(grid.shape[1])
        valid = (reflected >= 0) & (reflected < grid.shape[1])
        keep = np.zeros_like(mask, dtype=bool)
        keep[:, valid] = (mask[:, valid] != 0) & (mask[:, reflected[valid]] != 0)
        output[keep] = color
        axes[color] = axis
    return output, axes


def _prediction(grid, orientation):
    work = grid if orientation == 0 else grid.T
    out, axes = _erase_vertical(work)
    return (out if orientation == 0 else out.T), axes


def _fits(pair, orientation):
    try:
        output, _ = _prediction(pair["input"], orientation)
        return np.array_equal(output, pair["output"])
    except AssertionError:
        return False


def _learn_orientation(train_pairs):
    choices = [orientation for orientation in (0, 1)
               if all(_fits(pair, orientation) for pair in train_pairs)]
    assert len(choices) == 1, "training must uniquely demonstrate the orientation"
    assert any(not np.array_equal(p["input"], p["output"]) for p in train_pairs)
    return choices[0]


def solve_from_training(train_pairs, test_input):
    """No test label, seed, or generator state is used to choose an answer."""
    return _prediction(test_input, _learn_orientation(train_pairs))[0]


def _make_symmetric_mask(rng, height, width):
    """Compositional silhouettes: contours, profiles, necks, stems and lobes."""
    radius = width // 2
    odd = width % 2
    family = int(rng.integers(0, 7))
    mask = np.zeros((height, width), dtype=bool)
    if family == 0:
        # A variable-thickness hollow contour, including the 3x3 hollow square.
        side = int(rng.integers(1, max(2, radius // 2 + 1)))
        cap = int(rng.integers(1, max(2, height // 3 + 1)))
        mask[:cap] = True
        mask[-cap:] = True
        mask[:, :side] = True
        mask[:, -side:] = True
    else:
        levels = []
        center = float(rng.uniform(0, max(1, height - 1)))
        period = float(rng.uniform(2, max(3, height)))
        level = int(rng.integers(0 if odd else 1, radius + 1))
        for row in range(height):
            if family == 1:  # diamond/oval, with freely shifted widest section
                distance = abs(row - center) / max(center, height - 1 - center, 1)
                level = int(round(radius * (1 - distance)))
            elif family == 2:  # hourglass, independently thickened neck
                distance = abs(row - center) / max(center, height - 1 - center, 1)
                level = int(round(radius * distance))
            elif family == 3:  # several lobes along a thin stem
                level = int(round(radius * (0.5 + 0.5 * np.cos(row * 2 * np.pi / period + center))))
            elif family == 4:  # correlated irregular silhouette
                level = int(np.clip(level + rng.integers(-2, 3), 0, radius))
            elif family == 5:  # independent branches, short necks and bars
                level = int(rng.integers(0, radius + 1))
            else:  # pointed contour, six-cell 4x3 examples are possible
                level = 0 if row in (0, height - 1) else int(rng.integers(1, radius + 1))
            levels.append(max(0 if odd else 1, level))
        # Width is attained somewhere, with no prescribed middle-row extremity.
        levels[int(rng.integers(height))] = radius
        hollow = rng.random() < 0.48 or family == 6
        fill_probability = float(rng.uniform(0.35, 1.0))
        for row, level in enumerate(levels):
            thickness = int(rng.integers(1, max(2, level + 1)))
            for distance in range(1, level + 1):
                present = distance > level - thickness if hollow else rng.random() < fill_probability
                if distance == level:
                    present = True
                if present:
                    left = (width - 1) // 2 - distance + (1 - odd)
                    mask[row, left] = mask[row, width - 1 - left] = True
            if odd:
                mask[row, radius] = level == 0 or (not hollow and rng.random() < 0.8)
        # Gaps make supported islands possible. They are never removed merely
        # because they are disconnected; foreground grouping is by color.
        if height >= 6 and rng.random() < 0.22:
            row = int(rng.integers(1, height - 1))
            mask[row] = False
        if odd and height >= 5 and rng.random() < 0.15:
            mask[1] = False
            mask[0] = False
            mask[0, radius] = True
    rows, cols = np.nonzero(mask)
    mask = mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return mask


def _sample_dimensions(rng, height, width, scale, parity):
    # Scale probabilities remain overlapping, so split never determines size.
    category = int(rng.choice(3, p=((.68, .27, .05), (.30, .43, .27), (.13, .32, .55))[scale]))
    low, high = ((2, 7), (4, 11), (7, 19))[category]
    h = int(rng.integers(low, min(height, high) + 1))
    widths = [w for w in range(max(2, low), min(width, high) + 1) if (w - 1) % 2 == parity]
    if not widths:
        widths = [3 if parity == 0 else 2]
    w = int(rng.choice(widths))
    # Shape aspect is independent of canvas aspect.
    if rng.random() < 0.30:
        if rng.random() < 0.5:
            h = int(rng.integers(2, min(4, height) + 1))
        else:
            w = int(rng.choice([v for v in range(2, min(5, width) + 1) if (v - 1) % 2 == parity]))
    return h, w


def _neighbors(cells, diagonal=False):
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    return {(r + dr, c + dc) for r, c in cells for dr, dc in offsets} - cells


def _place_cores(rng, shape, count, parities, scale):
    height, width = shape
    occupied = set()
    figures = []
    for index in range(count):
        placed = False
        for attempt in range(100):
            local_scale = scale if attempt < 55 else 0
            h, w = _sample_dimensions(rng, height, width, local_scale, parities[index])
            mask = _make_symmetric_mask(rng, h, w)
            h, w = mask.shape
            local = set(map(tuple, np.argwhere(mask).tolist()))
            # No arbitrary per-row pair or eight-cell requirement. A complete
            # finite symmetric core with an off-axis pair has a unique full axis.
            if w < 2:
                continue
            if len(occupied) + len(local) > height * width * .38:
                continue
            top = int(rng.integers(1, height-h)) if height-h >= 2 else int(rng.integers(height-h+1))
            left = int(rng.integers(1, width-w)) if width-w >= 2 else int(rng.integers(width-w+1))
            if rng.random() < .12:
                if rng.random() < .5:
                    top = int(rng.choice([0, height - h]))
                else:
                    left = int(rng.choice([0, width - w]))
            core = {(r + top, c + left) for r, c in local}
            forbidden = occupied | (_neighbors(occupied, True) if rng.random() < .8 else set())
            if core & forbidden:
                continue
            axis = 2 * left + w - 1
            if axis % 2 != parities[index]:
                continue
            occupied |= core
            figures.append((core, axis))
            placed = True
            break
        if not placed:
            return None
    return figures


def _add_noise(rng, core, axis, occupied, shape, amount):
    height, width = shape
    r0, r1, c0, c1 = _bbox(core)
    halo = int(rng.integers(1, 5))
    local = {(r, c) for r in range(max(0, r0-halo), min(height, r1+halo+1))
             for c in range(max(0, c0-halo), min(width, c1+halo+1))}
    # Usually local, sometimes genuinely isolated marks anywhere on the canvas.
    if rng.random() < .20:
        local |= {(int(rng.integers(height)), int(rng.integers(width))) for _ in range(20)}
    candidates = {p for p in local - occupied
                  if 2*p[1] != axis and (p[0], axis-p[1]) not in core}
    selected = set()
    near = _neighbors(core, True)
    for _ in range(amount):
        if not candidates:
            break
        branch = candidates & _neighbors(selected)
        inside = {p for p in candidates if r0 <= p[0] <= r1 and c0 <= p[1] <= c1}
        attached = candidates & near
        detached = candidates - near
        roll = rng.random()
        pool = (branch if roll < .34 else inside if roll < .54
                else attached if roll < .85 else detached)
        choices = sorted(pool or candidates)
        # Reject a bad addition immediately: support at a competing axis can
        # only increase as more cells are added, so skipping it is safe.
        accepted = False
        for j in rng.permutation(len(choices))[:60]:
            p = choices[int(j)]
            candidates.discard(p)
            try:
                winner, supported = _best_vertical_axis(core | selected | {p}, width)
                if winner != axis or supported != core:
                    continue
            except AssertionError:
                continue
            selected.add(p)
            candidates.discard((p[0], axis-p[1]))
            accepted = True
            break
        if not accepted:
            break
    return selected


def _fallback_pair(rng, shape, colors, background, parities, anchor):
    """Guaranteed cell-safe small shapes, used only after bounded placement retries."""
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for index, (color, parity) in enumerate(zip(colors, parities)):
        top, left = (index // 2) * 3, (index % 2) * 5
        width = 3 if parity == 0 else 4
        out[top, left:left+width] = color
        out[top+1, left+1:left+width-1] = color
        inp[:] = np.where(out != background, out, inp)
        if index == anchor or rng.random() < .75:
            inp[top+2, left] = color
    return {"input": inp, "output": out}


def _generate_pair(rng, regime, parameters, background, count, anchor_parity=None):
    assert regime in ("source_train", "target_test")
    target = regime == "target_test"
    scale = parameters["complexity"]
    noise_level = int(rng.integers(3))
    shift = parameters["shift"]
    if shift in ("size", "source_scale"):
        scale = 2 if target else 0
    elif shift == "smaller_test" and target:
        scale = 0
    if shift in ("noise", "source_scale"):
        noise_level = 2 if target else 0
    if rng.random() < .15:
        shape = (22, 22)
    else:
        h = int(rng.integers(10, 31))
        shape = (h, h if rng.random() < .27 else int(rng.integers(10, 31)))
    # Only packing capacity couples canvas and count; no fixed slots are used.
    if count >= 4:
        shape = (max(15, shape[0]), max(15, shape[1]))
    colors = list(map(int, rng.choice([c for c in range(10) if c != background], count, replace=False)))
    allowed = parameters["parities"]
    pair_parities = allowed if rng.random() < .45 else (int(rng.choice(allowed)),)
    parities = list(map(int, rng.choice(pair_parities, count)))
    anchor = int(rng.integers(count))
    if anchor_parity is not None:
        parities[anchor] = anchor_parity
    unchanged = anchor_parity is None and rng.random() < .07
    for _ in range(32):
        figures = _place_cores(rng, shape, count, parities, scale)
        if figures is None:
            continue
        inp = np.full(shape, background, dtype=np.int8)
        out = inp.copy()
        occupied = set().union(*(core for core, _ in figures))
        for color, (core, axis) in zip(colors, figures):
            for p in core:
                inp[p] = out[p] = color
        for index, (color, (core, axis)) in enumerate(zip(colors, figures)):
            clean = unchanged or rng.random() < .22
            ratio = float(rng.uniform(*((.04, .18), (.10, .35), (.23, .52))[noise_level]))
            amount = 0 if clean else min(24, max(1, int(round(len(core) * ratio))))
            if anchor_parity is not None and index == anchor:
                amount = max(1, amount)
            amount = min(amount, max(0, int(inp.size * .48) - len(occupied)))
            additions = _add_noise(rng, core, axis, occupied, shape, amount)
            occupied |= additions
            for p in additions:
                inp[p] = color
        pair = {"input": inp, "output": out}
        if anchor_parity is not None:
            anchor_color = colors[anchor]
            if not np.any((inp == anchor_color) & (out != anchor_color)) or _fits(pair, 1):
                continue
        predicted, _ = _erase_vertical(inp)
        assert np.array_equal(predicted, out)
        return pair
    return _fallback_pair(rng, shape, colors, background, parities, anchor)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parity_mode = int(rng.choice(3, p=(.22, .22, .56)))
    parameters = {
        "orientation": int(rng.integers(2)),
        "parities": (0, 1) if parity_mode == 2 else (parity_mode,),
        "complexity": int(rng.integers(3)),
        "shift": str(rng.choice(["comparable", "size", "count", "noise", "source_scale", "smaller_test"],
                                p=[.4, .12, .12, .12, .12, .12])),
    }
    anchors = list(parameters["parities"])
    rng.shuffle(anchors)
    if len(anchors) == 1:
        anchors *= 2
    puzzle = {"train": [], "test": []}
    seen = set()
    prior_counts = []
    for split, length in (("train", train_count), ("test", test_count)):
        backgrounds = []
        for index in range(length):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            background_pool = list(range(10))
            if index == length - 1 and backgrounds and len(set(backgrounds)) == 1:
                background_pool.remove(backgrounds[0])
            background = int(pair_rng.choice(background_pool))
            max_count = (3, 5, 6)[parameters["complexity"]]
            min_count = 1
            if parameters["shift"] == "count":
                min_count, max_count = (3, 6) if split == "test" else (1, 3)
            counts = list(range(min_count, max_count + 1))
            if split == "train" and index == 1 and prior_counts[0] in counts:
                counts.remove(prior_counts[0])
            count = int(pair_rng.choice(counts))
            anchor_parity = anchors[index] if split == "train" and index < 2 else None
            for _ in range(30):
                pair = _generate_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                      parameters, background, count, anchor_parity)
                if parameters["orientation"]:
                    pair = {role: np.rot90(grid).copy() for role, grid in pair.items()}
                key = _grid_key(pair["input"])
                if key not in seen and not _source_copy(pair["input"]):
                    break
            else:
                raise AssertionError("duplicate after bounded resampling")
            puzzle[split].append(pair)
            backgrounds.append(background)
            prior_counts.append(count)
            seen.add(key)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            assert not _source_copy(inp)
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            bg = _mode_color(inp)
            assert _mode_color(out) == bg
            assert 1 <= len(np.unique(inp)) - 1 <= 6
            assert set(np.unique(inp)) == set(np.unique(out))
            assert np.all((inp == out) | (out == bg)), "only erase to background"
    orientation = _learn_orientation(puzzle["train"])
    demonstrated, queried = set(), set()
    backgrounds, counts = {s: set() for s in puzzle}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            prediction, axes = _prediction(inp, orientation)
            assert np.array_equal(prediction, out), "output is not the unique supported subset"
            backgrounds[split].add(_mode_color(inp))
            counts.add(len(axes))
            for color, axis in axes.items():
                if split == "train" and np.any((inp == color) & (out != color)):
                    demonstrated.add(axis % 2)
                if split == "test":
                    queried.add(axis % 2)
    assert queried <= demonstrated, "test parity was never demonstrated with deletion"
    assert len(backgrounds["train"]) >= 2
    if len(puzzle["test"]) > 1:
        assert len(backgrounds["test"]) >= 2
    assert len(counts) >= 2


def _components(cells):
    remaining = set(cells)
    components = []
    while remaining:
        start = remaining.pop()
        component, frontier = {start}, {start}
        while frontier:
            frontier = _neighbors(frontier) & remaining
            remaining -= frontier
            component |= frontier
        components.append(component)
    return components


def _holes(core):
    r0, r1, c0, c1 = _bbox(core)
    empty = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)} - core
    boundary = {p for p in empty if p[0] in (r0, r1) or p[1] in (c0, c1)}
    exterior, frontier = set(boundary), boundary
    while frontier:
        frontier = (_neighbors(frontier) & empty) - exterior
        exterior |= frontier
    return empty - exterior


def measure_pair(pair, orientation):
    """Review features derived solely from returned grids and learned direction."""
    inp, out = pair["input"], pair["output"]
    bg = _mode_color(inp)
    original_shape = list(map(int, inp.shape))
    if orientation:
        inp, out = inp.T, out.T
    _, axes = _erase_vertical(inp)
    figures = []
    boxes = []
    for color, axis in axes.items():
        core = _cells(out, color)
        noise = _cells(inp, color) - core
        r0, r1, c0, c1 = _bbox(core)
        h, w = r1-r0+1, c1-c0+1
        row_counts = [sum((r, c) in core for c in range(c0, c1+1)) for r in range(r0, r1+1)]
        components = _components(core)
        noise_components = _components(noise)
        hole_cells = _holes(core)
        boundary = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                    if r in (r0, r1) or c in (c0, c1)}
        mid = (r0+r1)//2
        boxes.append(_bbox(core | noise))
        figures.append({
            "color": color, "axis2": axis, "parity": axis % 2,
            "height": h, "width": w, "cells": len(core), "noise": len(noise),
            "density": len(core)/(h*w), "row_patterns": len({tuple(c-c0 for r2, c in sorted(core) if r2 == r)
                                                               for r in range(r0, r1+1)}),
            "axis_only_rows": sum(n == 1 and (r0+i, axis//2) in core and axis % 2 == 0
                                  for i, n in enumerate(row_counts)),
            "narrow_rows": sum(0 < n <= 2 for n in row_counts),
            "variable_thickness": len(set(n for n in row_counts if n)) >= 3,
            "middle_not_widest": (mid, c0) not in core and (mid, c1) not in core,
            "thin_contour": h >= 3 and w >= 3 and core == boundary,
            "holes": len(hole_cells), "noise_in_holes": len(noise & hole_cells),
            "noise_inside_box": sum(r0 <= r <= r1 and c0 <= c <= c1 for r, c in noise),
            "noise_outside_box": sum(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in noise),
            "attached_noise": len(noise & _neighbors(core, True)),
            "detached_noise": len(noise - _neighbors(core, True)),
            "noise_cluster": max(map(len, noise_components), default=0),
            "noise_branch": any(len(comp) >= 3 and min(_bbox(comp)[1]-_bbox(comp)[0],
                                                       _bbox(comp)[3]-_bbox(comp)[2]) <= 1
                                for comp in noise_components),
            "retained_components": len(components),
            "retained_singletons": sum(len(c) == 1 for c in components),
            "retained_axis_singletons": sum(len(c) == 1 and 2*next(iter(c))[1] == axis for c in components),
            "edge_core": any(r in (0, out.shape[0]-1) or c in (0, out.shape[1]-1) for r, c in core),
            "box": [r0, r1, c0, c1],
        })
    parity_set = {f["parity"] for f in figures}
    overlap = sum(not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])
                  for i, a in enumerate(boxes) for b in boxes[i+1:])
    return {
        "shape": original_shape, "background": bg,
        "palette": list(map(int, np.unique(inp))),
        "orientation": "horizontal" if orientation else "vertical",
        "count": len(figures), "parity_mix": "mixed" if len(parity_set) == 2 else "integral" if parity_set == {0} else "half",
        "noise": sum(f["noise"] for f in figures), "max_noise": max(f["noise"] for f in figures),
        "clean_count": sum(f["noise"] == 0 for f in figures),
        "large_count": sum(max(f["height"], f["width"]) >= 10 for f in figures),
        "edge_core": any(f["edge_core"] for f in figures),
        "overlapping_boxes": overlap, "figures": figures,
    }


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s])
                                            for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    records, within = [], []
    train_counts, test_counts, fingerprints = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        orientation = _learn_orientation(puzzle["train"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_grid_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                # Coordinate-set reconstruction also covers training. The
                # validator above uses convolutions, not these set operations.
                grid = pair["input"] if orientation == 0 else pair["input"].T
                out = pair["output"] if orientation == 0 else pair["output"].T
                bg = _mode_color(grid)
                for color in set(map(int, np.unique(grid))) - {bg}:
                    _, supported = _best_vertical_axis(_cells(grid, color), grid.shape[1])
                    assert supported == _cells(out, color)
                if split == "test":
                    training = [{k: v.copy() for k, v in p.items()} for p in puzzle["train"]]
                    prediction = solve_from_training(training, pair["input"].copy())
                    assert np.array_equal(prediction, pair["output"])
                item = measure_pair(pair, orientation)
                item["split"] = split
                records.append(item)
                local.append(item)
        within.append(local)
        if seed < 3:
            assert _puzzles_equal(puzzle, generate_puzzle(seed))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42
    assert {p["orientation"] for p in records} == {"horizontal", "vertical"}
    assert {p["parity_mix"] for p in records} == {"integral", "half", "mixed"}
    assert len({tuple(p["shape"]) for p in records}) >= 60
    assert min(min(p["shape"]) for p in records) == 10
    assert max(max(p["shape"]) for p in records) == 30
    assert {p["count"] for p in records} == set(range(1, 7))
    assert {p["background"] for p in records} == set(range(10))
    assert len({tuple(p["palette"]) for p in records}) >= 80
    assert any(not set(p["palette"]) & {5, 7, 9} for p in records)
    assert all(len({p["count"] for p in local}) >= 2 and
               len({p["background"] for p in local if p["split"] == "train"}) >= 2 for local in within)
    assert sum(len({tuple(p["shape"]) for p in local}) >= 3 for local in within) >= 35
    assert sum(p["overlapping_boxes"] > 0 for p in records) >= 20
    assert sum(p["edge_core"] for p in records) >= 40
    assert sum(p["large_count"] >= 2 for p in records) >= 10
    for split in ("train", "test"):
        pairs = [p for p in records if p["split"] == split]
        assert any(p["noise"] >= 17 for p in pairs)
        assert any(p["max_noise"] >= 12 for p in pairs)
        assert any(p["clean_count"] == 0 for p in pairs)
        assert any(0 < p["clean_count"] < p["count"] for p in pairs)
        assert any(p["noise"] == 0 for p in pairs)
        assert any(p["large_count"] for p in pairs)
        assert any(all(max(f["height"], f["width"]) <= 7 for f in p["figures"]) for p in pairs)
    figures = [f for p in records for f in p["figures"]]
    assert sum(f["cells"] <= 7 for f in figures) >= 30
    assert any((f["height"], f["width"], f["cells"]) == (4, 3, 6) for f in figures)
    assert any((f["height"], f["width"], f["cells"]) == (2, 5, 7) for f in figures)
    assert any(f["height"] == 3 and f["width"] >= 9 for f in figures)
    assert sum(f["height"] <= 3 and f["width"] >= 8 for f in figures) >= 10
    assert sum(f["width"] <= 3 and f["height"] >= 8 for f in figures) >= 10
    for feature in ("axis_only_rows", "thin_contour", "middle_not_widest", "variable_thickness",
                    "noise_in_holes", "noise_inside_box", "noise_outside_box", "attached_noise",
                    "detached_noise", "retained_singletons", "retained_axis_singletons", "noise_branch"):
        assert sum(bool(f[feature]) for f in figures) >= 5, feature
    assert any(f["density"] < .25 for f in figures) and any(f["density"] == 1 for f in figures)
    assert any(f["noise_cluster"] >= 5 for f in figures)
    assert sum(p["edge_core"] and p["count"] >= 3 and p["noise"] >= 6 and p["shape"][0] != p["shape"][1]
               for p in records) >= 15
    # Observe split differences, instead of treating sampled regime names as
    # evidence of coverage. Relative noise separates corruption from scale.
    spectrum, exchanged_roles = set(), 0
    for local in within:
        training = [p for p in local if p["split"] == "train"]
        testing = [p for p in local if p["split"] == "test"]
        train_colors = {f["color"] for p in training for f in p["figures"]}
        exchanged_roles += bool(train_colors & {p["background"] for p in testing})
        split_stats = []
        for pairs in (training, testing):
            fs = [f for p in pairs for f in p["figures"]]
            split_stats.append((sum(max(f["height"], f["width"]) for f in fs)/len(fs),
                                sum(p["count"] for p in pairs)/len(pairs),
                                sum(f["noise"] for f in fs)/sum(f["cells"] for f in fs)))
        signature = []
        for index, (before, after) in enumerate(zip(*split_stats)):
            tolerance = .75 if index == 1 else max(.04, before*.25)
            signature.append(1 if after > before+tolerance else -1 if after < before-tolerance else 0)
        spectrum.add(tuple(signature))
    assert exchanged_roles >= 20
    assert {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)} <= spectrum
    assert any(signature[0] == -1 for signature in spectrum)
    assert sum(len({f["box"][0] for f in p["figures"]}) > 1 and
               len({f["box"][2] for f in p["figures"]}) > 1 for p in records) >= 80
    assert sum(len({f["axis2"] for f in p["figures"]}) > 1 for p in records) >= 80
    # Mutation rejection must run even when a selected test happens to be clean.
    bad = generate_puzzle(901, 2, 1)
    output = bad["test"][0]["output"]
    background = _mode_color(output)
    row, col = np.argwhere(output != background)[0]
    output[row, col] = background
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted deletion from the supported core")
    for parity in (0, 1):
        for count in range(1, 7):
            pair = _fallback_pair(np.random.default_rng(0), (10, 10), list(range(1, count+1)),
                                  0, [parity]*count, count-1)
            assert _fits(pair, 0) and not _fits(pair, 1)
    for seed, requested in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + seed, *requested)
        validate_generated_puzzle(puzzle)
        for split, expected in zip(("train", "test"), requested):
            assert expected is None or len(puzzle[split]) == expected
        assert _puzzles_equal(puzzle, generate_puzzle(1000 + seed, *requested))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(700, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
