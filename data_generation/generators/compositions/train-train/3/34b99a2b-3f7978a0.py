"""Registered twin windows: overlay the two complete capped-rail windows and XOR their payloads.

Input-only family rule
----------------------
Every input uses exactly three colours: background, rail and marker. Somewhere on the canvas there are
exactly two *complete windows*.  A complete window is an axis-aligned rectangle whose two opposite sides
(without corners) are uninterrupted runs of the rail colour, whose four corners are marker-coloured caps
that sit immediately beyond both ends of both rails, which contains no other rail cell, and which contains
at least one background cell (parent B's capped-rail window).  The two complete windows are congruent:
same height, width and rail direction.  Every cell of a window that is not a rail or cap is a payload cell
and holds either background or marker.

Overlay the two windows by their caps (registration marks).  The output is the common window: rails and
caps are copied, and each payload cell is marker exactly when one window -- but not both -- has a marker
there (parent A's coordinate-wise exclusive-or); otherwise it is background.

Distractors share the marker colour (scattered/clustered noise) or the rail colour (frames with missing
caps, capped rails of unequal extent, lone capped rails, isolated rail cells).  None of them forms a third
complete window.
"""
from __future__ import annotations

import numpy as np

VERTICAL, HORIZONTAL = "vertical", "horizontal"
MAX_SIDE = 30

_SHIFTS = ("scale_up", "scale_down", "orientation_holdout", "relation_boundary",
           "clutter_up", "arrangement_shift", "plain")
_SHIFT_P = (0.17, 0.09, 0.15, 0.17, 0.15, 0.12, 0.15)
_DECOY_KINDS = ("broken", "misaligned", "lone", "specks")
_BOUNDARY_RELATIONS = ("identical", "complementary", "contained", "disjoint", "one_empty")
_STYLES = ("scatter", "band", "radial", "islands", "strokes", "repeat", "shapes", "stripes")


class _Retry(Exception):
    """Internal signal: the current random attempt violated a constraint."""


# ---------------------------------------------------------------------------------------------
# Shared geometry and the input-only solver
# ---------------------------------------------------------------------------------------------

def _frame_mask(shape, orientation):
    """Rails plus caps of a window of the given shape."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if orientation == VERTICAL:
        mask[:, 0] = mask[:, -1] = True
    else:
        mask[0, :] = mask[-1, :] = True
    return mask


def _payload_shape(height, width, orientation):
    return (height, width - 2) if orientation == VERTICAL else (height - 2, width)


def _payload_view(window, orientation):
    return window[:, 1:-1] if orientation == VERTICAL else window[1:-1, :]


def _capped_runs(work, rail, marker):
    """Group maximal rail runs (length >= 2) down each column that are capped by marker at both ends."""
    height, width = work.shape
    groups = {}
    is_rail = work == rail
    for col in range(width):
        column = is_rail[:, col]
        row = 0
        while row < height:
            if not column[row]:
                row += 1
                continue
            start = row
            while row < height and column[row]:
                row += 1
            stop = row
            if stop - start < 2 or start == 0 or stop == height:
                continue
            if work[start - 1, col] == marker and work[stop, col] == marker:
                groups.setdefault((start, stop), []).append(col)
    return groups


def _find_windows(grid):
    """All complete capped-rail windows under every role hypothesis (input-only, parent B detector)."""
    grid = np.asarray(grid)
    colors = [int(c) for c in np.unique(grid)]
    if len(colors) != 3:
        return []
    found = []
    for rail in colors:
        for marker in colors:
            if marker == rail:
                continue
            background = next(c for c in colors if c not in (rail, marker))
            for orientation in (VERTICAL, HORIZONTAL):
                work = grid if orientation == VERTICAL else grid.T
                for (start, stop), columns in _capped_runs(work, rail, marker).items():
                    for i, left in enumerate(columns):
                        for right in columns[i + 1:]:
                            if right - left < 2:
                                continue
                            crop = work[start - 1:stop + 1, left:right + 1]
                            if np.count_nonzero(crop == rail) != 2 * (stop - start):
                                continue
                            if not np.any(crop == background):
                                continue
                            if orientation == VERTICAL:
                                bounds = (start - 1, left, stop, right)
                            else:
                                bounds = (left, start - 1, right, stop)
                            found.append({"bounds": bounds, "orientation": orientation,
                                          "palette": (background, rail, marker)})
    found.sort(key=lambda item: item["bounds"])
    return found


def _background_is_plurality(grid, background):
    values, counts = np.unique(grid, return_counts=True)
    best = counts.max()
    return int(counts[values == background][0]) == best and int(np.count_nonzero(counts == best)) == 1


def _crop(grid, bounds):
    top, left, bottom, right = bounds
    return grid[top:bottom + 1, left:right + 1]


def _analyse(grid):
    grid = np.asarray(grid)
    windows = _find_windows(grid)
    if len(windows) != 2:
        raise ValueError(f"expected exactly two complete capped-rail windows, found {len(windows)}")
    first, second = windows
    if first["orientation"] != second["orientation"] or first["palette"] != second["palette"]:
        raise ValueError("the two windows must share rail direction and colour roles")
    crops = [_crop(grid, w["bounds"]) for w in windows]
    if crops[0].shape != crops[1].shape:
        raise ValueError("the two windows must be congruent")
    frame = _frame_mask(crops[0].shape, first["orientation"])
    if not np.array_equal(crops[0][frame], crops[1][frame]):
        raise ValueError("the two frames must match")
    background, rail, marker = first["palette"]
    marks = [c == marker for c in crops]
    payload = ~frame
    cases = {
        "neither": int(np.count_nonzero(payload & ~marks[0] & ~marks[1])),
        "first_only": int(np.count_nonzero(payload & marks[0] & ~marks[1])),
        "second_only": int(np.count_nonzero(payload & ~marks[0] & marks[1])),
        "both": int(np.count_nonzero(payload & marks[0] & marks[1])),
    }
    return {"windows": windows, "crops": crops, "frame": frame, "palette": first["palette"],
            "orientation": first["orientation"], "shape": crops[0].shape, "cases": cases,
            "marks": marks}


def _combine(info, operation):
    background, rail, marker = info["palette"]
    first, second = info["marks"]
    if operation == "xor":
        keep = first ^ second
    elif operation == "union":
        keep = first | second
    elif operation == "intersection":
        keep = first & second
    elif operation == "first":
        keep = first
    elif operation == "second":
        keep = second
    else:
        raise ValueError(operation)
    out = info["crops"][0].copy()
    payload = ~info["frame"]
    out[payload] = np.where(keep[payload], marker, background)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Find the two complete windows, overlay them by their caps and keep exclusive payload marks."""
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("input must be a 2D grid")
    return _combine(_analyse(grid), "xor").astype(int)


# ---------------------------------------------------------------------------------------------
# Payload masks
# ---------------------------------------------------------------------------------------------

def _score(rng, height, width, style):
    rows, cols = np.indices((height, width), dtype=float)
    y = rows / max(1.0, height - 1)
    x = cols / max(1.0, width - 1)
    jitter = rng.random((height, width))
    if style == "scatter":
        return jitter
    if style == "band":
        angle = float(rng.uniform(0, 2 * np.pi))
        return np.cos(angle) * y + np.sin(angle) * x + 0.15 * jitter
    if style == "radial":
        cy, cx = rng.uniform(-0.2, 1.2, size=2)
        return (y - cy) ** 2 + (x - cx) ** 2 + 0.08 * jitter
    if style == "islands":
        field = np.zeros((height, width))
        for _ in range(int(rng.integers(1, 5))):
            cy, cx = rng.uniform(0, 1, size=2)
            sy, sx = rng.uniform(0.1, 0.35, size=2)
            field += float(rng.uniform(0.6, 1.4)) * np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))
        return -field + 0.05 * jitter
    if style == "strokes":
        distance = np.full((height, width), 10.0)
        for _ in range(int(rng.integers(1, 4))):
            ay, ax, by, bx = rng.uniform(-0.1, 1.1, size=4)
            dy, dx = by - ay, bx - ax
            t = np.clip(((y - ay) * dy + (x - ax) * dx) / (dy * dy + dx * dx + 1e-9), 0, 1)
            distance = np.minimum(distance, (y - ay - t * dy) ** 2 + (x - ax - t * dx) ** 2)
        return distance + 0.03 * jitter
    if style == "repeat":
        py, px = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        oy, ox = rng.uniform(0, 2 * np.pi, size=2)
        return np.sin(rows * 2 * np.pi / py + oy) + np.sin(cols * 2 * np.pi / px + ox) + 0.2 * jitter
    if style == "shapes":
        score = np.full((height, width), 10.0)
        for _ in range(int(rng.integers(1, 5))):
            cy, cx = rng.uniform(0, 1, size=2)
            sy, sx = rng.uniform(0.1, 0.45, size=2)
            if rng.random() < 0.5:
                part = np.maximum(np.abs(y - cy) / sy, np.abs(x - cx) / sx)
            else:
                part = np.abs(y - cy) / sy + np.abs(x - cx) / sx
            score = np.minimum(score, part)
        return score + 0.05 * jitter
    # stripes: whole rows or columns share a level
    if rng.random() < 0.5:
        return rng.random(height)[:, None] + 0.25 * jitter
    return rng.random(width)[None, :] + 0.25 * jitter


def _mask(rng, shape, density, style=None):
    height, width = shape
    total = height * width
    if style is None:
        style = _STYLES[int(rng.integers(len(_STYLES)))]
    score = _score(rng, height, width, style)
    count = int(np.clip(round(density * total), 1, max(1, total - 1)))
    flat = np.zeros(total, dtype=bool)
    flat[np.argsort(score.ravel(), kind="stable")[:count]] = True
    return flat.reshape(shape)


def _shift(mask, dy, dx):
    out = np.zeros_like(mask)
    height, width = mask.shape
    src = mask[max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)]
    out[max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
    return out


def _payload_pair(rng, shape, relation):
    """Two payload masks with the requested relation; neither is completely marked."""
    total = shape[0] * shape[1]
    for _ in range(80):
        first = _mask(rng, shape, float(rng.uniform(0.2, 0.6)))
        if relation in ("ordinary", "ordinary_full"):
            second = _mask(rng, shape, float(rng.uniform(0.2, 0.6)))
            if relation == "ordinary_full" and total >= 4:
                picks = rng.choice(total, size=4, replace=False)
                for position, (a, b) in zip(picks, ((0, 0), (1, 0), (0, 1), (1, 1))):
                    first.ravel()[position] = bool(a)
                    second.ravel()[position] = bool(b)
        elif relation == "shifted":
            moves = [(dy, dx) for dy in (-1, 0, 1, 2) for dx in (-1, 0, 1, 2)
                     if (dy, dx) != (0, 0) and abs(dy) < shape[0] and abs(dx) < shape[1]]
            dy, dx = moves[int(rng.integers(len(moves)))]
            second = _shift(first, dy, dx)
        elif relation == "identical":
            second = first.copy()
        elif relation == "complementary":
            second = ~first
        elif relation == "disjoint":
            second = _mask(rng, shape, float(rng.uniform(0.2, 0.6))) & ~first
        elif relation == "contained":
            positions = np.flatnonzero(first.ravel())
            if len(positions) < 2:
                continue
            order = positions[np.argsort(_score(rng, *shape, _STYLES[int(rng.integers(len(_STYLES)))]).ravel()[positions])]
            take = int(rng.integers(1, len(positions)))
            second = np.zeros_like(first)
            second.ravel()[order[:take]] = True
        elif relation == "one_empty":
            second = np.zeros_like(first)
        else:
            raise ValueError(relation)
        if first.all() or second.all() or not first.any():
            continue
        if relation != "one_empty" and not second.any():
            continue
        if relation in ("ordinary", "shifted"):
            # ordinary overlays always contain a cancelled overlap and marks unique to each window
            if not ((first & second).any() and (first & ~second).any() and (~first & second).any()):
                continue
        if relation == "ordinary_full":
            cases = (~first & ~second, first & ~second, ~first & second, first & second)
            if not all(c.any() for c in cases):
                continue
        if rng.random() < 0.5:
            first, second = second, first
        return first, second
    raise _Retry("payload relation")


def _window_grid(shape, orientation, palette, payload):
    background, rail, marker = palette
    height, width = shape
    grid = np.full((height, width), background, dtype=int)
    if orientation == VERTICAL:
        grid[:, 0] = grid[:, -1] = rail
        grid[:, 1:-1] = np.where(payload, marker, background)
    else:
        grid[0, :] = grid[-1, :] = rail
        grid[1:-1, :] = np.where(payload, marker, background)
    grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = marker
    return grid


# ---------------------------------------------------------------------------------------------
# Decoys: every one is an incomplete version of a window
# ---------------------------------------------------------------------------------------------

def _broken_decoy(rng, shape, orientation, palette):
    """A window-like frame with full rails and a payload, but one or more caps missing."""
    background = palette[0]
    payload = _mask(rng, _payload_shape(*shape, orientation), float(rng.uniform(0.15, 0.55)))
    patch = _window_grid(shape, orientation, palette, payload)
    corners = ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1))
    missing = int(rng.choice((1, 1, 1, 2, 2, 4)))
    for index in rng.choice(4, size=missing, replace=False):
        patch[corners[int(index)]] = background
    return patch


def _misaligned_decoy(rng, palette, max_axis):
    """Two parallel capped rails whose extents differ; a payload sits between them."""
    background, rail, marker = palette
    separation = int(rng.integers(3, 8))
    for _ in range(40):
        l1, l2 = int(rng.integers(2, max_axis + 1)), int(rng.integers(2, max_axis + 1))
        d1, d2 = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        if (d1, l1) != (d2, l2):
            break
    else:
        raise _Retry("misaligned decoy")
    height = max(d1 + l1 + 2, d2 + l2 + 2)
    patch = np.full((height, separation), background, dtype=int)
    inner = _mask(rng, (height, separation - 2), float(rng.uniform(0.1, 0.4)))
    patch[:, 1:-1] = np.where(inner, marker, background)
    for col, offset, length in ((0, d1, l1), (separation - 1, d2, l2)):
        patch[offset, col] = marker
        patch[offset + 1:offset + 1 + length, col] = rail
        patch[offset + 1 + length, col] = marker
    return patch if rng.random() < 0.5 else patch.T.copy()


def _lone_decoy(rng, palette, max_axis):
    background, rail, marker = palette
    length = int(rng.integers(2, max_axis + 1))
    patch = np.full((length + 2, 1), rail, dtype=int)
    patch[0, 0] = patch[-1, 0] = marker
    return patch if rng.random() < 0.5 else patch.T.copy()


def _speck_decoy(palette):
    return np.full((1, 1), palette[1], dtype=int)


# ---------------------------------------------------------------------------------------------
# Puzzle planning
# ---------------------------------------------------------------------------------------------

def _as_int(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _resolve_counts(num_train, num_test):
    test = 1 if num_test is None else _as_int(num_test, "num_test", 1, 4)
    train = max(4, test + 1) if num_train is None else _as_int(num_train, "num_train", 2, 10)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _other(orientation):
    return HORIZONTAL if orientation == VERTICAL else VERTICAL


def _random_palette(rng, previous=None, background=None):
    for _ in range(50):
        if background is None:
            colors = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
        else:
            rest = [c for c in range(10) if c != background]
            pick = rng.choice(rest, size=2, replace=False)
            colors = (background, int(pick[0]), int(pick[1]))
        if colors != previous:
            return colors
    return colors


def _plan(rng, num_train, num_test):
    shift = str(rng.choice(_SHIFTS, p=_SHIFT_P))
    total = num_train + num_test
    palette_mode = str(rng.choice(("stable", "bg_fixed", "varied"), p=(0.25, 0.25, 0.5)))
    base_palette = tuple(int(c) for c in rng.choice(10, size=3, replace=False))
    base_orientation = VERTICAL if rng.random() < 0.5 else HORIZONTAL
    orientation_policy = "holdout" if shift == "orientation_holdout" else (
        "fixed" if rng.random() < 0.3 else "mixed")
    if shift == "clutter_up" or rng.random() < 0.8:
        size = int(rng.integers(1, 4))
        kinds = rng.choice(_DECOY_KINDS, size=size, replace=False, p=(0.4, 0.2, 0.2, 0.2))
        decoy_kinds = tuple(sorted(str(k) for k in kinds))
    else:
        decoy_kinds = ()
    allow_shared = shift != "arrangement_shift" and rng.random() < 0.35
    arrangement_pair = tuple(str(a) for a in rng.permutation(("side", "stacked")))
    noise_bias = str(rng.choice(("light", "mixed", "heavy"), p=(0.3, 0.45, 0.25)))
    demo = {kind: int(rng.integers(num_train)) for kind in decoy_kinds}
    shared_demo = int(rng.integers(num_train)) if allow_shared else None

    specs = []
    palette = None
    for index in range(total):
        training = index < num_train
        spec = {"index": index, "training": training}
        # colours
        if palette_mode == "stable":
            palette = base_palette
        elif palette_mode == "bg_fixed":
            palette = _random_palette(rng, palette, base_palette[0]) if index else base_palette
        else:
            palette = _random_palette(rng, palette) if index else base_palette
        spec["palette"] = palette
        # rail direction
        if orientation_policy == "holdout":
            orientation = base_orientation if training else _other(base_orientation)
        elif orientation_policy == "fixed":
            orientation = base_orientation
        elif index == 0:
            orientation = base_orientation
        elif index == 1:
            orientation = _other(base_orientation)
        else:
            orientation = base_orientation if rng.random() < 0.5 else _other(base_orientation)
        spec["orientation"] = orientation
        # payload relation
        if training and index == 0:
            relation = "ordinary_full"
        elif shift == "relation_boundary":
            relation = (str(rng.choice(("ordinary", "shifted"), p=(0.65, 0.35))) if training
                        else str(rng.choice(_BOUNDARY_RELATIONS, p=(0.3, 0.15, 0.3, 0.15, 0.1))))
        else:
            weights = (0.34, 0.16, 0.12, 0.08, 0.1, 0.14, 0.06) if training else \
                (0.4, 0.18, 0.12, 0.06, 0.06, 0.14, 0.04)
            relation = str(rng.choice(("ordinary", "shifted", "identical", "complementary",
                                       "disjoint", "contained", "one_empty"), p=weights))
        spec["relation"] = relation
        # arrangement of the two windows
        if shift == "arrangement_shift":
            arrangement = arrangement_pair[0] if training else arrangement_pair[1]
        elif allow_shared and training and index == shared_demo:
            arrangement = "shared"
        else:
            options, weights = ["side", "stacked"], [0.5, 0.5]
            if allow_shared:
                options.append("shared")
                weights = [0.4, 0.4, 0.2]
            arrangement = str(rng.choice(options, p=weights))
        spec["arrangement"] = arrangement
        # exterior marker clutter
        if shift == "clutter_up":
            noise = str(rng.choice(("none", "sparse"), p=(0.35, 0.65))) if training else \
                str(rng.choice(("medium", "dense"), p=(0.45, 0.55)))
        else:
            levels = ("none", "sparse", "medium", "dense")
            weights = {"light": (0.25, 0.5, 0.2, 0.05), "mixed": (0.1, 0.35, 0.35, 0.2),
                       "heavy": (0.05, 0.2, 0.4, 0.35)}[noise_bias]
            noise = str(rng.choice(levels, p=weights))
        spec["noise"] = noise
        # decoys (only of kinds demonstrated in training)
        required = [k for k in decoy_kinds if training and demo[k] == index]
        optional = []
        for kind in decoy_kinds:
            if kind in required:
                continue
            if shift == "clutter_up" and not training:
                optional.append(kind)
            elif rng.random() < (0.3 if training else 0.35):
                optional.append(kind)
        if shift == "clutter_up" and not training and decoy_kinds:
            optional.append(str(rng.choice(decoy_kinds)))
        spec["required_decoys"] = required
        spec["optional_decoys"] = optional
        # size regime
        if shift == "scale_up":
            regime = str(rng.choice(("small", "medium"))) if training else \
                str(rng.choice(("large", "long", "wide")))
        elif shift == "scale_down":
            regime = str(rng.choice(("medium", "large", "long", "wide"))) if training else "small"
        else:
            regime = str(rng.choice(("small", "medium", "long", "wide", "large"),
                                    p=(0.22, 0.32, 0.17, 0.17, 0.12)))
        spec["regime"] = regime
        spec["edge"] = (shift == "clutter_up" and not training) or rng.random() < 0.15
        specs.append(spec)
    return {"shift": shift, "palette_mode": palette_mode, "orientation_policy": orientation_policy,
            "decoy_kinds": decoy_kinds, "allow_shared": allow_shared, "specs": specs,
            "noise_bias": noise_bias}


def _sample_size(rng, regime, orientation, arrangement, bounds):
    """Window (height, width) for the regime that fits the arrangement and scale bounds."""
    for _ in range(200):
        if regime == "small":
            axis, sep = int(rng.integers(4, 7)), int(rng.integers(3, 7))
        elif regime == "medium":
            axis, sep = int(rng.integers(5, 11)), int(rng.integers(4, 10))
        elif regime == "long":
            axis, sep = int(rng.integers(9, 26)), int(rng.integers(3, 7))
        elif regime == "wide":
            axis, sep = int(rng.integers(4, 8)), int(rng.integers(7, 15))
        else:
            axis, sep = int(rng.integers(10, 17)), int(rng.integers(8, 15))
        shape = (axis, sep) if orientation == VERTICAL else (sep, axis)
        height, width = shape
        payload = np.prod(_payload_shape(height, width, orientation))
        if payload < bounds.get("min_payload", 6):
            continue
        area = height * width
        if area > bounds.get("max_area", 10 ** 6) or area < bounds.get("min_area", 0):
            continue
        if arrangement == "side" and 2 * width + 1 > MAX_SIDE:
            continue
        if arrangement == "stacked" and 2 * height + 1 > MAX_SIDE:
            continue
        if arrangement == "shared":
            if orientation == VERTICAL and 2 * width - 1 > MAX_SIDE:
                continue
            if orientation == HORIZONTAL and 2 * height - 1 > MAX_SIDE:
                continue
        if max(height, width) > MAX_SIDE:
            continue
        return shape
    raise _Retry("size")


def _layout(rng, shape, orientation, arrangement, roomy, edge):
    height, width = shape
    gap = int(rng.integers(1, 6))
    if arrangement == "shared":
        block = (height, 2 * width - 1) if orientation == VERTICAL else (2 * height - 1, width)
    elif arrangement == "side":
        block = (height, 2 * width + gap)
    else:
        block = (2 * height + gap, width)
    if block[0] > MAX_SIDE or block[1] > MAX_SIDE:
        gap = 1
        block = (height, 2 * width + 1) if arrangement == "side" else (
            (2 * height + 1, width) if arrangement == "stacked" else block)
        if block[0] > MAX_SIDE or block[1] > MAX_SIDE:
            raise _Retry("layout")
    room_h, room_w = MAX_SIDE - block[0], MAX_SIDE - block[1]
    low = 4 if roomy else 0
    extra_h = int(rng.integers(min(low, room_h), min(room_h, 14 if roomy else 10) + 1))
    extra_w = int(rng.integers(min(low, room_w), min(room_w, 14 if roomy else 10) + 1))
    if extra_h == 0 and extra_w == 0:
        if room_w:
            extra_w = 1
        elif room_h:
            extra_h = 1
    canvas = (block[0] + extra_h, block[1] + extra_w)
    rows, cols = canvas
    if arrangement == "shared":
        top = int(rng.integers(0, rows - block[0] + 1))
        left = int(rng.integers(0, cols - block[1] + 1))
        if orientation == VERTICAL:
            positions = [(top, left), (top, left + width - 1)]
        else:
            positions = [(top, left), (top + height - 1, left)]
    elif arrangement == "side":
        l1 = int(rng.integers(0, cols - block[1] + 1))
        l2 = int(rng.integers(l1 + width + gap, cols - width + 1))
        positions = [(int(rng.integers(0, rows - height + 1)), l1),
                     (int(rng.integers(0, rows - height + 1)), l2)]
    else:
        t1 = int(rng.integers(0, rows - block[0] + 1))
        t2 = int(rng.integers(t1 + height + gap, rows - height + 1))
        positions = [(t1, int(rng.integers(0, cols - width + 1))),
                     (t2, int(rng.integers(0, cols - width + 1)))]
    if edge:
        # Push one window (or the shared pair) against a canvas edge or corner while keeping the
        # arrangement's defining property (column-disjoint, row-disjoint or sharing a rail).
        index = int(rng.integers(2))
        top, left = positions[index]
        candidates = [(0, left), (rows - height, left), (top, 0), (top, cols - width)]
        candidates += [(r, c) for r in (0, rows - height) for c in (0, cols - width)]
        order = rng.permutation(len(candidates))
        for k in order:
            trial = list(positions)
            new = tuple(int(v) for v in candidates[int(k)])
            if arrangement == "shared":
                dy, dx = new[0] - positions[index][0], new[1] - positions[index][1]
                trial = [(p[0] + dy, p[1] + dx) for p in positions]
                if any(p[0] < 0 or p[1] < 0 or p[0] + height > rows or p[1] + width > cols for p in trial):
                    continue
            else:
                trial[index] = new
                if not _arrangement_ok(trial, shape, arrangement):
                    continue
            positions = trial
            break
    return canvas, positions


def _arrangement_ok(positions, shape, arrangement):
    """Side-by-side windows keep a free column between them; stacked ones keep a free row."""
    height, width = shape
    (t1, l1), (t2, l2) = positions
    if arrangement == "side":
        return l1 + width < l2 or l2 + width < l1
    return t1 + height < t2 or t2 + height < t1


def _place_patch(rng, blocked, patch_shape):
    rows, cols = blocked.shape
    ph, pw = patch_shape
    if ph > rows or pw > cols:
        return None
    options = []
    for top in range(rows - ph + 1):
        for left in range(cols - pw + 1):
            if not blocked[top:top + ph, left:left + pw].any():
                options.append((top, left))
    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _block(blocked, top, left, shape):
    rows, cols = blocked.shape
    blocked[max(0, top - 1):min(rows, top + shape[0] + 1), max(0, left - 1):min(cols, left + shape[1] + 1)] = True


def _noise_cells(rng, free, level, window_rects):
    cells = np.argwhere(free)
    n = len(cells)
    if level == "none" or n == 0:
        return cells[:0]
    if level == "sparse":
        count = int(rng.integers(1, max(2, int(0.05 * n)) + 1))
    elif level == "medium":
        count = int(round(n * rng.uniform(0.06, 0.13)))
    else:
        count = int(round(n * rng.uniform(0.14, 0.23)))
    count = max(1, min(count, n - 1)) if n > 1 else n
    style = str(rng.choice(("scatter", "cluster", "adjacent", "lines")))
    if style == "scatter":
        chosen = rng.choice(n, size=count, replace=False)
        return cells[chosen]
    if style == "cluster":
        centers = cells[rng.choice(n, size=min(n, int(rng.integers(1, 5))), replace=False)]
        dist = np.min(np.abs(cells[:, None, :] - centers[None, :, :]).sum(axis=2), axis=1)
        score = dist + rng.random(n) * 2.0
    elif style == "adjacent":
        dist = np.full(n, 99.0)
        for top, left, bottom, right in window_rects:
            dy = np.maximum.reduce((top - cells[:, 0], cells[:, 0] - bottom, np.zeros(n)))
            dx = np.maximum.reduce((left - cells[:, 1], cells[:, 1] - right, np.zeros(n)))
            dist = np.minimum(dist, dy + dx)
        score = dist + rng.random(n) * 2.5
    else:
        score = np.full(n, 99.0)
        for _ in range(int(rng.integers(2, 7))):
            row, col = cells[int(rng.integers(n))]
            if rng.random() < 0.5:
                line = (cells[:, 0] == row) & (np.abs(cells[:, 1] - col) <= int(rng.integers(1, 3)))
            else:
                line = (cells[:, 1] == col) & (np.abs(cells[:, 0] - row) <= int(rng.integers(1, 3)))
            score = np.where(line, rng.random(n), score)
        score = score + rng.random(n) * 0.5
    chosen = np.argpartition(score, count - 1)[:count]
    return cells[chosen]


def _attempt(rng, spec, bounds):
    palette = spec["palette"]
    background, rail, marker = palette
    orientation = spec["orientation"]
    arrangement = spec["arrangement"]
    shape = _sample_size(rng, spec["regime"], orientation, arrangement, bounds)
    decoys = list(spec["required_decoys"]) + list(spec["optional_decoys"])
    canvas, positions = _layout(rng, shape, orientation, arrangement, bool(decoys), spec["edge"])
    first, second = _payload_pair(rng, _payload_shape(*shape, orientation), spec["relation"])
    grid = np.full(canvas, background, dtype=int)
    occupied = np.zeros(canvas, dtype=bool)
    blocked = np.zeros(canvas, dtype=bool)
    rects = []
    for (top, left), payload in zip(positions, (first, second)):
        grid[top:top + shape[0], left:left + shape[1]] = _window_grid(shape, orientation, palette, payload)
        occupied[top:top + shape[0], left:left + shape[1]] = True
        _block(blocked, top, left, shape)
        rects.append((top, left, top + shape[0] - 1, left + shape[1] - 1))
    # In the shared arrangement the second window rewrites the common rail line with identical
    # rail and cap cells, so both windows remain complete.
    placed = []
    max_axis = max(2, min(12, max(canvas) - 3))
    for number, kind in enumerate(decoys):
        required = number < len(spec["required_decoys"])
        if kind == "broken":
            congruent = required or rng.random() < 0.6
            if congruent:
                patch = _broken_decoy(rng, shape, orientation, palette)
            else:
                o = VERTICAL if rng.random() < 0.5 else HORIZONTAL
                size = (int(rng.integers(4, 9)), int(rng.integers(3, 8)))
                size = size if o == VERTICAL else size[::-1]
                patch = _broken_decoy(rng, size, o, palette)
        elif kind == "misaligned":
            patch = _misaligned_decoy(rng, palette, min(8, max_axis))
        elif kind == "lone":
            patch = _lone_decoy(rng, palette, min(10, max_axis))
        else:
            patch = None
        if kind == "specks":
            for _ in range(int(rng.integers(1, 4))):
                where = _place_patch(rng, blocked, (1, 1))
                if where is None:
                    break
                grid[where] = rail
                occupied[where] = True
                _block(blocked, where[0], where[1], (1, 1))
                placed.append(kind)
            if required and kind not in placed:
                raise _Retry("speck placement")
            continue
        where = _place_patch(rng, blocked, patch.shape)
        if where is None:
            if required:
                raise _Retry("decoy placement")
            continue
        top, left = where
        grid[top:top + patch.shape[0], left:left + patch.shape[1]] = patch
        occupied[top:top + patch.shape[0], left:left + patch.shape[1]] = True
        _block(blocked, top, left, patch.shape)
        placed.append(kind)
    noise = _noise_cells(rng, ~occupied, spec["noise"], rects)
    if len(noise):
        grid[noise[:, 0], noise[:, 1]] = marker
    found = _find_windows(grid)
    expected = sorted(rects)
    if [w["bounds"] for w in found] != expected:
        raise _Retry("window uniqueness")
    if any(w["palette"] != palette or w["orientation"] != orientation for w in found):
        raise _Retry("roles")
    if not _background_is_plurality(grid, background):
        raise _Retry("background plurality")
    output = solve(grid)
    meta = {"shape": shape, "orientation": orientation, "arrangement": arrangement,
            "relation": spec["relation"], "decoys": sorted(set(placed)), "noise": int(len(noise)),
            "palette": palette, "canvas": canvas, "rects": expected, "area": shape[0] * shape[1]}
    return {"input": grid, "output": output}, meta


def _generate(seed, num_train, num_test):
    seed = _as_int(seed, "seed", 0, None)
    num_train, num_test = _resolve_counts(num_train, num_test)
    master = np.random.default_rng(seed)
    for _restart in range(12):
        rng = np.random.default_rng(int(master.integers(0, 2 ** 63 - 1)))
        plan = _plan(rng, num_train, num_test)
        try:
            puzzle, metas = _build(rng, plan, num_train)
            validate_generated_puzzle(puzzle)
            return puzzle, plan, metas
        except (_Retry, AssertionError, ValueError):
            continue
    raise RuntimeError("could not construct a valid puzzle")


def _build(rng, plan, num_train):
    puzzle = {"train": [], "test": []}
    metas = []
    seen = set()
    for spec in plan["specs"]:
        bounds = {"min_payload": 8 if spec["index"] == 0 else 6}
        train_metas = metas[:num_train]
        if plan["shift"] == "scale_up":
            if spec["training"]:
                bounds["max_area"] = 56
            else:
                bounds["min_area"] = max(m["area"] for m in train_metas) + 1
        elif plan["shift"] == "scale_down":
            if spec["training"]:
                bounds["min_area"] = 30
            else:
                bounds["max_area"] = min(m["area"] for m in train_metas) - 1
        for _ in range(300):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
            try:
                pair, meta = _attempt(pair_rng, spec, bounds)
            except _Retry:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key in seen:
                continue
            seen.add(key)
            puzzle["train" if spec["training"] else "test"].append(pair)
            metas.append(meta)
            break
        else:
            raise _Retry("pair construction")
    return puzzle, metas


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _plan_info, _metas = _generate(seed, num_train, num_test)
    return puzzle


# ---------------------------------------------------------------------------------------------
# Validation (reads only the returned grids)
# ---------------------------------------------------------------------------------------------

def _maximal_runs(work, rail):
    height, width = work.shape
    groups = {}
    for col in range(width):
        row = 0
        while row < height:
            if work[row, col] != rail:
                row += 1
                continue
            start = row
            while row < height and work[row, col] == rail:
                row += 1
            if row - start >= 2:
                groups.setdefault((start, row), []).append(col)
    return groups


def _rail_only_frames(grid, rail):
    """Rail-pair frames that ignore caps: what a solver skipping cap verification would accept."""
    frames = []
    for orientation in (VERTICAL, HORIZONTAL):
        work = grid if orientation == VERTICAL else grid.T
        for (start, stop), columns in _maximal_runs(work, rail).items():
            if start == 0 or stop == work.shape[0]:
                continue
            for i, left in enumerate(columns):
                for right in columns[i + 1:]:
                    if right - left < 2:
                        continue
                    crop = work[start - 1:stop + 1, left:right + 1]
                    if np.count_nonzero(crop == rail) != 2 * (stop - start):
                        continue
                    bounds = (start - 1, left, stop, right) if orientation == VERTICAL else \
                        (left, start - 1, right, stop)
                    frames.append((bounds, orientation))
    return frames


def _distractors(grid, info):
    background, rail, marker = info["palette"]
    inside = np.zeros(grid.shape, dtype=bool)
    for window in info["windows"]:
        top, left, bottom, right = window["bounds"]
        inside[top:bottom + 1, left:right + 1] = True
    true_bounds = {w["bounds"] for w in info["windows"]}
    extra = [f for f in _rail_only_frames(grid, rail) if f[0] not in true_bounds]
    stray = int(np.count_nonzero((grid == rail) & ~inside))
    noise = int(np.count_nonzero((grid == marker) & ~inside))
    return {"extra_frames": extra, "stray_rail": stray, "noise": noise}


def _substitution_changes_answer(grid, info, output, frame_bounds):
    """Would pairing a cap-less look-alike frame with either true window give a different answer?"""
    background, rail, marker = info["palette"]
    decoy = _crop(grid, frame_bounds)
    if decoy.shape != info["shape"]:
        return False
    payload = ~info["frame"]
    decoy_marks = decoy == marker
    for marks in info["marks"]:
        guess = output.copy()
        guess[payload] = np.where((marks ^ decoy_marks)[payload], marker, background)
        if np.array_equal(guess, output):
            return False
    return True


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "split lists"
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts"
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            grids = []
            for role in ("input", "output"):
                grid = np.asarray(pair[role])
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "integer 2D grid"
                assert 1 <= grid.shape[0] <= MAX_SIDE and 1 <= grid.shape[1] <= MAX_SIDE, "grid size"
                assert grid.min() >= 0 and grid.max() <= 9, "colour range"
                grids.append(grid.astype(int))
            x, y = grids
            key = (x.shape, x.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert len(np.unique(x)) == 3, "inputs use exactly background, rail and marker"
            info = _analyse(x)
            assert _background_is_plurality(x, info["palette"][0]), "background is the most frequent colour"
            assert np.array_equal(solve(x), y), "output must be the registered XOR window"
            assert y.shape == info["shape"] and y.shape != x.shape, "output is the window crop size"
            records[split].append((x, y, info, _distractors(x, info)))
    # Demonstrations: the first training pair shows all four overlay cases.
    first_cases = records["train"][0][2]["cases"]
    assert all(first_cases[k] > 0 for k in ("neither", "first_only", "second_only", "both")), "truth table"
    # Simpler alternatives are contradicted by training.
    for operation in ("union", "intersection", "first", "second"):
        assert any(not np.array_equal(_combine(info, operation), y) for _, y, info, _ in records["train"]), operation
    # Cap-less look-alike frames appear in a test only after training shows that they are ignored.
    if any(d["extra_frames"] for *_, d in records["test"]):
        assert any(_substitution_changes_answer(x, info, y, bounds)
                   for x, y, info, d in records["train"] for bounds, _ in d["extra_frames"]), \
            "training must show a congruent frame with missing caps being ignored"
    if any(d["stray_rail"] for *_, d in records["test"]):
        assert any(d["stray_rail"] for *_, d in records["train"]), "rail-coloured distractors demonstrated"
    # Windows that share a rail appear in a test only after training shows them.
    def shares_rail(info):
        a, b = (w["bounds"] for w in info["windows"])
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
    if any(shares_rail(info) for _, _, info, _ in records["test"]):
        assert any(shares_rail(info) for _, _, info, _ in records["train"]), "shared rail demonstrated"


# ---------------------------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------------------------

def self_test() -> None:
    shifts, orientations, arrangements, relations, decoys = set(), set(), set(), set(), set()
    shapes, canvases, palettes = set(), set(), set()
    mixed_palette = mixed_orientation = mixed_shape = shortcut_union = 0
    decoy_trap = boundary_tests = edge_touch = zero_noise = heavy_noise = 0
    fingerprints = set()
    for seed in range(42):
        puzzle, plan, metas = _generate(seed, 4, 1)
        again = generate_puzzle(seed, 4, 1)
        fingerprint = tuple((p["input"].tobytes(), p["output"].tobytes())
                            for split in ("train", "test") for p in puzzle[split])
        assert fingerprint == tuple((p["input"].tobytes(), p["output"].tobytes())
                                    for split in ("train", "test") for p in again[split]), "determinism"
        fingerprints.add(fingerprint)
        validate_generated_puzzle(puzzle)
        shifts.add(plan["shift"])
        pairs = puzzle["train"] + puzzle["test"]
        for pair, meta in zip(pairs, metas):
            info = _analyse(pair["input"])
            orientations.add(info["orientation"])
            arrangements.add(meta["arrangement"])
            relations.add(meta["relation"])
            decoys.update(meta["decoys"])
            shapes.add(info["shape"])
            canvases.add(pair["input"].shape)
            palettes.add(info["palette"])
            rows, cols = pair["input"].shape
            edge_touch += any(r[0] == 0 or r[1] == 0 or r[2] == rows - 1 or r[3] == cols - 1
                              for r in meta["rects"])
            noise = _distractors(pair["input"], info)["noise"]
            zero_noise += noise == 0
            heavy_noise += noise >= 40
        train_infos = [_analyse(p["input"]) for p in puzzle["train"]]
        test_infos = [_analyse(p["input"]) for p in puzzle["test"]]
        mixed_palette += len({i["palette"] for i in train_infos + test_infos}) > 1
        mixed_orientation += len({i["orientation"] for i in train_infos + test_infos}) > 1
        mixed_shape += len({i["shape"] for i in train_infos + test_infos}) > 1
        shortcut_union += any(not np.array_equal(_combine(i, "union"), p["output"])
                              for i, p in zip(test_infos, puzzle["test"]))
        boundary_tests += any(m["relation"] in _BOUNDARY_RELATIONS for m in metas[len(puzzle["train"]):])
        for pair in puzzle["train"]:
            info = _analyse(pair["input"])
            extra = _distractors(pair["input"], info)["extra_frames"]
            if any(_substitution_changes_answer(pair["input"], info, pair["output"], b) for b, _ in extra):
                decoy_trap += 1
                break
        # Shift-specific train/test properties.
        if plan["shift"] == "scale_up":
            assert min(m["area"] for m in metas[4:]) > max(m["area"] for m in metas[:4])
        if plan["shift"] == "scale_down":
            assert max(m["area"] for m in metas[4:]) < min(m["area"] for m in metas[:4])
        if plan["shift"] == "orientation_holdout":
            assert {i["orientation"] for i in train_infos}.isdisjoint({i["orientation"] for i in test_infos})
        if plan["shift"] == "arrangement_shift":
            assert {m["arrangement"] for m in metas[:4]}.isdisjoint({m["arrangement"] for m in metas[4:]})
    assert len(fingerprints) == 42
    assert shifts == set(_SHIFTS), shifts
    assert orientations == {VERTICAL, HORIZONTAL}
    assert arrangements == {"side", "stacked", "shared"}
    assert relations >= {"ordinary_full", "ordinary", "shifted"} | set(_BOUNDARY_RELATIONS), relations
    assert decoys == set(_DECOY_KINDS), decoys
    assert len(shapes) >= 60 and len(canvases) >= 80 and len(palettes) >= 60
    assert mixed_palette >= 25 and mixed_orientation >= 20 and mixed_shape >= 38
    assert shortcut_union >= 20 and decoy_trap >= 12 and boundary_tests >= 12
    assert edge_touch >= 40 and zero_noise >= 5 and heavy_noise >= 15
    # Every legal explicit count, plus rejection of illegal ones.
    for train_count, test_count in ((2, 1), (10, 4), (5, 4), (3, 2)):
        puzzle = generate_puzzle(7000 + train_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for train_count, test_count in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(7100, train_count, test_count)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {train_count}/{test_count}")
