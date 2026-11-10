"""Span-derived stencils: exact-run rectangles become exchanged-tile mosaics.

All conventions are fixed or visible. No parent module, seed, training output,
or generation metadata is used by solve. Only NumPy and the standard library.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


# Clockwise order, starting at top left; direction key starts at up.
_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_DIRECTIONS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _integer(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _runs(row, ground):
    """Maximal runs, with exclusive right endpoints."""
    result, c = [], 0
    while c < len(row):
        color = int(row[c])
        e = c + 1
        while e < len(row) and row[e] == color:
            e += 1
        if color != ground:
            result.append((color, c, e))
        c = e
    return result


def _rectangles(coarse, ground):
    rows = [_runs(row, ground) for row in coarse]
    result = []
    for r, runs in enumerate(rows):
        previous = set(rows[r - 1]) if r else set()
        for color, left, right in runs:
            if (color, left, right) in previous:
                continue
            end = r + 1
            while end < len(rows) and (color, left, right) in rows[end]:
                end += 1
            result.append((r, end, left, right, color))
    return result


def _stencil(coarse, ground, behavior="normal"):
    """Rectangular macro-cells retain every span, with one-site gutters.

    A rectangle [r:s,c:e] occupies [2r:2s-1,2c:2e-1]. The entire
    (2R-1,2C-1) extent is retained, including blank coarse edge slots.
    """
    h, w = coarse.shape
    mask = np.zeros((2 * h - 1, 2 * w - 1), bool)
    if behavior in ("no_vertical_merge", "loose_vertical_merge"):
        rows = [_runs(row, ground) for row in coarse]
        for r, runs in enumerate(rows):
            for color, left, right in runs:
                mask[2 * r, 2 * left:2 * right - 1] = True
        for r in range(h - 1):
            for color, left, right in rows[r]:
                for other, low, high in rows[r + 1]:
                    if color != other:
                        continue
                    if behavior == "loose_vertical_merge":
                        a, b = max(left, low), min(right, high)
                        if a < b:
                            mask[2 * r + 1, 2 * a:2 * b - 1] = True
        return mask
    if behavior == "coarse_occupancy":
        return coarse != ground
    for r, end, left, right, _ in _rectangles(coarse, ground):
        if behavior == "collapse_spans":
            mask[2 * r, 2 * left] = True
        else:
            mask[2 * r:2 * end - 1, 2 * left:2 * right - 1] = True
    return mask


def _features(coarse, ground):
    rows = [_runs(row, ground) for row in coarse]
    return {
        "horizontal": any(e - c >= 2 for runs in rows for _, c, e in runs),
        "exact": any(set(a) & set(b) for a, b in zip(rows, rows[1:])),
        "unequal": any(x == y and (a, b) != (c, d) and max(a, c) < min(b, d)
                       for upper, lower in zip(rows, rows[1:])
                       for x, a, b in upper for y, c, d in lower),
    }


def _frames(panel, divider):
    """Locate closed divider frames whose interiors contain no divider."""
    h, w = panel.shape
    boxes = []
    for top in range(h - 3):
        for left in range(w - 3):
            if panel[top, left] != divider or (left and panel[top, left - 1] == divider):
                continue
            right = left
            while right < w and panel[top, right] == divider:
                right += 1
            if right - left < 4:
                continue
            for bottom in range(top + 3, h):
                if not (np.all(panel[top:bottom + 1, left] == divider)
                        and np.all(panel[top:bottom + 1, right - 1] == divider)):
                    break
                if (np.all(panel[bottom, left:right] == divider)
                        and not np.any(panel[top + 1:bottom, left + 1:right - 1] == divider)):
                    boxes.append((top, bottom + 1, left, right))
                    break
    return sorted(boxes)


def _decode_panel(panel, divider):
    ground = int(panel[0, 0])
    if ground == divider or not np.all(panel[0] == ground) or not np.all(panel[-1] == ground):
        raise ValueError("panel must have visible blank end rows")
    boxes = _frames(panel, divider)
    if len(boxes) != 4:
        raise ValueError("panel needs tile, lattice and two bounded keys")
    (t, b, l, r), code_box, key1, key2 = boxes
    tile = panel[t + 1:b - 1, l + 1:r - 1].copy()
    if tile.shape[0] != tile.shape[1] or not 2 <= len(tile) <= 5 or len(np.unique(tile)) != 2:
        raise ValueError("tile must be a complete two-color square")
    ct, cb, cl, cr = code_box
    if ct <= b:
        raise ValueError("tile and lattice require a blank separating row")
    code = panel[ct + 1:cb - 1, cl + 1:cr - 1]
    if (min(code.shape) < 3 or any(n % 2 != 1 for n in code.shape)
            or not np.all(code[::2] == ground) or not np.all(code[:, ::2] == ground)):
        raise ValueError("invalid alternating lattice scaffold")
    coarse = code[1::2, 1::2].copy()
    if not np.any(coarse != ground):
        raise ValueError("lattice must have a payload")
    if key1[0] <= cb or key2[0] != key1[1] - 1 or key1[2:] != key2[2:]:
        raise ValueError("keys need a shared visible frame edge below the lattice")
    info = {}
    rebuilt = np.full(panel.shape, ground, int)
    for box in boxes:
        top, bottom, left, right = box
        rebuilt[top:bottom, left:right] = divider
        rebuilt[top + 1:bottom - 1, left + 1:right - 1] = panel[top + 1:bottom - 1, left + 1:right - 1]
    if not np.array_equal(rebuilt, panel):
        raise ValueError("unframed panel content")
    for top, bottom, left, right in (key1, key2):
        field = panel[top + 1:bottom - 1, left + 1:right - 1]
        if field.shape != (3, 3):
            raise ValueError("key fields must be 3 by 3")
        points = set(map(tuple, np.argwhere(field != ground)))
        vals = np.unique(field[field != ground])
        if len(vals) != 1:
            raise ValueError("a key must use one marker color")
        corners = [{(2 * y, 2 * x)} for y, x in _CORNERS]
        directions = [{(1, 1), neighbor} for neighbor in _DIRECTIONS]
        if points in corners and "anchor" not in info:
            info["anchor"] = corners.index(points)
        elif points in directions and "turn" not in info:
            info["turn"] = directions.index(points)
        else:
            raise ValueError("invalid or duplicated geometric key")
    if set(info) != {"anchor", "turn"}:
        raise ValueError("missing geometric control")
    info.update(tile=tile, coarse=coarse, ground=ground, divider=divider,
                panel_width=panel.shape[1], boxes=tuple(boxes))
    return info


def _decode_input(grid):
    """Recover panels from visible frames and full-height separator columns."""
    h, w = grid.shape
    sides = {}
    for side in ("left", "right"):
        found = []
        for width in range(5, min(14, w - 1)):
            col = width if side == "left" else w - width - 1
            divider = int(grid[0, col])
            if not np.all(grid[:, col] == divider):
                continue
            panel = grid[:, :width] if side == "left" else grid[:, w - width:]
            try:
                ins = _decode_panel(panel, divider)
            except ValueError:
                continue
            found.append((width, ins))
        if len(found) > 1:
            raise ValueError("ambiguous panel extent")
        if found:
            sides[side] = found[0]
    if not sides:
        raise ValueError("no readable side panel")
    left = sides["left"][0] + 1 if "left" in sides else 0
    right = w - sides["right"][0] - 1 if "right" in sides else w
    if right <= left:
        raise ValueError("missing canvas")
    canvas = grid[:, left:right].copy()
    panels = [sides[s][1] for s in ("left", "right") if s in sides]
    if len({p["divider"] for p in panels}) != 1 or np.any(canvas == panels[0]["divider"]):
        raise ValueError("structural divider is reserved outside the canvas")
    return canvas, panels, "flank" if len(panels) == 2 else next(iter(sides))


def _anchor_map(anchor, transform):
    y, x = _CORNERS[anchor]
    if transform >= 4:
        x = 1 - x
    for _ in range(transform % 4):
        y, x = x, 1 - y
    return y, x


def _paint(canvas, panels, behavior="normal", convention=(1, 0, 0)):
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    for ins in panels:
        mask = _stencil(ins["coarse"], ins["ground"], behavior)
        if behavior == "crop_stencil":
            coords = np.argwhere(mask)
            lo, hi = coords.min(0), coords.max(0) + 1
            mask = mask[lo[0]:hi[0], lo[1]:hi[1]]
        sign, offset, transform = convention
        turn = (sign * ins["turn"] + offset) % 4
        if behavior == "no_rotation":
            turn = 0
        mask = np.rot90(mask, -turn)
        if behavior == "fill_holes":
            mask = np.ones(mask.shape, bool)
        tile = ins["tile"]
        colors = np.unique(tile)
        swapped = colors[0] + colors[1] - tile
        if behavior == "no_exchange":
            swapped = tile
        if behavior == "rotate_tile":
            swapped = np.rot90(swapped, -turn)
        n = len(tile)
        mh, mw = np.array(mask.shape) * n
        if mh > canvas.shape[0] or mw > canvas.shape[1]:
            raise ValueError("mosaic does not fit the canvas")
        ay, ax = _anchor_map(ins["anchor"], transform)
        if behavior == "top_left":
            ay, ax = 0, 0
        y, x = ay * (len(canvas) - mh), ax * (canvas.shape[1] - mw)
        for r, c in np.argwhere(mask):
            block = np.s_[y + r * n:y + (r + 1) * n, x + c * n:x + (c + 1) * n]
            if behavior == "normal" and convention == (1, 0, 0) and covered[block].any():
                raise ValueError("overlapping panel footprints")
            if behavior in ("only_first", "only_second"):
                patch = out[block]
                active = swapped == colors[int(behavior == "only_second")]
                patch[active] = swapped[active]
            else:
                out[block] = swapped
            covered[block] = True
    return out, covered


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the complete transformation from this input alone."""
    canvas, panels, _ = _decode_input(_grid(input_grid))
    return _paint(canvas, panels)[0]


def _bridge_stencil(coarse, ground):
    """Validation formulation: cells plus horizontal and exact vertical bridges."""
    h, w = coarse.shape
    m = np.zeros((2 * h - 1, 2 * w - 1), bool)
    for r in range(h):
        for c in range(w):
            if coarse[r, c] == ground:
                continue
            m[2 * r, 2 * c] = True
            if c + 1 < w and coarse[r, c + 1] == coarse[r, c]:
                m[2 * r, 2 * c + 1] = True
    for r in range(h - 1):
        for color, left, right in set(_runs(coarse[r], ground)) & set(_runs(coarse[r + 1], ground)):
            m[2 * r + 1, 2 * left:2 * right - 1] = True
    return m


def _inverse_paint(canvas, panels):
    """Check output pixels by inverse rotation, without forward block stamping."""
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    rr, cc = np.indices(canvas.shape)
    for ins in panels:
        source = _bridge_stencil(ins["coarse"], ins["ground"])
        h, w = source.shape
        k, n = ins["turn"], len(ins["tile"])
        rh, rw = (w, h) if k % 2 else (h, w)
        assert rh * n <= len(canvas) and rw * n <= canvas.shape[1], "mosaic exceeds canvas"
        ay, ax = _CORNERS[ins["anchor"]]
        y, x = rr - ay * (len(canvas) - rh * n), cc - ax * (canvas.shape[1] - rw * n)
        inside = (y >= 0) & (x >= 0) & (y < rh * n) & (x < rw * n)
        r, c = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
        if k == 0:
            sy, sx = r, c
        elif k == 1:
            sy, sx = h - 1 - c, r
        elif k == 2:
            sy, sx = h - 1 - r, w - 1 - c
        else:
            sy, sx = c, w - 1 - r
        active = inside & source[sy, sx]
        if np.any(active & covered):
            raise AssertionError("intersecting painted blocks")
        colors = np.unique(ins["tile"])
        values = colors.sum() - ins["tile"][y % n, x % n]
        out[active] = values[active]
        covered |= active
    return out, covered


_ALTERNATIVES = ("no_vertical_merge", "loose_vertical_merge", "coarse_occupancy", "collapse_spans",
                 "no_exchange", "rotate_tile", "fill_holes", "only_first",
                 "only_second", "no_rotation", "top_left", "crop_stencil")


def _training_evidence(decoded):
    """Finite, explicit alternatives; this is not independent semantic review."""
    for behavior in _ALTERNATIVES:
        distinguished = False
        for canvas, panels, expected in decoded:
            try:
                predicted = _paint(canvas, panels, behavior)[0]
                distinguished |= not np.array_equal(predicted, expected)
            except ValueError:
                distinguished = True
        if not distinguished:
            raise AssertionError(f"training does not distinguish {behavior}")
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = sign, offset, transform
                try:
                    matches = all(np.array_equal(_paint(c, p, convention=convention)[0], o)
                                  for c, p, o in decoded)
                except ValueError:
                    matches = False
                if matches:
                    survivors.append(convention)
    if survivors != [(1, 0, 0)]:
        raise AssertionError("training leaves direction/corner conventions ambiguous")


def _coarse(rng, shape, ground, colors, role):
    h, w = shape
    for _ in range(64):
        a = np.full(shape, ground, int)
        density = float(rng.uniform(.2, .9))
        occupied = rng.random(shape) < density
        a[occupied] = rng.choice(colors, size=int(occupied.sum()))
        kind = role if role != "random" else str(rng.choice(["exact", "unequal", "return", "bars", "random"]))
        color = int(rng.choice(colors))
        if kind in ("exact", "unequal") and h >= 2:
            spans = [(l, e) for l in range(w) for e in range(l + 1, w + 1)]
            start = int(rng.integers(h - 1))
            first = spans[int(rng.integers(len(spans)))]
            if kind == "exact":
                second = first
            else:
                options = [s for s in spans if s != first and max(s[0], first[0]) < min(s[1], first[1])]
                if not options:
                    continue
                second = options[int(rng.integers(len(options)))]
            for r, (l, e) in ((start, first), (start + 1, second)):
                a[r, :] = rng.choice([ground] + [c for c in colors if c != color], size=w)
                a[r, l:e] = color
            if kind == "exact" and h > start + 2 and rng.random() < .45:
                for r in range(start + 2, h):
                    if rng.random() < .3:
                        break
                    a[r] = a[start]
        elif kind == "return" and h >= 3:
            col = int(rng.integers(w))
            a[0, col] = a[-1, col] = color
            a[1:-1, col] = ground
        elif kind == "bars":
            for r in range(0, h, 2):
                l = int(rng.integers(w))
                a[r, l:] = color
        if not np.any(a != ground):
            a[int(rng.integers(h)), int(rng.integers(w))] = color
        f = _features(a, ground)
        if role in ("exact", "unequal") and not f[role]:
            continue
        return a
    raise RuntimeError("could not sample coarse runs")


def _framed(content, divider):
    box = np.full((content.shape[0] + 2, content.shape[1] + 2), divider, int)
    box[1:-1, 1:-1] = content
    return box


def _panel(rng, h, width, ins, marker):
    ground, divider = ins["ground"], ins["divider"]
    tile, coarse = ins["tile"], ins["coarse"]
    code = np.full((2 * coarse.shape[0] + 1, 2 * coarse.shape[1] + 1), ground, int)
    code[1::2, 1::2] = coarse
    tile_box, code_box = _framed(tile, divider), _framed(code, divider)
    keys = []
    for kind in rng.permutation(2):
        field = np.full((3, 3), ground, int)
        if kind == 0:
            ay, ax = _CORNERS[ins["anchor"]]
            field[2 * ay, 2 * ax] = marker
        else:
            field[1, 1] = marker
            field[_DIRECTIONS[ins["turn"]]] = marker
        keys.append(_framed(field, divider))
    key_box = np.concatenate([keys[0], keys[1][1:]], axis=0)
    minimum = len(tile_box) + len(code_box) + 9 + 4
    spare = h - minimum
    if spare < 0:
        raise ValueError("panel objects exceed available height")
    gaps = np.ones(4, int) + rng.multinomial(spare, np.full(4, .25))
    panel = np.full((h, width), ground, int)
    top = int(gaps[0])
    for index, box in enumerate((tile_box, code_box, key_box)):
        left = int(rng.integers(width - box.shape[1] + 1))
        panel[top:top + len(box), left:left + box.shape[1]] = box
        top += len(box) + int(gaps[index + 1])
    return panel


def _canvas(rng, h, w, colors, simple):
    rr, cc = np.indices((h, w))
    style = int(rng.integers(3 if simple else 6))
    if style == 0:
        index = np.zeros((h, w), int)
    elif style == 1:
        index = rr // int(rng.integers(2, 8))
    elif style == 2:
        index = cc // int(rng.integers(2, 7))
    elif style == 3:
        index = rr // int(rng.integers(3, 8)) + cc // int(rng.integers(2, 7))
    elif style == 4:
        index = (rr + cc) // int(rng.integers(2, 6))
    else:
        index = np.minimum.reduce([rr, h - rr - 1, cc, w - cc - 1])
    return np.asarray(colors, int)[index % len(colors)]


def _make_pair(rng, controls, roles, layout, witness):
    count = len(controls)
    for _ in range(180):
        ground, divider = map(int, rng.choice(10, 2, replace=False))
        ink = [c for c in range(10) if c not in (ground, divider)]
        instructions, widths = [], []
        for (turn, anchor), role in zip(controls, roles):
            n = int(rng.choice([2, 2, 3] if count == 2 else [2, 2, 3, 4, 5]))
            if witness:
                rows = int(rng.choice([2, 2, 3]))
                cols = 2 if count == 2 else int(rng.integers(2, 5))
            else:
                rows = int(rng.integers(1, min(5, (30 - n - 18) // 2) + 1))
                cols = int(rng.integers(1, 4 if count == 2 else 6))
            width = max(2 * cols + 3, n + 2, 5) + int(rng.integers(0, 2))
            if width > 13:
                break
            colors = list(map(int, rng.choice(ink, size=int(rng.integers(1, min(5, rows * cols) + 1)), replace=False)))
            coarse = _coarse(rng, (rows, cols), ground, colors, role)
            tile_colors = rng.choice([c for c in range(10) if c != divider], 2, replace=False)
            bits = rng.random((n, n)) < rng.uniform(.25, .75)
            if not bits.any() or bits.all() or (witness and np.array_equal(bits, np.rot90(bits, 2))):
                break
            tile = np.where(bits, tile_colors[0], tile_colors[1])
            ins = dict(tile=tile, coarse=coarse, turn=turn, anchor=anchor,
                       ground=ground, divider=divider)
            mask = _stencil(coarse, ground)
            if witness and (mask.all() or np.array_equal(mask, np.rot90(mask, 2))):
                break
            instructions.append(ins)
            widths.append(width)
        if len(instructions) != count:
            continue
        min_h = max(len(p["tile"]) + 2 * p["coarse"].shape[0] + 18 for p in instructions)
        max_w = 30 - sum(widths) - count
        sizes = [np.array(np.rot90(_stencil(p["coarse"], ground), -p["turn"]).shape) * len(p["tile"])
                 for p in instructions]
        min_h = max(min_h, max(int(s[0]) + int(witness) for s in sizes))
        min_w = max(5, max(int(s[1]) + int(witness) for s in sizes))
        if min_h > 30 or min_w > max_w:
            continue
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, max_w + 1))
        # Contrasting backgrounds keep both tile roles visibly overwritten in witnesses.
        used = {int(c) for p in instructions for c in np.unique(p["tile"])}
        pool = [c for c in range(10) if c != divider and (not witness or c not in used)]
        canvas_colors = rng.choice(pool, size=int(rng.integers(1, min(4, len(pool)) + 1)), replace=False)
        canvas = _canvas(rng, h, w, canvas_colors, witness)
        try:
            output, footprint = _paint(canvas, instructions)
            if not footprint.any() or not np.any(output != canvas):
                continue
            panels = [_panel(rng, h, width, p, int(rng.choice(ink)))
                      for width, p in zip(widths, instructions)]
            div = np.full((h, 1), divider, int)
            parts = ([panels[0], div, canvas, div, panels[1]] if count == 2 else
                     [panels[0], div, canvas] if layout == "left" else [canvas, div, panels[0]])
            inp = np.concatenate(parts, axis=1)
            recovered, decoded, _ = _decode_input(inp)
            if not np.array_equal(recovered, canvas) or not np.array_equal(_inverse_paint(recovered, decoded)[0], output):
                raise AssertionError("visible construction failed to round trip")
        except ValueError:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded panel sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Create one coherent puzzle; all counts and retries have finite bounds."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer("num_train", num_train, 2, 10)
    nq = _integer("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    quarter, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    adjacent = (anchor + int(rng.choice([1, 3]))) % 4
    witnesses = {(quarter, anchor), (quarter, adjacent), (even, anchor)}
    held = [(k, a) for k in range(4) for a in range(4) if (k, a) not in witnesses]
    held = held[int(rng.integers(len(held)))]
    pool = [(k, a) for k in range(4) for a in range(4) if (k, a) != held]
    for _ in range(24):
        puzzle, seen = {"train": [], "test": []}, set()
        # Three visibly independent controls in two examples, with both A witnesses.
        first = [([(quarter, anchor), (quarter, adjacent)], ["exact", "unequal"], "flank"),
                 ([(even, anchor)], ["random"], str(rng.choice(["left", "right"])))]
        if rng.random() < .5:
            first.reverse()
        try:
            for split, count in (("train", nt), ("test", nq)):
                for index in range(count):
                    witness = split == "train" and index < 2
                    if witness:
                        controls, roles, layout = first[index]
                    else:
                        layout = str(rng.choice(["left", "right", "flank"]))
                        panels = 2 if layout == "flank" else 1
                        controls = [pool[int(rng.integers(len(pool)))] for _ in range(panels)]
                        if split == "test":
                            controls[0] = held
                        if panels == 2 and controls[0][1] == controls[1][1]:
                            options = [(k, a) for k, a in pool if a != controls[0][1]]
                            controls[1] = options[int(rng.integers(len(options)))]
                        roles = ["random"] * panels
                    for _ in range(8):
                        pair = _make_pair(rng, controls, roles, layout, witness)
                        key = _key(pair["input"])
                        if key not in seen:
                            break
                    else:
                        raise RuntimeError("duplicate input sampling exhausted")
                    seen.add(key)
                    puzzle[split].append(pair)
            validate_generated_puzzle(puzzle)
            return puzzle
        except (RuntimeError, AssertionError):
            continue
    raise RuntimeError("bounded informative-puzzle sampling exhausted")


def validate_generated_puzzle(puzzle):
    """Check visible syntax, span geometry, output pixels, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, teaching, features = set(), [], Counter()
    train_controls, test_controls = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            canvas, panels, _ = _decode_input(inp)
            predicted, covered = _inverse_paint(canvas, panels)
            assert out.shape == canvas.shape and np.array_equal(out, predicted), "wrong output"
            assert np.any(covered) and np.any(out != canvas)
            assert np.array_equal(out[~covered], canvas[~covered]), "transparent cells changed"
            controls = {(p["turn"], p["anchor"]) for p in panels}
            for p in panels:
                m = _stencil(p["coarse"], p["ground"])
                assert np.array_equal(m, _bridge_stencil(p["coarse"], p["ground"]))
                # Check that full rectangle widths/depths, not singleton markers, survive.
                for r, e, c, d, _ in _rectangles(p["coarse"], p["ground"]):
                    assert m[2 * r:2 * e - 1, 2 * c:2 * d - 1].all()
                if split == "train":
                    features.update(k for k, v in _features(p["coarse"], p["ground"]).items() if v)
            if split == "train":
                teaching.append((canvas, panels, out))
                train_controls.update(controls)
            else:
                test_controls.append(controls)
    assert all(features[k] for k in ("horizontal", "exact", "unequal")), "missing run witnesses"
    assert all(c - train_controls for c in test_controls), "test lacks a held-out control combination"
    _training_evidence(teaching)


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """Bounded in-memory checks over 42 seeds; returns compact measurements."""
    shapes, coarse_shapes, layouts, controls, sizes, structures, digests = (set() for _ in range(7))
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                canvas, panels, layout = _decode_input(pair["input"])
                shapes.add(canvas.shape)
                layouts.add(layout)
                stats["pairs"] += 1
                stats["multi_panel"] += len(panels) == 2
                for p in panels:
                    c, g = p["coarse"], p["ground"]
                    coarse_shapes.add(c.shape)
                    controls.add((p["turn"], p["anchor"]))
                    sizes.add(len(p["tile"]))
                    mapping = {g: 0}
                    norm = tuple(mapping.setdefault(int(v), len(mapping)) for v in c.flat)
                    structures.add((c.shape, norm))
                    stats["exact_panels"] += _features(c, g)["exact"]
                    stats["unequal_panels"] += _features(c, g)["unequal"]
                    stats["blank_edge_panels"] += bool(np.all(c[0] == g) or np.all(c[-1] == g)
                                                       or np.all(c[:, 0] == g) or np.all(c[:, -1] == g))
                    stats["zero_payload_panels"] += bool(g != 0 and np.any(c == 0))
    assert len(digests) == 42 and len(shapes) >= 25 and len(coarse_shapes) >= 12
    assert layouts == {"left", "right", "flank"} and sizes == {2, 3, 4, 5}
    assert controls == {(k, a) for k in range(4) for a in range(4)}
    assert len(structures) >= 100 and stats["blank_edge_panels"] >= 10
    assert _puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact-span vs partial overlap and an interruption regression, without sampling.
    c = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1]])
    m = _stencil(c, 0)
    assert m[1, :3].all() and not m[3].any() and not m[5].any() and not m[7].any()
    assert np.array_equal(m, _bridge_stencil(c, 0))
    assert not np.array_equal(m, _stencil(c, 0, "loose_vertical_merge"))
    assert not np.array_equal(m, _stencil(c, 0, "collapse_spans"))
    assert not _stencil(np.array([[1, 1], [2, 2]]), 0)[1].any()
    damaged = generate_puzzle(17, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "puzzles": len(digests), "canvas_shapes": len(shapes),
            "coarse_shapes": len(coarse_shapes), "coarse_structures": len(structures),
            "controls": len(controls), "tile_sizes": sorted(sizes), **dict(stats)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
