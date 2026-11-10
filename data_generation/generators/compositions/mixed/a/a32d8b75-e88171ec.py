"""Exchange-tile mosaics clipped to the largest blank square's interior.

Only Python/NumPy are used. Generation paints blocks forward; solve decodes
pixels and uses inverse coordinates. No artifacts or execution traces are saved.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_DIRECTIONS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _largest_square(canvas, blank):
    """Dynamic programming; ties between positions are rejected, never ordered."""
    h, w = canvas.shape
    sizes = np.zeros((h + 1, w + 1), dtype=int)
    best, winners = 0, []
    for r in range(h):
        for c in range(w):
            if canvas[r, c] == blank:
                s = 1 + min(sizes[r, c], sizes[r, c + 1], sizes[r + 1, c])
                sizes[r + 1, c + 1] = s
                if s > best:
                    best, winners = int(s), [(r - s + 1, c - s + 1, int(s))]
                elif s == best:
                    winners.append((r - s + 1, c - s + 1, int(s)))
    _require(best >= 3 and len(winners) == 1, "need one largest blank square of side >=3")
    return tuple(map(int, winners[0]))


def _maximal_squares(canvas, blank):
    """Validation search using integral images, independent of the DP search.

    Locally maximal means not contained in a solid square of side s+1 at
    any of the four possible containing offsets. Grid edges block growth.
    """
    h, w = canvas.shape
    ii = np.pad((canvas != blank).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    solid = {}
    for s in range(3, min(h, w) + 1):
        solid[s] = (ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]) == 0
    result = []
    for s, mask in solid.items():
        larger = solid.get(s + 1)
        for r, c in np.argwhere(mask):
            extendable = False
            if larger is not None:
                for dr, dc in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
                    y, x = r + dr, c + dc
                    if 0 <= y < larger.shape[0] and 0 <= x < larger.shape[1] and larger[y, x]:
                        extendable = True
                        break
            if not extendable:
                result.append((int(r), int(c), s))
    return result


def _largest_rectangle(canvas, blank):
    """Competing rule: maximum-area blank rectangle, sides at least three."""
    h, w = canvas.shape
    best, winner = 0, None
    for top in range(h):
        columns = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            columns &= canvas[bottom] == blank
            if bottom - top + 1 < 3:
                continue
            padded = np.pad(columns.astype(int), (1, 1))
            starts = np.flatnonzero(np.diff(padded) == 1)
            ends = np.flatnonzero(np.diff(padded) == -1)
            for left, right in zip(starts, ends):
                area = (bottom - top + 1) * (right - left)
                if right - left >= 3 and area > best:
                    best, winner = int(area), (top, bottom + 1, int(left), int(right))
    return best, winner


def _decode_panel(panel, frame):
    blank = int(panel[0, 0])
    _require(blank != frame, "panel ground equals frame")
    rows = np.flatnonzero(np.any(panel == frame, axis=1))
    groups = [list(g) for _, g in itertools.groupby(rows, key=lambda v, ct=itertools.count(): int(v) - next(ct))]
    _require(len(groups) == 4, "panel must have four separate framed fields")
    fields, rebuilt, origins = [], np.full(panel.shape, blank, int), []
    for group in groups:
        top, bottom = int(group[0]), int(group[-1]) + 1
        cols = np.flatnonzero(np.any(panel[top:bottom] == frame, axis=0))
        left, right = int(cols[0]), int(cols[-1]) + 1
        _require(top >= 1 and bottom < len(panel) and left >= 1 and right < panel.shape[1],
                 "fields must have visible surrounding ground")
        box = panel[top:bottom, left:right]
        _require(min(box.shape) >= 3 and np.all(box[0] == frame) and np.all(box[-1] == frame)
                 and np.all(box[:, 0] == frame) and np.all(box[:, -1] == frame), "broken field frame")
        interior = box[1:-1, 1:-1]
        _require(not np.any(interior == frame), "frame occurs inside a field")
        fields.append(interior.copy())
        rebuilt[top:bottom, left:right] = box
        origins.append((top, left))
    _require(np.array_equal(rebuilt, panel), "unframed panel content")
    tile, stencil_area = fields[:2]
    _require(tile.shape[0] == tile.shape[1] and 2 <= len(tile) <= 5
             and len(np.unique(tile)) == 2, "first field must be a complete binary square tile")
    stencil = stencil_area != blank
    _require(stencil.any() and len(np.unique(stencil_area[stencil])) == 1, "second field must be a monochrome stencil")
    _require(stencil[0].any() and stencil[-1].any() and stencil[:, 0].any() and stencil[:, -1].any(),
             "stencil must have a tight occupied bounding box")
    info = {}
    for field in fields[2:]:
        _require(field.shape == (3, 3), "controls must be framed 3x3 fields")
        points = set(map(tuple, np.argwhere(field != blank)))
        _require(len(np.unique(field[field != blank])) == 1, "control ink must be monochrome")
        for anchor, (ar, ac) in enumerate(_CORNERS):
            if points == {(2 * ar, 2 * ac)}:
                _require("anchor" not in info, "duplicate corner control")
                info["anchor"] = anchor
        for turn, neighbor in enumerate(_DIRECTIONS):
            if points == {(1, 1), neighbor}:
                _require("turn" not in info, "duplicate direction control")
                info["turn"] = turn
    _require(set(info) == {"anchor", "turn"}, "invalid controls")
    return dict(info, tile=tile, stencil=stencil, blank=blank, frame=int(frame), origins=origins)


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    candidates = []
    for side in ("left", "right"):
        for p in range(7, min(11, w - 3)):
            col = p if side == "left" else w - p - 1
            frame = int(grid[0, col])
            if not np.all(grid[:, col] == frame):
                continue
            panel = grid[:, :p] if side == "left" else grid[:, w - p:]
            try:
                info = _decode_panel(panel, frame)
                canvas = grid[:, p + 1:] if side == "left" else grid[:, :col]
                _require(not np.any(canvas == frame), "frame color is reserved outside canvas")
            except (AssertionError, IndexError, ValueError):
                continue
            candidates.append(dict(info, canvas=canvas.copy(), side=side, panel_width=p))
    _require(len(candidates) == 1, "input must have one visibly decodable side panel")
    return candidates[0]


def _bounds(square):
    r, c, s = square
    return r, r + s, c, c + s


def _anchor_transform(anchor, transform):
    r, c = _CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return _CORNERS.index((r, c))


def _paint(canvas, tile, stencil, turn, anchor, bounds, behavior="normal"):
    """Forward block painter used for construction and counterfactual checks."""
    top, bottom, left, right = bounds
    inset = 0 if behavior in ("whole_square", "canvas_aperture") else 1
    if behavior == "canvas_aperture":
        top, bottom, left, right = 0, canvas.shape[0], 0, canvas.shape[1]
    top, bottom, left, right = top + inset, bottom - inset, left + inset, right - inset
    if behavior == "no_rotation":
        turn = 0
    elif behavior == "reverse_rotation":
        turn = -turn
    mask = np.rot90(stencil, -turn)
    if behavior == "fill_holes":
        mask = np.ones(mask.shape, dtype=bool)
    colors = np.unique(tile)
    payload = tile.copy() if behavior == "no_exchange" else colors[0] + colors[1] - tile
    if behavior == "rotate_tile":
        payload = np.rot90(payload, -turn)
    n = len(tile)
    mh, mw = np.array(mask.shape) * n
    ar, ac = _CORNERS[anchor]
    oy, ox = top + ar * (bottom - top - mh), left + ac * (right - left - mw)
    ylo, yhi, xlo, xhi = (0, len(canvas), 0, canvas.shape[1]) if behavior == "no_clip" else (top, bottom, left, right)
    ylo, yhi, xlo, xhi = max(0, ylo), min(len(canvas), yhi), max(0, xlo), min(canvas.shape[1], xhi)
    out, footprint = canvas.copy(), np.zeros(canvas.shape, dtype=bool)
    for mr, mc in np.argwhere(mask):
        y, x = int(oy + mr * n), int(ox + mc * n)
        a, b, c, d = max(ylo, y), min(yhi, y + n), max(xlo, x), min(xhi, x + n)
        if a < b and c < d:
            out[a:b, c:d] = payload[a - y:b - y, c - x:d - x]
            footprint[a:b, c:d] = True
    return out, footprint


def _inverse(info, square):
    """Cellwise inverse mapping: does not call the forward block painter."""
    canvas, tile, mask = info["canvas"], info["tile"], info["stencil"]
    r, c, s = square
    k, n = info["turn"], len(tile)
    sh, sw = mask.shape
    rh, rw = (sw, sh) if k % 2 else (sh, sw)
    ar, ac = _CORNERS[info["anchor"]]
    oy, ox = r + 1 + ar * (s - 2 - n * rh), c + 1 + ac * (s - 2 - n * rw)
    colors = np.unique(tile)
    out = canvas.copy()
    for y in range(r + 1, r + s - 1):
        for x in range(c + 1, c + s - 1):
            dy, dx = y - oy, x - ox
            if not (0 <= dy < n * rh and 0 <= dx < n * rw):
                continue
            mr, mc = dy // n, dx // n
            if k == 0:
                sy, sx = mr, mc
            elif k == 1:
                sy, sx = sh - 1 - mc, mr
            elif k == 2:
                sy, sx = sh - 1 - mr, sw - 1 - mc
            else:
                sy, sx = mc, sw - 1 - mr
            if mask[sy, sx]:
                out[y, x] = int(colors[0] + colors[1] - tile[dy % n, dx % n])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the visible input and the fixed family rule."""
    info = _decode(input_grid)
    return _inverse(info, _largest_square(info["canvas"], info["blank"]))


def _predict_variant(info, square, behavior="normal", convention=(1, 0, 0)):
    bounds = _bounds(square)
    if behavior == "largest_rectangle":
        _, bounds = _largest_rectangle(info["canvas"], info["blank"])
    sign, offset, transform = convention
    return _paint(info["canvas"], info["tile"], info["stencil"],
                  (sign * info["turn"] + offset) % 4,
                  _anchor_transform(info["anchor"], transform), bounds, behavior)[0]


_ALTERNATIVES = ("no_exchange", "rotate_tile", "no_rotation", "reverse_rotation",
                 "fill_holes", "whole_square", "canvas_aperture", "no_clip", "largest_rectangle")


def _training_evidence(train):
    decoded = []
    for pair in train[:2]:
        info = _decode(pair["input"])
        decoded.append((info, _largest_square(info["canvas"], info["blank"]), np.asarray(pair["output"])))
    rejected = {}
    for behavior in _ALTERNATIVES:
        rejected[behavior] = any(not np.array_equal(_predict_variant(info, sq, behavior), out)
                                 for info, sq, out in decoded)
    _require(all(rejected.values()), "training fails a shortcut witness: " + str(rejected))
    survivors = []
    for sign, offset, transform in itertools.product((1, -1), range(4), range(8)):
        convention = sign, offset, transform
        if all(np.array_equal(_predict_variant(info, sq, convention=convention), out)
               for info, sq, out in decoded):
            survivors.append(convention)
    _require(survivors == [(1, 0, 0)], "ambiguous direction/corner geometry in training")
    return rejected


def _pattern(rng, h, w):
    style = int(rng.integers(5))
    rr, cc = np.indices((h, w))
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(.25, .8)
    elif style == 1:
        mask = (rr == 0) | (cc == 0) | ((rr == h - 1) & (cc > w // 2))
    elif style == 2:
        mask = (rr % 2 == 0) & (cc % 2 == 0)
    elif style == 3:
        mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        mask[int(rng.integers(h)), int(rng.integers(w))] = False
    else:
        mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
    # Keep full support visible, without prescribing particular occupied corners.
    for r in (0, h - 1):
        if not mask[r].any():
            mask[r, int(rng.integers(w))] = True
    for c in (0, w - 1):
        if not mask[:, c].any():
            mask[int(rng.integers(h)), c] = True
    return mask


def _panel(rng, h, p, blank, frame, tile, stencil, ink, turn, anchor):
    direction = np.full((3, 3), blank, int)
    direction[1, 1] = direction[_DIRECTIONS[turn]] = ink[1]
    corner = np.full((3, 3), blank, int)
    ar, ac = _CORNERS[anchor]
    corner[2 * ar, 2 * ac] = ink[2]
    keys = [corner, direction]
    if rng.integers(2):
        keys.reverse()
    fields = [tile, np.where(stencil, ink[0], blank), *keys]
    minimum = sum(len(a) + 2 for a in fields) + 5
    gaps = 1 + rng.multinomial(h - minimum, [.2] * 5)
    result, r = np.full((h, p), blank, int), int(gaps[0])
    for index, a in enumerate(fields):
        fh, fw = np.array(a.shape) + 2
        c = int(rng.integers(1, p - fw))
        result[r:r + fh, c:c + fw] = frame
        result[r + 1:r + fh - 1, c + 1:c + fw - 1] = a
        r += int(fh + gaps[index + 1])
    return result


def _field(rng, h, w, blank, colors):
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        ix = rng.integers(len(colors), size=(h, w))
    elif style == 1:
        ix = (rr // int(rng.integers(1, 4)) + cc // int(rng.integers(2, 6))) % len(colors)
    elif style == 2:
        ix = (rr + cc // int(rng.integers(1, 5))) % len(colors)
    else:
        ix = np.zeros((h, w), int)
        for _ in range(6):
            y, x = int(rng.integers(h)), int(rng.integers(w))
            ix[y:y + int(rng.integers(2, 7)), x:x + int(rng.integers(2, 7))] = int(rng.integers(len(colors)))
    result = np.asarray(colors)[ix].copy()
    probability = rng.uniform(.16, .46)
    result[rng.random((h, w)) < probability] = blank
    # Thin runs can touch other blank patches; no artificial solid halo.
    for _ in range(int(rng.integers(2, 6))):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        if rng.integers(2):
            result[y:y + int(rng.integers(2, 5)), x] = blank
        else:
            result[y, x:x + int(rng.integers(2, 6))] = blank
    return result


def _canvas(rng, h, w, s, blank, textures):
    """A target square and a wider rectangle of strictly greater area."""
    dh = s - 2
    dw = s * s // dh + 1
    if dw > w or s + dh + 1 > h:
        return None
    grid = _field(rng, h, w, blank, textures)
    spare = h - s - dh - 1
    gaps = rng.multinomial(spare, [1 / 3] * 3)
    if rng.integers(2):
        sr, dr = int(gaps[0]), int(gaps[0] + s + 1 + gaps[1])
    else:
        dr, sr = int(gaps[0]), int(gaps[0] + dh + 1 + gaps[1])
    sc, dc = int(rng.integers(w - s + 1)), int(rng.integers(w - dw + 1))
    grid[sr:sr + s, sc:sc + s] = blank
    grid[dr:dr + dh, dc:dc + dw] = blank
    # Sometimes add another, smaller solid square, without changing either patch.
    if rng.random() < .6:
        q = int(rng.integers(3, min(6, s)))
        for _ in range(12):
            y, x = int(rng.integers(h - q + 1)), int(rng.integers(w - q + 1))
            if not (y < sr + s + 1 and y + q > sr - 1 and x < sc + s + 1 and x + q > sc - 1) and not (
                    y < dr + dh + 1 and y + q > dr - 1 and x < dc + dw + 1 and x + q > dc - 1):
                grid[y:y + q, x:x + q] = blank
                break
    # Ensure an incidental blank cell touches the square when there is room.
    contacts = [(sr - 1, sc + s // 2), (sr + s, sc + s // 2),
                (sr + s // 2, sc - 1), (sr + s // 2, sc + s)]
    contacts = [(y, x) for y, x in contacts if 0 <= y < h and 0 <= x < w]
    y, x = contacts[int(rng.integers(len(contacts)))]
    grid[y, x] = blank
    try:
        if _largest_square(grid, blank) != (sr, sc, s):
            return None
    except AssertionError:
        return None
    return grid, (sr, sc, s)


def _dimensions(rng, regime, witness):
    if witness or regime == "fit":
        n = int(rng.choice([2, 3]))
        sh, sw = (2, 3) if rng.integers(2) else (3, 2)
        if regime == "fit":
            d = 3 * n + int(rng.integers(1, 4))
        else:
            d = 2 * n + int(rng.integers(1, n))
    elif regime == "one_axis":
        n = int(rng.integers(2, 5))
        sh, sw = sorted(map(int, rng.choice(np.arange(2, 6), size=2, replace=False)))
        if rng.integers(2):
            sh, sw = sw, sh
        choices = [d for d in range(n * min(sh, sw) + 1, min(12, n * max(sh, sw) - 1) + 1) if d % n]
        if not choices:
            return None
        d = int(rng.choice(choices))
    else:
        n = int(rng.integers(2, 6))
        sh, sw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        choices = [d for d in range(max(3, n + 1), min(12, n * min(sh, sw) - 1) + 1) if d % n]
        d = int(rng.choice(choices))
    return n, sh, sw, d + 2


def _make_pair(rng, blank, frame, control, regime, witness=False):
    turn, anchor = control
    for _ in range(240):
        dims = _dimensions(rng, regime, witness)
        if dims is None:
            continue
        n, sh, sw, s = dims
        pmin = max(n, sw, 3) + 4
        hmin = max(n + sh + 19, 2 * s - 1)
        if pmin > 10 or hmin > 30:
            continue
        p = int(rng.integers(pmin, 11))
        dw = s * s // (s - 2) + 1
        wmin = max(s, dw, 14)
        if wmin > 29 - p:
            continue
        h, w = int(rng.integers(hmin, 31)), int(rng.integers(wmin, 30 - p))
        palette = [int(c) for c in rng.permutation(10) if c not in (blank, frame)]
        tile_choices = [blank, *palette] if not witness and rng.random() < .25 else palette
        colors = rng.choice(tile_choices, size=2, replace=False)
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if not bits.any() or bits.all() or np.array_equal(bits, np.rot90(bits, 2)):
            continue
        tile = np.where(bits, colors[0], colors[1]).astype(int)
        mask = _pattern(rng, sh, sw)
        if mask.all() or np.array_equal(mask, np.rot90(mask, 2)):
            continue
        textures = palette[:int(rng.integers(1, 4))]
        built = _canvas(rng, h, w, s, blank, textures)
        if built is None:
            continue
        canvas, square = built
        out, footprint = _paint(canvas, tile, mask, turn, anchor, _bounds(square))
        # Both inherited tile roles and an actually exposed transparent macrocell.
        r, c, _ = square
        interior = np.zeros(canvas.shape, bool)
        interior[r + 1:r + s - 1, c + 1:c + s - 1] = True
        _, filled = _paint(canvas, tile, np.ones(mask.shape, bool), turn, anchor, _bounds(square))
        if not footprint.any() or len(np.unique(out[footprint])) != 2 or not np.any(filled & ~footprint):
            continue
        if not np.any(out != canvas):
            continue
        info = dict(canvas=canvas, tile=tile, stencil=mask, turn=turn, anchor=anchor)
        required = ["no_exchange", "fill_holes", "whole_square", "canvas_aperture", "largest_rectangle"]
        if regime != "fit":
            required.append("no_clip")
        if witness and turn % 2:
            required += ["rotate_tile", "no_rotation", "reverse_rotation"]
        info["blank"] = blank
        if any(np.array_equal(_predict_variant(info, square, behavior), out) for behavior in required):
            continue
        panel = _panel(rng, h, p, blank, frame, tile, mask,
                       [int(rng.choice(palette)) for _ in range(3)], turn, anchor)
        separator = np.full((h, 1), frame, int)
        side = "left" if rng.integers(2) else "right"
        inp = np.concatenate([panel, separator, canvas] if side == "left" else [canvas, separator, panel], axis=1)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded pair sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    blank, frame = map(int, rng.choice(10, size=2, replace=False))
    quarter, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    first_controls = [(quarter, anchor), (even, (anchor + int(rng.choice([1, 3]))) % 4)]
    puzzle, seen, train_controls = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train":
                control = first_controls[i] if i < 2 else (int(rng.integers(4)), int(rng.integers(4)))
                regime = "fit" if i == 0 or (i >= 2 and rng.random() < .35) else "one_axis"
                train_controls.add(control)
            else:
                held = sorted(set(itertools.product(range(4), repeat=2)) - train_controls)
                control = held[int(rng.integers(len(held)))]
                regime = "two_axes"
            for _ in range(8):
                pair = _make_pair(rng, blank, frame, control, regime, split == "train" and i < 2)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate retry exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    """Compact review features recovered entirely from returned pixels."""
    info = _decode(pair["input"])
    canvas, blank = info["canvas"], info["blank"]
    sq = _largest_square(canvas, blank)
    r, c, s = sq
    n = len(info["tile"])
    mask = np.rot90(info["stencil"], -info["turn"])
    mh, mw = np.array(mask.shape) * n
    _, footprint = _paint(canvas, info["tile"], info["stencil"], info["turn"], info["anchor"], _bounds(sq))
    return {"canvas_shape": tuple(canvas.shape), "side": info["side"], "blank": blank,
            "frame": info["frame"], "square": sq, "tile_size": n,
            "stencil_shape": tuple(info["stencil"].shape), "control": (info["turn"], info["anchor"]),
            "clipped_axes": int(mh > s - 2) + int(mw > s - 2),
            "partial_tile": bool((s - 2) % n), "painted_cells": int(footprint.sum()),
            "changed_cells": int(np.count_nonzero(np.asarray(pair["output"]) != canvas)),
            "blank_tile_role": bool(blank in info["tile"]),
            "local_candidates": len(_maximal_squares(canvas, blank)),
            "larger_rectangle_area": _largest_rectangle(canvas, blank)[0]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, roles, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            canvas, blank = info["canvas"], info["blank"]
            sq = _largest_square(canvas, blank)
            candidates = _maximal_squares(canvas, blank)
            largest = max(x[2] for x in candidates)
            _require([x for x in candidates if x[2] == largest] == [sq], "ambiguous maximal-square search")
            expected, footprint = _paint(canvas, info["tile"], info["stencil"], info["turn"], info["anchor"], _bounds(sq))
            _require(out.shape == canvas.shape and np.array_equal(out, expected), "incorrect forward label")
            _require(np.array_equal(out, _inverse(info, sq)), "inverse reconstruction disagrees")
            r, c, s = sq
            interior = np.zeros(canvas.shape, bool)
            interior[r + 1:r + s - 1, c + 1:c + s - 1] = True
            _require(np.array_equal(out[~interior], canvas[~interior]), "perimeter or exterior changed")
            _require(footprint.any() and np.any(out != canvas), "no visible overlay")
            _require(len(np.unique(out[footprint])) == 2, "tile roles are not both visible")
            _, filled = _paint(canvas, info["tile"], np.ones(info["stencil"].shape, bool),
                               info["turn"], info["anchor"], _bounds(sq))
            _require(np.any(filled & ~footprint), "no exposed transparent stencil cell")
            _require(np.all(out[filled & ~footprint] == blank), "stencil holes were filled")
            area, _ = _largest_rectangle(canvas, blank)
            _require(area > s * s and len(candidates) >= 2, "missing geometric competitors")
            roles.add((blank, info["frame"]))
            features[split].append(pair_features(pair))
    _require(len(roles) == 1, "blank/frame roles must stay fixed within a puzzle")
    train = features["train"]
    _require(train[0]["clipped_axes"] == 0 and train[1]["clipped_axes"] == 1,
             "first demonstrations must teach fitting then clipping")
    _require(all(f["clipped_axes"] <= 1 for f in train), "training clipping regime violated")
    controls = {f["control"] for f in train}
    _require(all(f["clipped_axes"] == 2 and f["partial_tile"] and f["control"] not in controls
                 for f in features["test"]), "test must recombine controls and clip both axes")
    _training_evidence(puzzle["train"])


def self_test():
    """42 seeds, deterministic replay, boundary cases, and causal mutations.

    Puzzles are processed one at a time; retained diagnostics are sets of small
    feature tuples and SHA-256 digests, not duplicated grids or saved traces.
    """
    shapes, positions, tile_sizes, stencil_shapes, controls, sides = set(), set(), set(), set(), set(), set()
    roles, structures, puzzle_digests = set(), set(), set()
    clipping, blank_tiles = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                digest.update(bytes(inp.shape) + inp.astype(np.uint8).tobytes())
                f = pair_features(pair)
                shapes.add(f["canvas_shape"])
                tile_sizes.add(f["tile_size"])
                stencil_shapes.add(f["stencil_shape"])
                controls.add(f["control"])
                sides.add(f["side"])
                roles.add((f["blank"], f["frame"]))
                clipping.add(f["clipped_axes"])
                blank_tiles.add(f["blank_tile_role"])
                r, c, s = f["square"]
                h, w = f["canvas_shape"]
                positions.add((r <= (h - s) / 2, c <= (w - s) / 2))
                mapping = {}
                norm = bytes(mapping.setdefault(int(v), len(mapping)) for v in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + norm).digest())
                _require(np.array_equal(solve(inp.tolist()), pair["output"]), "input-only solve failed")
        puzzle_digests.add(digest.digest())
    _require(len(puzzle_digests) == 42 and len(structures) >= 200, "insufficient structural variation")
    _require(len(shapes) >= 20 and len(stencil_shapes) >= 10 and len(roles) >= 20, "insufficient geometry/palette variation")
    _require(tile_sizes == {2, 3, 4, 5} and controls == set(itertools.product(range(4), repeat=2)), "missing size/control coverage")
    _require(sides == {"left", "right"} and len(positions) == 4 and clipping == {0, 1, 2}
             and blank_tiles == {False, True}, "missing compatible family variation")
    p, q = generate_puzzle(7301, 2, 1), generate_puzzle(7301, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], q[split])
                 for k in ("input", "output")), "nondeterministic generation")
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        result = generate_puzzle(900 + nt + nq, nt, nq)
        _require(len(result["train"]) == nt and len(result["test"]) == nq, "count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Boundary search: side three at a grid edge; ties have no silent tie-break.
    canvas = np.ones((7, 8), int)
    canvas[:3, :3] = 0
    _require(_largest_square(canvas, 0) == (0, 0, 3), "side-three edge square failed")
    canvas[-3:, -3:] = 0
    try:
        _largest_square(canvas, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("square tie was accepted")
    # Relocate the square while keeping the A instructions, and mutate a tile.
    info = _decode(p["train"][0]["input"])
    sq = _largest_square(info["canvas"], info["blank"])
    texture = next(c for c in range(10) if c not in (info["blank"], info["frame"]))
    ca, cb = (np.full(info["canvas"].shape, texture, int) for _ in range(2))
    s = sq[2]
    ca[:s, :s] = info["blank"]
    cb[-s:, -s:] = info["blank"]
    ia, ib = dict(info, canvas=ca), dict(info, canvas=cb)
    qa, qb = _largest_square(ca, info["blank"]), _largest_square(cb, info["blank"])
    _require(qa != qb and not np.array_equal(_inverse(ia, qa) != ca, _inverse(ib, qb) != cb),
             "moving B's square does not move A's painted footprint")
    mutated = dict(info, tile=info["tile"].copy())
    colors = np.unique(info["tile"])
    mutated["tile"] = colors[0] + colors[1] - info["tile"]
    _require(not np.array_equal(_inverse(info, sq), _inverse(mutated, sq)), "tile contribution has no effect")
    damaged = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Full input-only solves at small apertures and negative mosaic origins.
    # This exercises broader legal supports than the informative random sampler.
    rng = np.random.default_rng(414)
    for n in range(2, 6):
        tile = np.full((n, n), 2, int)
        tile[0, -1] = tile[-1, 0] = 3
        stencil = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
        for turn, anchor in itertools.product(range(4), repeat=2):
            panel = _panel(rng, 30, 10, 0, 9, tile, stencil, [4, 2, 4], turn, anchor)
            for side in (3, 4, 7, 12):
                canvas = np.ones((30, 19), int)
                square = (30 - side, 19 - side, side)
                canvas[-side:, -side:] = 0
                inp = np.concatenate([canvas, np.full((30, 1), 9, int), panel], axis=1)
                answer = _paint(canvas, tile, stencil, turn, anchor, _bounds(square))[0]
                _require(np.array_equal(solve(inp), answer), "small/edge aperture or negative-origin phase failed")
