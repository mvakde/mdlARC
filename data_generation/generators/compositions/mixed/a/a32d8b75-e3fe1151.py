"""Repair a shared tile histogram, exchange its colors, and stamp keyed mosaics.

Only NumPy and the standard library are used. No parent module or artifact is
read at runtime. Saved diagnostics are optional and belong in scratch/.
"""
from __future__ import annotations

import hashlib
import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_DIRECTIONS = ((0, 1), (1, 2), (2, 1), (1, 0))
_HOST_SEEDS = (0, 767501569, 773153749, 114696693, 978712124, 797255585,
               596063960, 700107391, 160347365, 505896768, 292595425,
               546982068, 672882419, 51213346, 689114752, 753177594,
               539832342, 185675964, 690968167, 142220287, 415082492,
               881301784, 92216478, 446432774, 70533914, 692926240,
               368509846, 435742106, 984061182, 414119089, 819897874,
               185716753, 67920913, 744624386, 811406916, 118889947,
               180255960, 219528202, 189945197, 782042039, 69379634,
               897613788)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and
             isinstance(value, (int, np.integer)), f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad grid color")
    return a.astype(np.int64, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _hist(a, structural):
    result = np.bincount(a.ravel(), minlength=10)
    result[structural] = 0
    return result


def _decode_panel(panel, structural):
    """Decode actual frame geometry, not generator offsets or split metadata."""
    h, w = panel.shape
    _require(5 <= w <= 8, "bad instruction strip width")
    ground = int(panel[0, 0])
    _require(ground != structural and np.all(panel[0] == ground), "bad panel ground")
    fields = []
    for r in range(1, h - 3):
        for c in range(1, w - 3):
            box = panel[r:r + 3, c:c + 3]
            if np.any(box == structural):
                continue
            if not (np.all(panel[r - 1, c - 1:c + 4] == structural)
                    and np.all(panel[r + 3, c - 1:c + 4] == structural)
                    and np.all(panel[r:r + 3, c - 1] == structural)
                    and np.all(panel[r:r + 3, c + 3] == structural)):
                continue
            pts = set(map(tuple, np.argwhere(box != ground)))
            colors = np.unique(box[box != ground])
            if len(colors) != 1:
                continue
            for anchor, (ar, ac) in enumerate(_CORNERS):
                if pts == {(2 * ar, 2 * ac)}:
                    fields.append((r, c, "anchor", anchor))
            for turn, point in enumerate(_DIRECTIONS):
                if pts == {(1, 1), point}:
                    fields.append((r, c, "turn", turn))
    _require(len(fields) == 2 and {f[2] for f in fields} == {"anchor", "turn"},
             "two bounded controls are required")
    key_top = min(f[0] for f in fields) - 1
    while key_top and np.all(panel[key_top - 1] == structural):
        key_top -= 1
    expected_keys = np.full(panel[key_top:].shape, structural, dtype=int)
    for r, c, _, _ in fields:
        expected_keys[r - key_top:r - key_top + 3, c:c + 3] = panel[r:r + 3, c:c + 3]
    _require(np.array_equal(expected_keys, panel[key_top:]), "stray pixels in key area")
    upper = panel[:key_top]
    rows = np.flatnonzero(np.any(upper != ground, axis=1))
    _require(len(rows) > 0, "missing tile frame")
    tr = int(rows[0])
    end = tr
    while end < key_top and np.any(upper[end] != ground):
        end += 1
    band = upper[tr:end]
    cols = np.flatnonzero(np.any(band != ground, axis=0))
    tc = int(cols[0])
    frame = band[:, tc:int(cols[-1]) + 1]
    _require(frame.shape[0] == frame.shape[1] and 4 <= len(frame) <= 6,
             "tile must have an explicit square frame")
    _require(all(np.all(edge == structural) for edge in
                 (frame[0], frame[-1], frame[:, 0], frame[:, -1])), "broken tile frame")
    raw = frame[1:-1, 1:-1].copy()
    _require(np.count_nonzero(raw == structural) == 1, "exactly one tile placeholder")
    _require(not np.any(raw == ground), "panel ground cannot be tile payload")
    rest = upper[end:]
    points = np.argwhere(rest != ground)
    _require(len(points) > 0, "missing stencil")
    lo, hi = points.min(0), points.max(0) + 1
    stencil_box = rest[lo[0]:hi[0], lo[1]:hi[1]]
    inks = np.unique(stencil_box[stencil_box != ground])
    _require(len(inks) == 1 and inks[0] != structural, "stencil must be monochrome")
    mask = stencil_box != ground
    reconstructed = np.full(upper.shape, ground, dtype=int)
    reconstructed[tr:end, tc:tc + len(frame)] = frame
    sr, sc = end + int(lo[0]), int(lo[1])
    _require(sr > end, "tile and stencil need a ground gap")
    reconstructed[sr:sr + mask.shape[0], sc:sc + mask.shape[1]] = stencil_box
    _require(np.array_equal(reconstructed, upper), "stray pixels in upper panel")
    result = {f[2]: f[3] for f in fields}
    result.update(raw=raw, stencil=mask, ground=ground, structural=structural,
                  tile_origin=(tr + 1, tc + 1), stencil_origin=(sr, sc),
                  key_origins=tuple((f[0], f[1]) for f in fields), width=w)
    return result


def _decode_input(input_grid):
    a = _grid(input_grid)
    candidates = []
    constant = np.flatnonzero(np.all(a == a[:1], axis=0))
    for structural in np.unique(a[0, constant]):
        separators = [int(c) for c in constant if a[0, c] == structural]
        if len(separators) != 2 or separators[0] == 0 or separators[-1] == a.shape[1] - 1:
            continue
        c1, c2 = separators
        parts = (a[:, :c1], a[:, c1 + 1:c2], a[:, c2 + 1:])
        panels, canvases = [], []
        for index, part in enumerate(parts):
            try:
                ins = _decode_panel(part, int(structural))
                panels.append((index, ins))
            except (ValueError, IndexError):
                canvases.append((index, part))
        if len(panels) != 2 or len(canvases) != 1:
            continue
        ci, canvas = canvases[0]
        if canvas.size == 0 or np.any(canvas == structural):
            continue
        if panels[0][1]["ground"] != panels[1][1]["ground"]:
            continue
        candidates.append((canvas.copy(), [p[1] for p in panels], ("RR", "LR", "LL")[ci]))
    _require(len(candidates) == 1, "input must have one unambiguous canvas and two panels")
    return candidates[0]


def _repair(panels):
    """Per-color maxima give the sole histogram if their sum fills the square."""
    shapes = {p["raw"].shape for p in panels}
    _require(len(shapes) == 1, "shared histograms need equal tile areas")
    structural = panels[0]["structural"]
    counts = [_hist(p["raw"], structural) for p in panels]
    target = np.maximum.reduce(counts)
    area = panels[0]["raw"].size
    _require(int(target.sum()) == area and np.count_nonzero(target) == 2,
             "common binary histogram is not uniquely determined")
    repaired, missing = [], []
    for panel, observed in zip(panels, counts):
        deficit = target - observed
        _require(np.all(deficit >= 0) and deficit.sum() == 1 and
                 np.count_nonzero(deficit) == 1, "ambiguous placeholder")
        color = int(np.flatnonzero(deficit)[0])
        tile = panel["raw"].copy()
        tile[tile == structural] = color
        repaired.append(dict(panel, tile=tile))
        missing.append(color)
    _require(len(set(missing)) == 2, "panels must omit different payload colors")
    return repaired, target


def _anchor(anchor, transform=0):
    r, c = _CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _predict(canvas, panels, convention=(1, 0, 0), behavior="normal"):
    """Input-only inverse-coordinate evaluation; construction uses forward stamps."""
    out = canvas.copy()
    covered = np.zeros(canvas.shape, dtype=bool)
    rr, cc = np.indices(canvas.shape)
    sign, offset, transform = convention
    for index, panel in enumerate(panels):
        tile, mask = panel["tile"], panel["stencil"]
        if behavior == "copy_other_tile":
            tile = panels[1 - index]["tile"]
        n, (sh, sw) = len(tile), mask.shape
        turn = (sign * panel["turn"] + offset) % 4
        if behavior == "no_rotation":
            turn = 0
        rh, rw = (sw, sh) if turn % 2 else (sh, sw)
        mh, mw = n * rh, n * rw
        _require(mh <= len(canvas) and mw <= canvas.shape[1], "mosaic does not fit")
        ar, ac = _anchor(panel["anchor"], transform)
        if behavior == "top_left":
            ar = ac = 0
        y, x = rr - ar * (len(canvas) - mh), cc - ac * (canvas.shape[1] - mw)
        inside = (y >= 0) & (x >= 0) & (y < mh) & (x < mw)
        mr, mc = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
        if turn == 0:
            sy, sx = mr, mc
        elif turn == 1:
            sy, sx = sh - 1 - mc, mr
        elif turn == 2:
            sy, sx = sh - 1 - mr, sw - 1 - mc
        else:
            sy, sx = mc, sw - 1 - mr
        active = inside & (np.ones(canvas.shape, dtype=bool) if behavior == "fill_holes" else mask[sy, sx])
        ty, tx = y % n, x % n
        if behavior == "rotate_tile":
            for _ in range(turn):
                ty, tx = n - 1 - tx, ty
        colors = np.unique(panel["tile"])
        source = tile
        if behavior in ("no_repair", "local_majority"):
            source = panel["raw"].copy()
            if behavior == "local_majority":
                observed = _hist(source, panel["structural"])
                source[source == panel["structural"]] = int(np.argmax(observed))
        original = source[ty, tx]
        values = np.where(original == colors[0], colors[1], colors[0])
        if behavior == "no_repair":
            values = np.where(original == panel["structural"], panel["structural"], values)
        if behavior == "no_exchange":
            values = original
        if behavior == "only_first":
            active &= values == colors[0]
        if behavior == "only_second":
            active &= values == colors[1]
        if convention == (1, 0, 0) and behavior == "normal":
            _require(not np.any(covered & active), "painted footprints overlap")
        out[active] = values[active]
        covered |= active
    return out, covered


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions and required data are visible in this input alone."""
    canvas, panels, _ = _decode_input(input_grid)
    repaired, _ = _repair(panels)
    return _predict(canvas, repaired)[0]


def _paint(canvas, panels):
    """Forward construction, independent of inverse stencil coordinate formulas."""
    out, covered = canvas.copy(), np.zeros(canvas.shape, dtype=bool)
    for panel in panels:
        tile, stencil = panel["tile"], np.rot90(panel["stencil"], -panel["turn"])
        n = len(tile)
        colors = np.unique(tile)
        swapped = np.where(tile == colors[0], colors[1], colors[0])
        mh, mw = n * stencil.shape[0], n * stencil.shape[1]
        _require(mh <= len(canvas) and mw <= canvas.shape[1], "forward mosaic fit")
        ar, ac = _CORNERS[panel["anchor"]]
        top, left = ar * (len(canvas) - mh), ac * (canvas.shape[1] - mw)
        for r, c in np.argwhere(stencil):
            block = np.s_[top + r * n:top + (r + 1) * n, left + c * n:left + (c + 1) * n]
            _require(not covered[block].any(), "forward footprint conflict")
            out[block] = swapped
            covered[block] = True
    return out, covered


def _make_panel(rng, h, width, ground, structural, raw, mask, ink, markers, turn, anchor):
    n = len(raw)
    a = np.full((h, width), ground, dtype=int)
    key_height = int(rng.integers(9, 13))
    key_top = h - key_height
    # Reserve a top margin, full tile frame, a ground gap, and a stencil.
    available = key_top - (n + 2) - mask.shape[0] - 1
    _require(available >= 1, "panel content fit")
    tr = int(rng.integers(1, available + 1))
    gap = int(rng.integers(1, available - tr + 2))
    sr = tr + n + 2 + gap
    tc = int(rng.integers(width - n - 1))
    a[tr:tr + n + 2, tc:tc + n + 2] = structural
    a[tr + 1:tr + n + 1, tc + 1:tc + n + 1] = raw
    sc = int(rng.integers(width - mask.shape[1] + 1))
    a[sr:sr + mask.shape[0], sc:sc + mask.shape[1]] = np.where(mask, ink, ground)
    a[key_top:] = structural
    first = key_top + 1 + int(rng.integers(key_height - 8))
    second = int(rng.integers(first + 4, h - 3))
    positions = [(first, int(rng.integers(1, width - 3))),
                 (second, int(rng.integers(1, width - 3)))]
    if rng.integers(2):
        positions.reverse()
    for kind, (r, c) in enumerate(positions):
        a[r:r + 3, c:c + 3] = ground
        if kind == 0:
            ar, ac = _CORNERS[anchor]
            a[r + 2 * ar, c + 2 * ac] = markers[0]
        else:
            a[r + 1, c + 1] = markers[1]
            dr, dc = _DIRECTIONS[turn]
            a[r + dr, c + dc] = markers[1]
    return a


def _tiles(rng, n, colors, structural, balanced):
    area = n * n
    first_count = area // 2 if balanced else int(rng.integers(1, area))
    values = np.array([colors[0]] * first_count + [colors[1]] * (area - first_count))
    for _ in range(64):
        tiles = [rng.permutation(values).reshape(n, n) for _ in range(2)]
        if any(np.array_equal(t, np.rot90(t, 2)) for t in tiles):
            continue
        missing = rng.permutation(colors)
        raw = []
        for tile, color in zip(tiles, missing):
            damaged = tile.copy()
            position = int(rng.choice(np.flatnonzero(tile.ravel() == color)))
            damaged.ravel()[position] = structural
            raw.append(damaged)
        # Visible disagreement proves that panels are not spatial copies.
        both_visible = (raw[0] != structural) & (raw[1] != structural)
        if np.any(both_visible & (raw[0] != raw[1])):
            return tiles, raw
    raise ValueError("could not sample differing tile arrangements")


def _mask(rng, h, w, witness):
    if witness:
        mask = np.ones((2, 2), dtype=bool)
        mask.ravel()[int(rng.integers(4))] = False
        return mask
    rows, cols = np.indices((h, w))
    style = int(rng.integers(6))
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(.25, .8)
    elif style == 1:
        mask = (rows == 0) | (cols == 0) | (rows == h - 1) | (cols == w - 1)
    elif style == 2:
        mask = (rows % 2 == 0) & (cols % 2 == 0)
    elif style == 3:
        mask = (rows == int(rng.integers(h))) | (cols == int(rng.integers(w)))
    elif style == 4:
        mask = np.ones((h, w), dtype=bool)
    else:
        mask = (cols == (rows % w)) | (rows == h - 1)
    if not mask.any():
        mask[0, 0] = True
    occupied = np.argwhere(mask)
    lo, hi = occupied.min(0), occupied.max(0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]]


def _canvas(rng, h, w, palette, witness):
    rows, cols = np.indices((h, w))
    if witness:
        return np.full((h, w), palette[0], dtype=int)
    style = int(rng.integers(6))
    if style == 0:
        indices = np.zeros((h, w), dtype=int)
    elif style == 1:
        indices = rows // int(rng.integers(2, 7)) + np.zeros_like(cols)
    elif style == 2:
        indices = cols // int(rng.integers(2, 7)) + np.zeros_like(rows)
    elif style == 3:
        indices = rows // int(rng.integers(2, 6)) + cols // int(rng.integers(2, 6))
    elif style == 4:
        indices = np.minimum.reduce((rows, h - rows - 1, cols, w - cols - 1))
    else:
        indices = np.zeros((h, w), dtype=int)
        for _ in range(4):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            indices[r:min(h, r + int(rng.integers(2, 9))),
                    c:min(w, c + int(rng.integers(2, 9)))] = int(rng.integers(1, 6))
    return np.asarray(palette)[indices % len(palette)]


def _make_pair(rng, n, controls, layout, witness=False, balanced=False):
    for _ in range(160):
        try:
            h = int(rng.integers(23, 31))
            # Informative demonstrations leave margins on both axes and room
            # for two adjacent-corner, size-2-by-2 stencil mosaics.
            min_width = 4 * n + 1 if witness else max(2 * n + 1, 8)
            widths = [int(rng.integers(max(5, n + 2), 9)) for _ in range(2)]
            max_width = 30 - sum(widths) - 2
            _require(max_width >= min_width, "canvas width fit")
            w = int(rng.integers(min_width, max_width + 1))
            palette = list(map(int, rng.choice(10, int(rng.integers(4, 8)), replace=False)))
            ground, structural, *inks = palette
            colors = inks[:2]
            tiles, raw = _tiles(rng, n, colors, structural, balanced)
            canvas_palette = [ground] if witness else list(rng.choice([ground] + inks,
                                  size=int(rng.integers(1, min(4, len(inks) + 1) + 1)), replace=False))
            canvas = _canvas(rng, h, w, canvas_palette, witness)
            panels, instructions = [], []
            for index, (turn, anchor) in enumerate(controls):
                max_sh = min(7, h - (n + 2) - 12 - 2,
                             (w if turn % 2 else h) // n)
                max_sw = min(widths[index], (h if turn % 2 else w) // n)
                _require(min(max_sh, max_sw) >= 1, "stencil fit")
                mask = _mask(rng, int(rng.integers(1, max_sh + 1)),
                             int(rng.integers(1, max_sw + 1)), witness)
                instructions.append(dict(tile=tiles[index], stencil=mask, turn=turn, anchor=anchor))
                panels.append(_make_panel(rng, h, widths[index], ground, structural,
                                          raw[index], mask, int(rng.choice(inks)),
                                          rng.choice(inks, 2), turn, anchor))
            output, _ = _paint(canvas, instructions)
            _require(np.any(output != canvas), "no visible painting")
            divider = np.full((h, 1), structural, dtype=int)
            parts = {"LR": (panels[0], divider, canvas, divider, panels[1]),
                     "LL": (panels[0], divider, panels[1], divider, canvas),
                     "RR": (canvas, divider, panels[0], divider, panels[1])}[layout]
            inp = np.concatenate(parts, axis=1)
            recovered, decoded, _ = _decode_input(inp)
            repaired, _ = _repair(decoded)
            _require(np.array_equal(recovered, canvas), "canvas decode mismatch")
            _require(np.array_equal(_predict(recovered, repaired)[0], output), "paint mismatch")
            return {"input": inp, "output": output}
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def _training_evidence(train):
    decoded = []
    for pair in train[:2]:
        canvas, panels, _ = _decode_input(pair["input"])
        repaired, _ = _repair(panels)
        decoded.append((canvas, repaired, np.asarray(pair["output"])))
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = sign, offset, transform
                try:
                    if all(np.array_equal(_predict(c, ps, convention)[0], out) for c, ps, out in decoded):
                        survivors.append(convention)
                except ValueError:
                    pass
    _require(survivors == [(1, 0, 0)], "demonstrations leave geometric conventions ambiguous")
    for behavior in ("no_repair", "local_majority", "copy_other_tile", "no_exchange",
                     "rotate_tile", "no_rotation", "fill_holes", "top_left", "only_first", "only_second"):
        differs = False
        for canvas, panels, expected in decoded:
            try:
                differs |= not np.array_equal(_predict(canvas, panels, behavior=behavior)[0], expected)
            except ValueError:
                differs = True
        _require(differs, f"training does not distinguish {behavior}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    quarter, even = int(rng.choice((1, 3))), int(rng.choice((0, 2)))
    a = int(rng.integers(4))
    b = (a + int(rng.choice((1, 3)))) % 4
    c = int(rng.choice([x for x in range(4) if x not in (a, b)]))
    witness_controls = (((quarter, a), (quarter, b)), ((even, a), (quarter, c)))
    used_controls = set(witness_controls[0] + witness_controls[1])
    held = tuple(map(int, rng.choice(np.array([(t, an) for t in range(4) for an in range(4)
                                             if (t, an) not in used_controls]))))
    puzzle = {"train": [], "test": []}
    for index, controls in enumerate(witness_controls):
        puzzle["train"].append(_make_pair(rng, index + 2, controls, str(rng.choice(("LR", "LL", "RR"))),
                                                witness=True, balanced=index == 0))
    _training_evidence(puzzle["train"])
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, remaining in (("train", num_train - 2), ("test", num_test)):
        for index in range(remaining):
            for _ in range(12):
                anchors = list(map(int, rng.choice(4, 2, replace=False)))
                controls = [(int(rng.integers(4)), anchor) for anchor in anchors]
                if split == "test" and index == 0:
                    controls[0] = held
                    if controls[1][1] == held[1]:
                        controls[1] = controls[1][0], (held[1] + 2) % 4
                if split == "train":
                    controls = [((turn + 1) % 4, anchor) if (turn, anchor) == held
                                else (turn, anchor) for turn, anchor in controls]
                pair = _make_pair(rng, int(rng.integers(2, 5)), controls,
                                  str(rng.choice(("LR", "LL", "RR"))), balanced=bool(rng.integers(2)))
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            canvas, panels, _ = _decode_input(inp)
            repaired, target = _repair(panels)
            structural = panels[0]["structural"]
            _require(all(np.array_equal(_hist(p["tile"], structural), target) for p in repaired),
                     "repaired histograms differ")
            common_visible = (panels[0]["raw"] != structural) & (panels[1]["raw"] != structural)
            _require(np.any(common_visible & (panels[0]["raw"] != panels[1]["raw"])),
                     "panels do not disprove spatial copying")
            expected, covered = _predict(canvas, repaired)
            _require(np.array_equal(out, expected), "incorrect generated output")
            _require(np.array_equal(out[~covered], canvas[~covered]), "uncovered canvas changed")
            _require(np.any(out != canvas), "painting has no effect")
            _require(not np.array_equal(_predict(canvas, repaired, behavior="no_repair")[0], out),
                     "B omission does not change the output")
            forward, _ = _paint(canvas, repaired)
            _require(np.array_equal(forward, expected), "forward/inverse disagreement")
    _training_evidence(puzzle["train"])


def self_test():
    """42 supplied seeds; retain only hashes and small feature sets in memory."""
    puzzle_hashes, shapes, layouts, histograms, controls, masks, structures = (set() for _ in range(7))
    sizes, structural_colors, canvas_styles = set(), set(), set()
    pair_count = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        train_controls = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                canvas, panels, layout = _decode_input(inp)
                repaired, target = _repair(panels)
                _require(np.array_equal(solve(inp.copy()), out), "solve failed")
                for role in ("input", "output"):
                    grid = pair[role]
                    digest.update(bytes(grid.shape) + grid.astype(np.uint8).tobytes())
                shapes.add((inp.shape, out.shape))
                layouts.add(layout)
                histograms.add(tuple(sorted(int(v) for v in target if v)))
                structural_colors.add(panels[0]["structural"])
                style = ("uniform" if np.all(canvas == canvas[0, 0]) else
                         "rows" if np.all(canvas == canvas[:, :1]) else
                         "cols" if np.all(canvas == canvas[:1]) else "other")
                canvas_styles.add(style)
                local_controls = {(p["turn"], p["anchor"]) for p in panels}
                controls.update(local_controls)
                if split == "train":
                    train_controls.update(local_controls)
                elif pair is puzzle["test"][0]:
                    _require(bool(local_controls - train_controls), "test lacks a held-out control combination")
                for panel in repaired:
                    sizes.add(len(panel["tile"]))
                    masks.add(_key(panel["stencil"]))
                    # Color-normalized tile geometry, independent of palette.
                    structures.add((panel["tile"].shape,
                                    (panel["tile"] == panel["tile"][0, 0]).tobytes()))
                pair_count += 1
        puzzle_hashes.add(digest.hexdigest())
    _require(len(puzzle_hashes) == 42, "cross-seed duplicate")
    _require(sizes == {2, 3, 4} and layouts == {"LR", "LL", "RR"}, "size/layout diversity")
    _require(len(shapes) >= 60 and len(masks) >= 50 and len(structures) >= 50, "structural diversity")
    _require(len(histograms) >= 8 and len(structural_colors) == 10, "histogram/palette diversity")
    _require(controls == {(t, a) for t in range(4) for a in range(4)}, "control diversity")
    _require(canvas_styles == {"uniform", "rows", "cols", "other"}, "canvas diversity")
    repeated = generate_puzzle(0)
    other = generate_puzzle(0)
    _require(all(np.array_equal(p[r], q[r]) for s in repeated
                 for p, q in zip(repeated[s], other[s]) for r in p), "nondeterministic seed")
    # A wrong output and an input with no unique histogram must fail closed.
    bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in repeated[s]] for s in repeated}
    bad["test"][0]["output"][0, 0] = (bad["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    canvas, panels, _ = _decode_input(repeated["train"][0]["input"])
    ambiguous = [dict(p) for p in panels]
    ambiguous[1]["raw"] = ambiguous[0]["raw"].copy()
    try:
        _repair(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous histogram accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": pair_count, "puzzles": len(puzzle_hashes),
            "shape_pairs": len(shapes), "stencils": len(masks), "tile_patterns": len(structures),
            "histogram_multiplicities": len(histograms), "layouts": sorted(layouts),
            "tile_sizes": sorted(sizes), "control_combinations": len(controls)}
