"""Empty-window centers control a keyed, clipped microtile mosaic.

Only NumPy and the standard library are used. Generation and tests write no files.
The input-only solver reads complete framed fields; it has no puzzle/seed state.
"""
from __future__ import annotations

import numpy as np

# Clockwise order, independently interpreted in the two visible control fields.
_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_ARROWS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid must contain integer colors 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _centers(a, blank):
    """A full-resolution center mask, including every overlapping 3x3 window."""
    h, w = a.shape
    result = np.zeros((h, w), bool)
    if h >= 3 and w >= 3:
        valid = np.ones((h - 2, w - 2), bool)
        for dr in range(3):
            for dc in range(3):
                valid &= a[dr:h - 2 + dr, dc:w - 2 + dc] == blank
        result[1:-1, 1:-1] = valid
    return result


def _crop(mask):
    points = np.argwhere(mask)
    if not len(points):
        return np.zeros((0, 0), bool)
    lo, hi = points.min(0), points.max(0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _paint(canvas, tile, stencil, turn, anchor, *, exchange=True, rotate_tile=False):
    """Forward block copying, with exact clipping of a corner-aligned mosaic."""
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    if not stencil.size:
        return out, covered
    mask = np.rot90(stencil, -turn)
    colors = np.unique(tile)
    stamp = colors[0] + colors[1] - tile if exchange else tile.copy()
    if rotate_tile:
        stamp = np.rot90(stamp, -turn)
    n = len(tile)
    mh, mw = mask.shape[0] * n, mask.shape[1] * n
    ar, ac = _CORNERS[anchor]
    top, left = ar * (len(canvas) - mh), ac * (canvas.shape[1] - mw)
    for r, c in np.argwhere(mask):
        y, x = top + int(r) * n, left + int(c) * n
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(len(canvas), y + n), min(canvas.shape[1], x + n)
        if y0 < y1 and x0 < x1:
            out[y0:y1, x0:x1] = stamp[y0 - y:y1 - y, x0 - x:x1 - x]
            covered[y0:y1, x0:x1] = True
    return out, covered


def _bands(panel, divider):
    rows = np.flatnonzero(np.any(panel != divider, axis=1))
    if not len(rows):
        raise ValueError("empty panel")
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    fields = []
    for group in groups:
        r0, r1 = int(group[0]), int(group[-1]) + 1
        cols = np.flatnonzero(np.any(panel[r0:r1] != divider, axis=0))
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        field = panel[r0:r1, c0:c1]
        if (r0 == 0 or r1 == len(panel) or c0 == 0 or c1 == panel.shape[1]
                or np.any(field == divider)):
            raise ValueError("each field needs a complete divider surround")
        fields.append((field.copy(), (r0, c0)))
    if len(fields) != 3:
        raise ValueError("expected an upper tile and two lower keys")
    return fields


def _decode_panel(panel, divider):
    fields = _bands(panel, divider)
    tile = fields[0][0]
    if tile.shape[0] != tile.shape[1] or not 2 <= len(tile) <= 5 or len(np.unique(tile)) != 2:
        raise ValueError("microtile must be a complete binary square of side 2..5")
    keys = {}
    blanks = []
    for box, _ in fields[1:]:
        if box.shape != (3, 3):
            raise ValueError("keys must have visible 3x3 interiors")
        colors, counts = np.unique(box, return_counts=True)
        if len(colors) != 2 or int(counts.max()) not in (7, 8):
            raise ValueError("invalid key palette")
        blank = int(colors[np.argmax(counts)])
        blanks.append(blank)
        points = set(map(tuple, np.argwhere(box != blank)))
        found = None
        for k, (ar, ac) in enumerate(_CORNERS):
            if points == {(2 * ar, 2 * ac)}:
                found = ("anchor", k)
        for k, other in enumerate(_ARROWS):
            if points == {(1, 1), other}:
                found = ("turn", k)
        if found is None or found[0] in keys:
            raise ValueError("keys must encode one corner and one direction")
        keys[found[0]] = found[1]
    if len(set(blanks)) != 1 or set(keys) != {"turn", "anchor"}:
        raise ValueError("inconsistent keys")
    blank = blanks[0]
    if blank in tile or divider in tile:
        raise ValueError("tile roles must differ from blank and divider")
    return dict(tile=tile, blank=blank, divider=divider, **keys,
                field_origins=tuple(origin for _, origin in fields))


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    candidates = []
    for side in ("left", "right"):
        divider = int(a[0, 0 if side == "left" else -1])
        edge = a[0] if side == "left" else a[0, ::-1]
        stops = np.flatnonzero(edge != divider)
        if not len(stops):
            continue
        band_width = int(stops[0])
        p = band_width - 1  # One full-height separator follows the framed panel.
        if not 5 <= p <= 8 or w - band_width < 3:
            continue
        if side == "left":
            panel, separator, canvas = a[:, :p], a[:, p], a[:, p + 1:]
        else:
            panel, separator, canvas = a[:, w - p:], a[:, w - p - 1], a[:, :w - p - 1]
        if not np.all(separator == divider) or np.any(canvas == divider):
            continue
        try:
            ins = _decode_panel(panel, divider)
            palette = set(map(int, np.unique(canvas)))
            if len(palette) != 2 or ins["blank"] not in palette:
                continue
            texture = (palette - {ins["blank"]}).pop()
            if texture in ins["tile"]:
                continue
        except ValueError:
            continue
        ins.update(side=side, panel_width=p, texture=texture)
        candidates.append((canvas.copy(), ins))
    if len(candidates) != 1:
        raise ValueError("input needs one uniquely readable framed instruction panel")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    canvas, ins = _decode(input_grid)
    stencil = _crop(_centers(canvas, ins["blank"]))
    return _paint(canvas, ins["tile"], stencil, ins["turn"], ins["anchor"])[0]


def _scan_stencil(canvas, blank):
    """Direct nine-cell enumeration for validation, separate from _centers."""
    coords = [(r + 1, c + 1) for r in range(len(canvas) - 2)
              for c in range(canvas.shape[1] - 2)
              if all(int(canvas[r + dr, c + dc]) == blank
                     for dr in range(3) for dc in range(3))]
    if not coords:
        return np.zeros((0, 0), bool), coords
    r0, c0 = min(r for r, _ in coords), min(c for _, c in coords)
    rh = max(r for r, _ in coords) - r0 + 1
    cw = max(c for _, c in coords) - c0 + 1
    mask = np.zeros((rh, cw), bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, coords


def _corner_map(anchor, transform):
    r, c = _CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _inverse(canvas, ins, stencil, convention=(1, 0, 0), behavior="normal"):
    """Canvas-coordinate validation oracle; does not rotate a mask or copy blocks."""
    out = canvas.copy()
    covered = np.zeros(canvas.shape, bool)
    if not stencil.size:
        return out, covered
    sign, offset, transform = convention
    k = (sign * ins["turn"] + offset) % 4
    if behavior == "no_rotation":
        k = 0
    sh, sw = stencil.shape
    rh, rw = (sw, sh) if k % 2 else (sh, sw)
    n = len(ins["tile"])
    ar, ac = _corner_map(ins["anchor"], transform)
    if behavior == "top_left":
        ar, ac = 0, 0
    top, left = ar * (len(canvas) - rh * n), ac * (canvas.shape[1] - rw * n)
    rr, cc = np.indices(canvas.shape)
    y, x = rr - top, cc - left
    inside = (y >= 0) & (x >= 0) & (y < rh * n) & (x < rw * n)
    mr, mc = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
    if k == 0:
        sr, sc = mr, mc
    elif k == 1:
        sr, sc = sh - 1 - mc, mr
    elif k == 2:
        sr, sc = sh - 1 - mr, sw - 1 - mc
    else:
        sr, sc = mc, sw - 1 - mr
    covered = inside & (True if behavior == "fill_rectangle" else stencil[sr, sc])
    ty, tx = y % n, x % n
    if behavior == "rotate_tile":
        for _ in range(k):
            ty, tx = n - 1 - tx, ty
    original = ins["tile"][ty, tx]
    colors = np.unique(ins["tile"])
    values = original if behavior == "no_exchange" else colors[0] + colors[1] - original
    if behavior == "only_one_color":
        covered &= values == colors[0]
    out[covered] = values[covered]
    return out, covered


def _training_evidence(train):
    decoded = []
    for pair in train[:2]:
        canvas, ins = _decode(pair["input"])
        stencil, coords = _scan_stencil(canvas, ins["blank"])
        decoded.append((canvas, ins, stencil, coords, np.asarray(pair["output"])))
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = (sign, offset, transform)
                if all(np.array_equal(_inverse(c, i, s, convention)[0], out)
                       for c, i, s, _, out in decoded):
                    survivors.append(convention)
    if survivors != [(1, 0, 0)]:
        return False
    for behavior in ("no_rotation", "top_left", "rotate_tile", "no_exchange",
                     "fill_rectangle", "only_one_color"):
        if not any(not np.array_equal(_inverse(c, i, s, behavior=behavior)[0], out)
                   for c, i, s, _, out in decoded):
            return False
    # Ignore A, fill every blank macro cell, union window footprints, or select
    # just one window. These all must disagree somewhere in the demonstrations.
    for alternative in ("all_blanks", "footprint", "first_window"):
        distinguished = False
        for canvas, ins, _, coords, out in decoded:
            mask = np.zeros(canvas.shape, bool)
            if alternative == "all_blanks":
                mask = canvas == ins["blank"]
            elif alternative == "first_window":
                mask[coords[0]] = True
            else:
                for r, c in coords:
                    mask[r - 1:r + 2, c - 1:c + 2] = True
            if not np.array_equal(_inverse(canvas, ins, _crop(mask))[0], out):
                distinguished = True
        if not distinguished:
            return False
    return True


def _tile(rng, n, a, b):
    for _ in range(32):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if bits.any() and not bits.all() and not np.array_equal(bits, np.rot90(bits, 2)):
            return np.where(bits, a, b).astype(np.int8)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, a, b).astype(np.int8)


def _source_pattern(rng, h, w, profile):
    mask = np.zeros((h, w), bool)
    if profile in ("islands", "overlap"):
        patch = np.zeros((7, 7), bool)
        if profile == "islands":
            starts = [(0, 0), (int(rng.integers(2)), 4), (4, int(rng.integers(4)))]
            for r, c in starts:
                patch[r:r + 3, c:c + 3] = True
        else:
            ph, pw = ((3, 4), (4, 3), (4, 4), (3, 5), (5, 3))[int(rng.integers(5))]
            patch[:ph, :pw] = True
            patch[4:7, 4:7] = True
        patch = np.rot90(patch, int(rng.integers(4)))
        if rng.integers(2):
            patch = patch[:, ::-1]
        # Include source-border windows in training, without making them a split cue.
        r = int(rng.choice([0, h - 7, int(rng.integers(h - 6))]))
        c = int(rng.choice([0, w - 7, int(rng.integers(w - 6))]))
        mask[r:r + 7, c:c + 7] = patch
        return mask
    style = int(rng.integers(6))
    if style == 0:  # One window: the tight stencil is a single macro cell.
        r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
        mask[r:r + 3, c:c + 3] = True
    elif style == 1:  # A solid patch gives a dense set of overlapping centers.
        ph, pw = int(rng.integers(3, min(9, h) + 1)), int(rng.integers(3, min(9, w) + 1))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        mask[r:r + ph, c:c + pw] = True
    elif style == 2:  # Window unions can bend and branch; no supplied stencil.
        r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
        for _ in range(int(rng.integers(2, 8))):
            mask[r:r + 3, c:c + 3] = True
            dr, dc = _ARROWS[int(rng.integers(4))]
            r = int(np.clip(r + (dr - 1) * int(rng.integers(1, 4)), 0, h - 3))
            c = int(np.clip(c + (dc - 1) * int(rng.integers(1, 4)), 0, w - 3))
    else:
        # Compact and widely separated mixtures have different spacing and crops.
        ah, aw = (h, w) if style == 5 else (min(h, int(rng.integers(7, 13))), min(w, int(rng.integers(7, 13))))
        r0, c0 = int(rng.integers(h - ah + 1)), int(rng.integers(w - aw + 1))
        for _ in range(int(rng.integers(2, 6))):
            ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            r, c = r0 + int(rng.integers(ah - ph + 1)), c0 + int(rng.integers(aw - pw + 1))
            mask[r:r + ph, c:c + pw] = True
    return mask


def _field(rng, h, w, blank, texture, profile):
    desired = _source_pattern(rng, h, w, profile)
    # An explicit eight-of-nine near miss, separated from all source patches.
    options = [(r, c) for r in range(h - 2) for c in range(w - 2)
               if not desired[max(0, r - 1):min(h, r + 4), max(0, c - 1):min(w, c + 4)].any()]
    if not options:
        return None
    r, c = options[int(rng.integers(len(options)))]
    near = np.ones((3, 3), bool)
    near[int(rng.integers(3)), int(rng.integers(3))] = False
    desired[r:r + 3, c:c + 3] = near
    protected = desired.copy()
    empty = rng.random((h, w)) < rng.uniform(.23, .57)
    empty[protected] = True
    empty[r:r + 3, c:c + 3] = near
    intended = _centers(desired, True)
    # Monotone repair only removes extra background cells. Every deliberate
    # window and all eight blank near-miss cells stay intact; no trace is saved.
    extras = np.argwhere(_centers(empty, True) & ~intended)
    for rr, cc in extras:
        if not empty[rr - 1:rr + 2, cc - 1:cc + 2].all():
            continue
        choices = [(y, x) for y in range(rr - 1, rr + 2) for x in range(cc - 1, cc + 2)
                   if not protected[y, x]]
        if not choices:
            return None
        empty[choices[int(rng.integers(len(choices)))]] = False
    if not np.array_equal(_centers(empty, True), intended):
        return None
    return np.where(empty, blank, texture).astype(np.int8)


def _panel(rng, h, p, divider, blank, tile, turn, anchor, markers):
    n = len(tile)
    panel = np.full((h, p), divider, np.int8)
    # Random allocation of spare divider rows makes all support boundaries visible.
    spare = h - (n + 6 + 4)
    gaps = np.ones(4, int)
    gaps += rng.multinomial(spare, [0.25] * 4)
    tile_row = int(gaps[0])
    first_key_row = tile_row + n + int(gaps[1])
    second_key_row = first_key_row + 3 + int(gaps[2])
    tc = int(rng.integers(1, p - n))
    panel[tile_row:tile_row + n, tc:tc + n] = tile
    order = ["turn", "anchor"] if rng.integers(2) else ["anchor", "turn"]
    for kind, row, marker in zip(order, (first_key_row, second_key_row), markers):
        col = int(rng.integers(1, p - 3))
        box = np.full((3, 3), blank, np.int8)
        if kind == "turn":
            box[1, 1] = marker
            box[_ARROWS[turn]] = marker
        else:
            ar, ac = _CORNERS[anchor]
            box[2 * ar, 2 * ac] = marker
        panel[row:row + 3, col:col + 3] = box
    return panel


def _make_pair(rng, roles, n, control, profile):
    blank, divider, a, b, textures = roles
    turn, anchor = control
    for _ in range(64):
        p = int(rng.integers(max(n, 3) + 2, min(8, max(n, 3) + 3) + 1))
        h = int(rng.integers(max(18, n + 11), 31))
        max_w = 29 - p
        if profile == "islands":
            w = int(rng.integers(5 * n + 2, max_w + 1))
        elif profile == "overlap":
            w = 5 * n - int(rng.integers(1, n))
        else:
            w = int(rng.integers(9, max_w + 1))
        texture = int(rng.choice(textures))
        canvas = _field(rng, h, w, blank, texture, profile)
        if canvas is None:
            continue
        tile = _tile(rng, n, a, b)
        stencil = _crop(_centers(canvas, blank))
        if not stencil.size:
            continue
        output, covered = _paint(canvas, tile, stencil, turn, anchor)
        if int(covered.sum()) < n * n or np.all(covered):
            continue
        # Overlap witness must include at least one genuinely partial tile block.
        if profile == "overlap" and int(covered.sum()) % (n * n) == 0:
            continue
        markers = rng.choice([a, b, texture], 2, replace=True)
        panel = _panel(rng, h, p, divider, blank, tile, turn, anchor, markers)
        separator = np.full((h, 1), divider, np.int8)
        side = "left" if rng.integers(2) else "right"
        inp = np.concatenate((panel, separator, canvas) if side == "left" else (canvas, separator, panel), axis=1)
        try:
            recovered, visible = _decode(inp)
        except ValueError:
            continue
        if (not np.array_equal(recovered, canvas) or not np.array_equal(visible["tile"], tile)
                or (visible["turn"], visible["anchor"]) != control):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded pair construction failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    roles = (*colors[:4], colors[4:])
    q, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    adjacent = (anchor + int(rng.choice([1, 3]))) % 4
    controls = [(q, anchor), (even, adjacent), (q, adjacent), (even, anchor)]
    sizes = [2, 3] if rng.integers(2) else [3, 2]
    # The first two demonstrate detection, overlap, scale, clipping and both keys
    # even for the minimum legal request. Reject only ambiguous demonstrations.
    for _ in range(16):
        first = _make_pair(rng, roles, sizes[0], controls[0], "islands")
        second = _make_pair(rng, roles, sizes[1], controls[1], "overlap")
        if _training_evidence([first, second]):
            break
    else:
        raise RuntimeError("could not construct informative demonstrations")
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", nt - 2), ("test", nq)):
        for index in range(count):
            if split == "train":
                n = int(rng.integers(2, 6))
                control = controls[index + 2] if index < 2 else (int(rng.integers(4)), int(rng.integers(4)))
            else:
                # Hold out a control combination, not a direction meaning.
                used = {(_decode(p["input"])[1]["turn"], _decode(p["input"])[1]["anchor"])
                        for p in puzzle["train"]}
                available = [(k, a) for k in range(4) for a in range(4) if (k, a) not in used]
                control = available[int(rng.integers(len(available)))]
                n = int(rng.choice([len(_decode(p["input"])[1]["tile"]) for p in puzzle["train"]]))
            for _ in range(8):
                pair = _make_pair(rng, roles, n, control, "free")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    """Bounded pixel-derived diagnostics; no sampled metadata or persisted grids."""
    canvas, ins = _decode(pair["input"])
    s, coords = _scan_stencil(canvas, ins["blank"])
    _, covered = _inverse(canvas, ins, s)
    n, k = len(ins["tile"]), ins["turn"]
    rh, rw = (s.shape[1], s.shape[0]) if k % 2 else s.shape
    mh, mw = rh * n, rw * n
    return dict(canvas_shape=tuple(map(int, canvas.shape)), side=ins["side"],
                panel_width=ins["panel_width"], tile_size=n, control=(k, ins["anchor"]),
                windows=len(coords), stencil_shape=tuple(map(int, s.shape)),
                stencil_cells=int(s.size), blank_fraction=round(float(np.mean(canvas == ins["blank"])), 3),
                overlapping=any(max(abs(r - y), abs(c - x)) < 3
                                for j, (r, c) in enumerate(coords) for y, x in coords[j + 1:]),
                edge_window=any(r in (1, len(canvas) - 2) or c in (1, canvas.shape[1] - 2) for r, c in coords),
                clip_axes=(int(mh > len(canvas)), int(mw > canvas.shape[1])),
                painted_cells=int(covered.sum()), partial_cells=int(covered.sum()) % (n * n),
                role_palette=(ins["blank"], ins["divider"], *map(int, np.unique(ins["tile"]))),
                field_origins=ins["field_origins"])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, palettes, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            canvas, ins = _decode(inp)
            s, coords = _scan_stencil(canvas, ins["blank"])
            assert len(coords) and out.shape == canvas.shape
            assert np.array_equal(s, _crop(_centers(canvas, ins["blank"])))
            # All-background windows anywhere in the entire input are precisely
            # the canvas windows: the panel has none and the separator stops crossings.
            assert int(_centers(inp, ins["blank"]).sum()) == len(coords)
            expected, covered = _inverse(canvas, ins, s)
            assert np.array_equal(out, expected)
            assert np.array_equal(inp, np.asarray(pair["input"]))
            assert covered.any() and not covered.all()
            assert np.array_equal(out != canvas, covered)  # Paint colors are disjoint.
            assert np.array_equal(out[~covered], canvas[~covered])
            assert set(map(int, np.unique(out[covered]))) <= set(map(int, np.unique(ins["tile"])))
            assert not np.array_equal(out, canvas), "omitting A must change the answer"
            palettes.add((ins["blank"], ins["divider"], *map(int, np.unique(ins["tile"]))))
            features[split].append(pair_features(pair))
    assert len(palettes) == 1, "core role conventions must agree within a puzzle"
    assert _training_evidence(puzzle["train"])
    train = features["train"]
    assert train[0]["tile_size"] != train[1]["tile_size"]
    assert not train[0]["overlapping"] and train[1]["overlapping"]
    assert train[0]["clip_axes"] == (0, 0)
    assert train[1]["clip_axes"] == (0, 1) and train[1]["partial_cells"] != 0
    controls = {f["control"] for f in train}
    sizes = {f["tile_size"] for f in train}
    assert all(f["control"] not in controls and f["tile_size"] in sizes for f in features["test"])


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """42 seeds, all legal counts, exact invariants and counterfactuals; no I/O."""
    shapes, masks, palettes, tiles, windows, clips, controls, sides = (set() for _ in range(8))
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                f = pair_features(pair)
                shapes.add(f["canvas_shape"])
                palettes.add(f["role_palette"])
                tiles.add(f["tile_size"])
                windows.add(f["windows"])
                clips.add(f["clip_axes"])
                controls.add(f["control"])
                sides.add(f["side"])
                canvas, ins = _decode(before)
                masks.add(_key(_crop(_centers(canvas, ins["blank"]))))
    assert len(fingerprints) == 42
    assert len(shapes) >= 40 and len(masks) >= 75 and len(palettes) >= 25
    assert tiles == {2, 3, 4, 5} and len(windows) >= 15
    assert clips == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert controls == {(k, a) for k in range(4) for a in range(4)}
    assert sides == {"left", "right"}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1234 + 10 * nt + nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    assert _puzzle_key(generate_puzzle(987, 5, 2)) == _puzzle_key(generate_puzzle(987, 5, 2))
    for nt, nq in ((1, 1), (11, 1), (2, 0), (8, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Hand-sized arithmetic witnesses independently pin clipping phase and overlap.
    c = np.full((6, 8), 8, np.int8)
    c[:3, :4] = 0  # Two adjacent centers, not one component or twelve macro cells.
    s, coords = _scan_stencil(c, 0)
    assert coords == [(1, 1), (1, 2)] and s.shape == (1, 2) and s.all()
    t = np.array([[2, 3], [3, 3]], np.int8)
    small = np.full((3, 3), 8, np.int8)
    out, _ = _paint(small, t, s, 0, 1)
    assert out.tolist() == [[2, 3, 2], [2, 2, 2], [8, 8, 8]]
    # Full input-only clipping witness with a label written directly as digit rows.
    canvas = np.full((18, 4), 8, np.int8)
    canvas[:3] = 0
    tile = np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]], np.int8)
    panel = _panel(np.random.default_rng(90), 18, 5, 6, 0, tile, 0, 1, [2, 3])
    inp = np.concatenate((panel, np.full((18, 1), 6, np.int8), canvas), axis=1)
    expected = canvas.copy()
    expected[:3] = [[2, 3, 2, 2], [2, 2, 3, 2], [3, 2, 2, 3]]
    assert np.array_equal(solve(inp), expected)
    # Geometry, rather than color numbers or the physical side of the panel,
    # determines the instruction. Exercise this away from sampled dimensions.
    moved = np.concatenate((canvas, np.full((18, 1), 6, np.int8), panel), axis=1)
    assert np.array_equal(solve(moved), expected)
    permutation = np.array([7, 0, 9, 1, 8, 3, 5, 2, 6, 4], np.int8)
    assert np.array_equal(solve(permutation[moved]), permutation[expected])
    # A legal no-window input is also handled without inventing an occupied cell.
    p = generate_puzzle(0, 2, 1)
    canvas, ins = _decode(p["train"][0]["input"])
    altered = np.asarray(p["train"][0]["input"]).copy()
    source_slice = np.s_[:, ins["panel_width"] + 1:] if ins["side"] == "left" else np.s_[:, :-ins["panel_width"] - 1]
    yy, xx = np.indices(canvas.shape)
    no_windows = np.where((yy + xx) % 2, ins["blank"], ins["texture"]).astype(np.int8)
    altered[source_slice] = no_windows
    assert np.array_equal(solve(altered), no_windows)
    damaged = {split: [{r: np.asarray(a).copy() for r, a in pair.items()} for pair in p[split]] for split in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
