"""Visible-panel mosaics: exchange a complete tile, rotate its stencil, anchor.

The source's large tile has five explicit rows. There is no completion rule.
Colors and positions are local geometry; neither the split nor a hidden palette
or layout convention is needed to decode an instruction.
"""
import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_DIRECTIONS = ((0, 1), (1, 2), (2, 1), (1, 0))


def _integer_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    a = _integer_argument("num_train", num_train, 2, 10)
    b = _integer_argument("num_test", num_test, 1, 4)
    if a is None and b is None:
        b = 1 + seed % 4
        a = max(2 + seed % 9, b + 1)
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    elif b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError("num_train must be greater than num_test")
    return a, b


def _grid_key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _crop(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


def _asymmetric(a):
    return not np.array_equal(a, np.rot90(a, 2))


def _tile(rng, n, colors, witness):
    for _ in range(24):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if bits.any() and not bits.all() and (not witness or _asymmetric(bits)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _stencil(rng, h, w, intensity, witness):
    style = int(rng.integers(0, 7))
    rows, cols = np.indices((h, w))
    if style <= 1:
        mask = rng.random((h, w)) < rng.uniform(.16, .88)
    elif style == 2:  # A wandering branch with optional cross branches.
        mask = np.zeros((h, w), bool)
        c = int(rng.integers(w))
        for r in range(h):
            nc = int(np.clip(c + rng.integers(-2, 3), 0, w - 1))
            mask[r, min(c, nc):max(c, nc) + 1] = True
            if rng.random() < .4:
                mask[r, int(rng.integers(w)):c + 1] = True
            c = nc
    elif style == 3:  # Contour, possibly interrupted.
        mask = (rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)
        if rng.random() < .55:
            mask &= rng.random((h, w)) > .13
    elif style == 4:  # Separated bars and islands are legitimate stencils.
        mask = (rows % int(rng.integers(2, 4)) == 0) & (cols % 2 == 0)
        mask |= rng.random((h, w)) < .1 * intensity
    elif style == 5:
        mask = (rows == int(rng.integers(h))) | (cols == int(rng.integers(w)))
    else:
        mask = np.ones((h, w), bool)
        if witness or rng.random() < .55:
            mask &= rng.random((h, w)) > .25
    if not mask.any():
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
    mask = _crop(mask)
    if witness and (mask.all() or not _asymmetric(mask) or min(mask.shape) < 2):
        # Preserve the requested scale, using random extrema rather than corners.
        mask = rng.random((max(2, h), max(2, w))) < .55
        mask[0, 0] = False
        mask[0, -1] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
        mask[1, 0] = True
    return mask


def _canvas(rng, h, w, colors, intensity):
    rows, cols = np.indices((h, w))
    style = int(rng.choice(8, p=([.23, .21, .21, .07, .05, .07, .08, .08]
                               if intensity == 0 else [.12, .14, .14, .12, .1, .12, .13, .13])))
    count = len(colors)
    offset = int(rng.integers(count))
    if style == 0:
        ix = np.full((h, w), offset)
    elif style in (1, 2):
        coord = rows if style == 1 else cols
        extent = h if style == 1 else w
        cuts = np.sort(rng.choice(np.arange(1, extent), size=int(rng.integers(1, min(6, extent))), replace=False))
        ix = np.searchsorted(cuts, coord, side="right") + offset
    elif style == 3:
        ix = rows // int(rng.integers(2, 9)) + cols // int(rng.integers(2, 8)) + offset
    elif style == 4:
        ix = (rows + int(rng.integers(1, 4)) * cols) // int(rng.integers(2, 8)) + offset
    elif style == 5:
        edge = np.minimum.reduce([rows, h - 1 - rows, cols, w - 1 - cols])
        ix = edge // int(rng.integers(1, 5)) + offset
    elif style == 6:
        ix = np.full((h, w), offset)
        for _ in range(int(rng.integers(2, 8))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(2, h + 1)), int(rng.integers(2, w + 1))
            ix[r:r + rh, c:c + cw] = int(rng.integers(count))
    else:
        centers = rng.random((int(rng.integers(2, 7)), 2)) * [h, w]
        distance = [(rows - r) ** 2 + rng.uniform(.4, 2) * (cols - c) ** 2 for r, c in centers]
        ix = np.argmin(distance, axis=0) + offset
    return np.asarray(colors, dtype=int)[ix % count]


def _panel(rng, h, p, ground, divider, tile, mask, stencil_color, markers, turn, anchor, framed):
    n = len(tile)
    key_height = int(rng.integers(8, min(13, h - n - mask.shape[0] - (2 if framed else 0) - 2) + 1))
    key_top = h - key_height
    tile_height = n + 2 * framed
    spare = key_top - tile_height - mask.shape[0] - 1
    assert spare >= 1
    tile_row = int(rng.integers(1, spare + 1))
    gap = int(rng.integers(1, spare - tile_row + 2))
    stencil_row = tile_row + tile_height + gap
    a = np.full((h, p), ground, int)
    tile_col = int(rng.integers(0, p - tile_height + 1))
    if framed:
        a[tile_row:tile_row + tile_height, tile_col:tile_col + tile_height] = divider
        a[tile_row + 1:tile_row + n + 1, tile_col + 1:tile_col + n + 1] = tile
    else:
        a[tile_row:tile_row + n, tile_col:tile_col + n] = tile
    stencil_col = int(rng.integers(0, p - mask.shape[1] + 1))
    a[stencil_row:stencil_row + mask.shape[0], stencil_col:stencil_col + mask.shape[1]] = np.where(mask, stencil_color, ground)
    a[key_top:] = divider
    # Shared divider row is allowed; the grid edge can close the final field.
    first = key_top + 1 + int(rng.integers(0, key_height - 7))
    second = int(rng.integers(first + 4, h - 2))
    positions = [(first, int(rng.integers(p - 2))), (second, int(rng.integers(p - 2)))]
    if rng.random() < .25:
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


def _paint(canvas, instructions):
    """Forward construction: rotate the mask, then copy complete tile blocks."""
    out, footprint = canvas.copy(), np.zeros(canvas.shape, bool)
    for ins in instructions:
        tile, stencil, k, anchor = ins["tile"], ins["stencil"], ins["turn"], ins["anchor"]
        colors = np.unique(tile)
        swapped = np.where(tile == colors[0], colors[1], colors[0])
        mask = np.rot90(stencil, -k)
        n = len(tile)
        mh, mw = np.array(mask.shape) * n
        assert mh <= len(canvas) and mw <= canvas.shape[1]
        ar, ac = _CORNERS[anchor]
        top, left = ar * (len(canvas) - mh), ac * (canvas.shape[1] - mw)
        for r, c in np.argwhere(mask):
            block = np.s_[top + r * n:top + (r + 1) * n, left + c * n:left + (c + 1) * n]
            assert not footprint[block].any()
            out[block] = swapped
            footprint[block] = True
    return out, footprint


def _decode_panel(panel, divider):
    """Find bounded fields and separated objects from pixels, including source panels."""
    h, p = panel.shape
    ground = int(panel[0, 0])
    assert ground != divider and np.all(panel[0] == ground)
    fields = []
    for r in range(1, h - 2):
        for c in range(p - 2):
            box = panel[r:r + 3, c:c + 3]
            if np.any(box == divider):
                continue
            if not np.all(panel[r - 1, max(0, c - 1):min(p, c + 4)] == divider):
                continue
            if r + 3 < h and not np.all(panel[r + 3, max(0, c - 1):min(p, c + 4)] == divider):
                continue
            if c and not np.all(panel[r:r + 3, c - 1] == divider):
                continue
            if c + 3 < p and not np.all(panel[r:r + 3, c + 3] == divider):
                continue
            points = set(map(tuple, np.argwhere(box != ground)))
            vals = np.unique(box[box != ground])
            if len(vals) != 1:
                continue
            for a, (ar, ac) in enumerate(_CORNERS):
                if points == {(2 * ar, 2 * ac)}:
                    fields.append((r, c, "anchor", a, int(vals[0])))
            for k, neighbor in enumerate(_DIRECTIONS):
                if points == {(1, 1), neighbor}:
                    fields.append((r, c, "turn", k, int(vals[0])))
    # Top objects cannot substitute for the two lowest control fields.
    fields = sorted(fields)[-2:]
    assert len(fields) == 2 and {f[2] for f in fields} == {"turn", "anchor"}
    key_top = min(f[0] for f in fields) - 1
    while key_top > 0 and np.all(panel[key_top - 1] == divider):
        key_top -= 1
    expected = np.full(panel[key_top:].shape, divider, int)
    for r, c, _, _, _ in fields:
        expected[r - key_top:r - key_top + 3, c:c + 3] = panel[r:r + 3, c:c + 3]
    assert np.array_equal(expected, panel[key_top:])
    upper = panel[:key_top]
    used_rows = np.flatnonzero(np.any(upper != ground, axis=1))
    assert len(used_rows)
    tr = int(used_rows[0])
    end = tr
    while end < key_top and np.any(upper[end] != ground):
        end += 1
    band = upper[tr:end]
    cs = np.flatnonzero(np.any(band != ground, axis=0))
    tc = int(cs[0])
    visible_tile = band[:, tc:int(cs[-1]) + 1]
    tile = visible_tile.copy()
    framed = (min(tile.shape) >= 4 and np.all(tile[0] == divider)
              and np.all(tile[-1] == divider) and np.all(tile[:, 0] == divider)
              and np.all(tile[:, -1] == divider))
    if framed:
        tile = tile[1:-1, 1:-1].copy()
    assert tile.shape[0] == tile.shape[1] and 2 <= len(tile) <= 5
    assert len(np.unique(tile)) == 2
    assert framed or ground not in tile
    rest = upper[end:]
    coords = np.argwhere(rest != ground)
    assert len(coords)
    lo, hi = coords.min(0), coords.max(0) + 1
    stencil_area = rest[lo[0]:hi[0], lo[1]:hi[1]]
    values = np.unique(stencil_area[stencil_area != ground])
    assert len(values) == 1
    mask = stencil_area != ground
    # Rebuild all occupied upper pixels to reject stray instructions/noise.
    allowed = np.full(upper.shape, ground, int)
    allowed[tr:end, tc:tc + visible_tile.shape[1]] = visible_tile
    sr, sc = end + int(lo[0]), int(lo[1])
    allowed[sr:sr + mask.shape[0], sc:sc + mask.shape[1]] = stencil_area
    assert np.array_equal(allowed, upper)
    info = {f[2]: f[3] for f in fields}
    info.update(tile=tile, stencil=mask, ground=ground, divider=int(divider),
                stencil_color=int(values[0]), markers=tuple(f[4] for f in fields),
                framed=bool(framed), panel_width=p, tile_origin=(tr, tc),
                stencil_origin=(sr, sc), key_origins=tuple((f[0], f[1]) for f in fields),
                gap=sr - end)
    return info


def _decode_input(grid):
    h, w = grid.shape
    candidates = {"left": [], "right": []}
    for side in candidates:
        for p in range(4, min(11, w - 3)):
            col = p if side == "left" else w - p - 1
            d = int(grid[0, col])
            if not np.all(grid[:, col] == d):
                continue
            panel = grid[:, :p] if side == "left" else grid[:, w - p:]
            try:
                info = _decode_panel(panel, d)
            except (AssertionError, IndexError, ValueError):
                continue
            candidates[side].append((p, info))
    assert all(len(v) <= 1 for v in candidates.values()), "ambiguous separator"
    assert any(candidates.values()), "no visible instruction panels"
    left = candidates["left"][0][0] + 1 if candidates["left"] else 0
    right = w - candidates["right"][0][0] - 1 if candidates["right"] else w
    assert right > left
    panels = [v[0][1] for v in candidates.values() if v]
    return grid[:, left:right].copy(), panels, ("flank" if len(panels) == 2 else next(s for s, v in candidates.items() if v))


def _anchor_transform(a, transform):
    r, c = _CORNERS[a]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _predict_decoded(canvas, panels, convention=(1, 0, 0), behavior="normal"):
    """Independent inverse-coordinate reconstruction of each output cell.

    No rotated-mask or forward-painting helper is shared with construction.
    Variants are reasonable competing rules used to check training evidence.
    """
    rr, cc = np.indices(canvas.shape)
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    sign, offset, transform = convention
    for ins in panels:
        t, s = ins["tile"], ins["stencil"]
        k = (sign * ins["turn"] + offset) % 4
        n, sh, sw = len(t), *s.shape
        rh, rw = (sw, sh) if k % 2 else (sh, sw)
        assert n * rh <= len(canvas) and n * rw <= canvas.shape[1]
        ar, ac = _anchor_transform(ins["anchor"], transform)
        y, x = rr - ar * (len(canvas) - n * rh), cc - ac * (canvas.shape[1] - n * rw)
        inside = (y >= 0) & (x >= 0) & (y < n * rh) & (x < n * rw)
        mr, mc = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
        if k == 0:
            sy, sx = mr, mc
        elif k == 1:
            sy, sx = sh - 1 - mc, mr
        elif k == 2:
            sy, sx = sh - 1 - mr, sw - 1 - mc
        else:
            sy, sx = mc, sw - 1 - mr
        active = inside & (np.ones(canvas.shape, bool) if behavior == "fill_holes" else s[sy, sx])
        ty, tx = y % n, x % n
        if behavior == "rotate_tile":
            for _ in range(k):
                ty, tx = n - 1 - tx, ty
        original = t[ty, tx]
        colors = np.unique(t)
        values = original if behavior == "no_exchange" else colors[0] + colors[1] - original
        if behavior == "only_first":
            active &= values == colors[0]
        if behavior == "only_second":
            active &= values == colors[1]
        if behavior == "normal" and convention == (1, 0, 0):
            assert not np.any(covered & active), "conflicting panel footprints"
        out[active] = values[active]
        covered |= active
    return out, covered


def _learn_convention(train):
    decoded = [(*_decode_input(p["input"])[:2], p["output"]) for p in train]
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = (sign, offset, transform)
                try:
                    if all(np.array_equal(_predict_decoded(c, ps, convention)[0], out) for c, ps, out in decoded):
                        survivors.append(convention)
                except AssertionError:
                    continue
    assert survivors == [(1, 0, 0)], "training leaves geometric conventions ambiguous"
    for behavior in ("rotate_tile", "no_exchange", "fill_holes", "only_first", "only_second"):
        assert any(not np.array_equal(_predict_decoded(c, ps, behavior=behavior)[0], out)
                   for c, ps, out in decoded), f"training does not distinguish {behavior}"
    return survivors[0]


def solve(train_pairs, test_input):
    """Predict using only training demonstrations and a new input."""
    convention = _learn_convention(train_pairs)
    canvas, panels, _ = _decode_input(test_input)
    return _predict_decoded(canvas, panels, convention)[0]


def _make_pair(rng, params, regime, layout, controls, witness):
    count = len(controls)
    for attempt in range(81):
        fallback = attempt == 80
        intensity = params["intensity"]
        if fallback:
            h, widths, sizes = 30, [5] * count, [2] * count
        else:
            hrange = (20, 26) if regime == "source_train" and params["mode"] == "source_shift" else (18, 31)
            if regime == "target_test" and params["mode"] == "source_shift":
                hrange = (26, 31)
            h = int(rng.integers(*hrange))
            widths = [int(rng.integers(4, 9 if count == 2 else 11)) for _ in controls]
            sizes = [int(rng.integers(2, min(5, p) + 1)) for p in widths]
        max_w = 30 - sum(widths) - count
        if max_w < max(8, max(sizes) * 2):
            continue
        min_w = max(8, 2 * max(sizes))
        if not fallback and rng.random() < .4:
            min_w = max(min_w, max_w - 3)
        w = max_w if fallback else int(rng.integers(min_w, max_w + 1))
        palette = np.asarray(rng.choice(10, size=int(rng.integers(4, params["richness"] + 1)), replace=False), int)
        ground, divider = map(int, palette[:2])
        ink = palette[2:]
        canvas_colors = palette[rng.choice(len(palette), size=int(rng.integers(1, min(5, len(palette)) + 1)), replace=False)]
        canvas = (np.full((h, w), divider, int) if fallback else
                  _canvas(rng, h, w, canvas_colors, intensity))
        panels, instructions = [], []
        try:
            for p, n, (k, anchor) in zip(widths, sizes, controls):
                framed = p >= n + 2 and rng.random() < .32
                choices = np.concatenate(([ground], ink)) if framed else ink
                tile_colors = rng.choice(choices, 2, replace=False)
                tile = _tile(rng, n, tile_colors, witness)
                # Witnesses leave a visible margin along both axes, so changing
                # either component of the anchor actually changes placement.
                fit_h, fit_w = h - int(witness), w - int(witness)
                max_sh = min(h - n - 2 * framed - 10, (fit_w if k % 2 else fit_h) // n)
                max_sw = min(p, (fit_h if k % 2 else fit_w) // n)
                assert max_sh >= 2 and max_sw >= 2
                if fallback:
                    mask = np.array([[1, 0], [1, 1], [1, 0]], bool)
                else:
                    lower = 2 if witness else 1
                    large = rng.random() < (.2 if intensity == 0 else .45)
                    low_sh = max(lower, (2 * max_sh + 2) // 3) if large else lower
                    low_sw = max(lower, (2 * max_sw + 2) // 3) if large else lower
                    sh = int(rng.integers(low_sh, max_sh + 1))
                    sw = int(rng.integers(low_sw, max_sw + 1))
                    mask = _stencil(rng, sh, sw, intensity, witness)
                if witness:
                    assert not mask.all() and _asymmetric(mask)
                stencil_color = int(rng.choice(ink))
                markers = rng.choice(ink, 2, replace=True)
                instruction = dict(tile=tile, stencil=mask, turn=k, anchor=anchor)
                instructions.append(instruction)
                panels.append(_panel(rng, h, p, ground, divider, tile, mask,
                                     stencil_color, markers, k, anchor, framed))
            output, covered = _paint(canvas, instructions)
            assert covered.any() and np.any(output != canvas)
            d = np.full((h, 1), divider, int)
            parts = ([panels[0], d, canvas, d, panels[1]] if count == 2 else
                     ([panels[0], d, canvas] if layout == "left" else [canvas, d, panels[0]]))
            inp = np.concatenate(parts, axis=1)
            recovered, visible, _ = _decode_input(inp)
            assert np.array_equal(recovered, canvas)
            assert np.array_equal(_predict_decoded(recovered, visible)[0], output)
            if witness:
                for behavior in ("rotate_tile", "no_exchange", "fill_holes", "only_first", "only_second"):
                    if behavior == "rotate_tile" and all(k % 2 == 0 for k, _ in controls):
                        continue
                    assert not np.array_equal(_predict_decoded(canvas, visible, behavior=behavior)[0], output)
            return {"input": inp, "output": output}
        except (AssertionError, ValueError):
            if fallback:
                raise AssertionError("conservative pair construction failed")
    raise AssertionError("unreachable construction state")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed % (1 << 64))
    nt, nq = _resolve_counts(seed, rng, num_train, num_test)
    params = dict(intensity=int(rng.integers(3)), richness=int(rng.integers(5, 11)),
                  mode=str(rng.choice(["mixed", "mixed", "matched", "source_shift", "held_out"])),
                  preferred=str(rng.choice(["left", "right", "flank"])))
    q, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    a = int(rng.integers(4))
    b = (a + int(rng.choice([1, 3]))) % 4
    first_multi = bool(rng.integers(2))
    held = (int(rng.integers(4)), int(rng.integers(4)))
    witness_controls = {(q, a), (q, b), (even, a)}
    if held in witness_controls:
        held = ((even + 2) % 4, b)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    for split, count in (("train", nt), ("test", nq)):
        regime = "source_train" if split == "train" else "target_test"
        for i in range(count):
            witness = split == "train" and i < 2
            if split == "train" and params["mode"] == "source_shift":
                layout = str(rng.choice(["left", "left", "right"]))
                controls = [[(1, 1)], [(0, 0)], [(2, 2)]][min(i, 2)]
            elif witness:
                multi = (i == 0) == first_multi
                layout = "flank" if multi else str(rng.choice(["left", "right"]))
                controls = [(q, a), (q, b)] if multi else [(even, a)]
            else:
                layout = (params["preferred"] if params["mode"] == "matched" and rng.random() < .75
                          else str(rng.choice(["left", "right", "flank"])))
                if split == "test" and nq >= 2 and i < 2:
                    layout = "flank" if i == 0 else str(rng.choice(["left", "right"]))
                anchors = list(map(int, rng.choice(4, 2 if layout == "flank" else 1, replace=False)))
                controls = [(int(rng.integers(4)), anchor) for anchor in anchors]
                if params["mode"] == "held_out":
                    if split == "test" and i == 0:
                        controls[0] = held
                        if len(controls) == 2 and controls[1][1] == held[1]:
                            controls[1] = (controls[1][0], (held[1] + 1) % 4)
                    elif split == "train":
                        controls = [((k + 1) % 4, an) if (k, an) == held else (k, an) for k, an in controls]
            for _ in range(8):
                pair = _make_pair(rng, params, regime, layout, controls, witness)
                _, visible, _ = _decode_input(pair["input"])
                roles = (visible[0]["ground"], visible[0]["divider"])
                if previous_roles is not None and any(x == y for x, y in zip(roles, previous_roles)):
                    shifts = [v for v in range(1, 10)
                              if all((x + v) % 10 != y for x, y in zip(roles, previous_roles))]
                    shift = int(rng.choice(shifts))
                    pair = {r: (grid + shift) % 10 for r, grid in pair.items()}
                    roles = tuple((v + shift) % 10 for v in roles)
                if _grid_key(pair["input"]) not in seen and not _is_source_pair(pair):
                    break
            else:
                raise AssertionError("duplicate construction")
            seen.add(_grid_key(pair["input"]))
            previous_roles = roles
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_pairs():
    return [{role: np.array([[int(v) for v in row] for row in text.split()], int)
             for role, text in zip(("input", "output"), case)} for case in _SOURCE_CASES]


def _is_source_pair(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"])) in _SOURCE_KEYS


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[s], list) for s in puzzle)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
        seen = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= d <= 30 for d in grid.shape)
                    assert np.all((0 <= grid) & (grid <= 9))
                key = _grid_key(pair["input"])
                assert key not in seen and not _is_source_pair(pair)
                seen.add(key)
                canvas, panels, _ = _decode_input(pair["input"])
                expected, covered = _predict_decoded(canvas, panels)
                assert pair["output"].shape == canvas.shape
                assert np.array_equal(pair["output"], expected)
                assert covered.any() and np.any(expected != canvas)
        # Conventions are inferred exclusively from training, before querying tests.
        convention = _learn_convention(puzzle["train"])
        for pair in puzzle["test"]:
            canvas, panels, _ = _decode_input(pair["input"].copy())
            predicted = _predict_decoded(canvas, panels, convention)[0]
            assert np.array_equal(predicted, pair["output"])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed generated puzzle: {exc}") from exc


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    count = 0
    while todo:
        count += 1
        stack = [todo.pop()]
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in todo:
                    todo.remove(point)
                    stack.append(point)
    return count


def pair_features(pair):
    """Review measurements reconstructed from returned pixels, not sampled state."""
    canvas, panels, layout = _decode_input(pair["input"])
    _, covered = _predict_decoded(canvas, panels)
    rectangles = []
    for ins in panels:
        full = dict(ins, stencil=np.ones(ins["stencil"].shape, bool))
        rectangles.append(_predict_decoded(canvas, [full])[1])
    return dict(layout=layout, canvas_shape=canvas.shape, panel_count=len(panels),
                palette_size=len(np.unique(pair["input"])), canvas_colors=len(np.unique(canvas)),
                canvas_geometry=("uniform" if len(np.unique(canvas)) == 1 else
                                 "horizontal_bands" if np.all(canvas == canvas[:, :1]) else
                                 "vertical_bands" if np.all(canvas == canvas[:1, :]) else "multidirectional"),
                painted_cells=int(covered.sum()), painted_fraction=float(covered.mean()),
                rectangle_overlap=bool(len(panels) == 2 and np.any(rectangles[0] & rectangles[1])),
                adjacent=bool(len(panels) == 2 and (panels[0]["anchor"] - panels[1]["anchor"]) % 2 == 1),
                controls=[(p["turn"], p["anchor"]) for p in panels],
                tile_sizes=[len(p["tile"]) for p in panels],
                stencil_shapes=[p["stencil"].shape for p in panels],
                stencil_components=[_components(p["stencil"]) for p in panels],
                stencil_holes=[_components(np.pad(~p["stencil"], 1, constant_values=True)) - 1 for p in panels],
                stencil_densities=[float(p["stencil"].mean()) for p in panels],
                opposite_extrema=[bool(p["stencil"][0, 0] and p["stencil"][-1, -1]) for p in panels],
                framed=[p["framed"] for p in panels],
                ground_tiles=[bool(p["ground"] in p["tile"]) for p in panels],
                role_sharing=[bool(set(np.unique(p["tile"])) & set(p["markers"])) for p in panels],
                canvas_tile_sharing=[bool(set(np.unique(p["tile"])) & set(np.unique(canvas))) for p in panels],
                grounds=[p["ground"] for p in panels], dividers=[p["divider"] for p in panels],
                marker_colors=[p["markers"] for p in panels],
                panel_widths=[p["panel_width"] for p in panels],
                tile_origins=[p["tile_origin"] for p in panels],
                stencil_origins=[p["stencil_origin"] for p in panels],
                key_origins=[p["key_origins"] for p in panels], gaps=[p["gap"] for p in panels])


def self_test() -> None:
    # Exact original numeric fixtures exercise all five source transformations.
    source = _source_pairs()
    for pair in source:
        c, ps, _ = _decode_input(pair["input"])
        assert np.array_equal(_predict_decoded(c, ps)[0], pair["output"])
    for pair in source[3:]:
        assert np.array_equal(solve(source[:3], pair["input"]), pair["output"])
    incomplete = source[-1]["input"].copy()
    incomplete[5, 1:6] = 0
    try:
        _decode_input(incomplete)
    except AssertionError:
        pass
    else:
        raise AssertionError("an erased tile row was silently completed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    features = {s: [pair_features(pair) for p in puzzles for pair in p[s]] for s in ("train", "test")}
    all_f = features["train"] + features["test"]
    for fs in features.values():
        assert {v for f in fs for v in f["tile_sizes"]} == {2, 3, 4, 5}
        assert {v for f in fs for v in f["controls"]} == {(r, a) for r in range(4) for a in range(4)}
        assert {f["layout"] for f in fs} == {"left", "right", "flank"}
        assert len({f["canvas_shape"] for f in fs}) >= 25
    assert len({f["canvas_shape"] for f in all_f}) >= 65
    assert len({v for f in all_f for v in f["stencil_shapes"]}) >= 35
    assert max(v[0] for f in all_f for v in f["stencil_shapes"]) >= 8
    assert max(v[1] for f in all_f for v in f["stencil_shapes"]) >= 5
    assert max(f["painted_cells"] for f in all_f if f["panel_count"] == 2) >= 124
    assert any(f["rectangle_overlap"] for f in all_f)
    assert any(f["adjacent"] for f in all_f)
    assert any(max(f["stencil_components"]) >= 6 for f in all_f)
    assert any(max(f["stencil_holes"]) >= 1 for f in all_f)
    assert any(min(f["stencil_densities"]) < .3 for f in all_f)
    assert any(1.0 in f["stencil_densities"] for f in all_f)
    assert any(not all(f["opposite_extrema"]) for f in all_f)
    assert min(f["palette_size"] for f in all_f) <= 4
    assert max(f["palette_size"] for f in all_f) >= 8
    assert any(f["canvas_colors"] == 1 for f in all_f)
    assert {f["canvas_geometry"] for f in all_f} == {
        "uniform", "horizontal_bands", "vertical_bands", "multidirectional"}
    assert any(any(f["ground_tiles"]) for f in all_f)
    assert any(any(f["role_sharing"]) for f in all_f)
    assert any(any(f["canvas_tile_sharing"]) for f in all_f)
    assert len({v for f in all_f for v in f["tile_origins"]}) >= 15
    assert len({v for f in all_f for v in f["key_origins"]}) >= 30
    assert len({v for f in all_f for v in f["gaps"]}) >= 5
    # Combination coverage, not just independent marginal ranges.
    for n in (2, 3, 4, 5):
        assert len({control for f in all_f for size, control in zip(f["tile_sizes"], f["controls"])
                    if size == n}) >= 12
        assert {f["layout"] for f in all_f if n in f["tile_sizes"]} == {"left", "right", "flank"}
    assert any(f["panel_count"] == 2 and len(set(f["tile_sizes"])) == 2 for f in all_f)
    varied_counts, independent_controls = 0, 0
    for p in puzzles:
        fs = [pair_features(pair) for pair in p["train"]]
        varied_counts += len({f["panel_count"] for f in fs}) > 1
        controls = {v for f in fs for v in f["controls"]}
        independent_controls += (any(sum(a == anchor for _, a in controls) > 1 for anchor in range(4)) and
                                 any(sum(k == turn for k, _ in controls) > 1 for turn in range(4)))
        assert len({v for f in fs for v in f["grounds"]}) > 1
        for pair in p["test"]:
            assert np.array_equal(solve(p["train"], pair["input"].copy()), pair["output"])
    assert varied_counts >= 28 and independent_controls >= 28
    key = lambda p: tuple((_grid_key(x["input"]), _grid_key(x["output"])) for s in ("train", "test") for x in p[s])
    assert key(generate_puzzle(123)) == key(generate_puzzle(123))
    assert len({key(p) for p in puzzles}) == 42
    for nt, nq in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        for seed in (913, 2026):
            p = generate_puzzle(seed, nt, nq)
            assert nt is None or len(p["train"]) == nt
            assert nq is None or len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (2, 0), (10, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(4, nt, nq)
        except ValueError:
            continue
        raise AssertionError("invalid pair counts accepted")
    damaged = {s: [{r: a.copy() for r, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect label accepted")


# Compact immutable regression evidence; exact source contents, not templates.
_SOURCE_CASES = (
    (
        """000006555553333333333333333355
077706555533333333333333333333
077406555333333333333333333333
074706555333333338888888883333
000006555333333888888888888833
000006553333388888888888888883
111006553333888888888888888883
001006553338888888888888888883
001006533338888888111888888883
011106533388888881111888888883
111116533388888881111888888883
101016533388888881111888888883
666666533388888888118888888883
600066533388888888888888888833
600066533338888888888888888833
600466553338888888888888888333
666666555333888888888888883333
600066555333388888888888333335
607066555333333333333333333355
607066555533333333333333333555""",
        """555553333333333333333355
555533333333333333333333
555333333444333444333444
555333333447888447883447
555333333474888474888474
553333388444444444444444
553333888447447447447447
553338888474474474474474
533338888888444444444883
533388888881447447447883
533388888881474474474883
533388888881111444888883
533388888888118447888883
533388888888888474888833
533338888888888444888833
553338888888888447888333
555333888888888474883333
555333388888888444444444
555333333333333447447447
555533333333333474474474""",
    ),
    (
        """000006888889999988888999998888
011206888889999988888999998888
011106888889999988888999998888
021206888889999988888999998888
000006888889999988888999998888
040406888889999988888999998888
004006888889999988888999998888
004006888889999988888999998888
040406888889999988888999998888
040406888889999988888999998888
004006888889999988888999998888
000006888889999988888999998888
666666888889999988888999998888
640066888889999988888999998888
600066888889999988888999998888
600066888889999988888999998888
666666888889999988888999998888
607066888889999988888999998888
607066888889999988888999998888
600066888889999988888999998888""",
        """221889221988888999998888
222889222988888999998888
121889121988888999998888
888221999988888999998888
888222999988888999998888
888121999988888999998888
888221999988888999998888
888222999988888999998888
888121999988888999998888
221889221988888999998888
222889222988888999998888
121889121988888999998888
221889221988888999998888
222889222988888999998888
121889121988888999998888
888221999988888999998888
888222999988888999998888
888121999988888999998888
888889999988888999998888
888889999988888999998888""",
    ),
    (
        """000006555555555555555555555555
033306555555555555555555555555
083806555555555555555555555555
083306555555555555555555555555
000006555555555555555555555555
000006444444444444444444444444
022206444444444444444444444444
020006444444444444444444444444
022206444444444444444444444444
000206444444444444444444444444
000206555555555555555555555555
000006555555555555555555555555
666666555555555555555555555555
600466555555555555555555555555
600066555555555555555555555555
600066444444444444444444444444
666666444444444444444444444444
600066444444444444444444444444
607766444444444444444444444444
600066444444444444444444444444""",
        """555555555555555888888888
555555555555555383383383
555555555555555388388388
555555555555555888555888
555555555555555383555383
444444444444444388444388
444444444888888888444888
444444444383383383444383
444444444388388388444388
444444444444444444444444
555555555555555555555555
555555555555555555555555
555555555555555555555555
555555555555555555555555
555555555555555555555555
444444444444444444444444
444444444444444444444444
444444444444444444444444
444444444444444444444444
444444444444444444444444""",
    ),
    (
        """000069999999900009999999960000
011069999999900009999999960000
013069999999900009999999960240
000069999999900009999999960420
000069999999900009999999960000
000069999999900009999999960000
000069999999900009999999960030
000069999999900009999999960033
000069999999900009999999960003
070069999999900009999999963333
770069999999900009999999963030
700769999999900009999999963333
777769999999900009999999960000
070060000000000000000000060000
077760000000000000000000060000
000760000000000000000000060000
077760000000000000000000060000
000069999999900009999999960000
000069999999900009999999960000
000069999999900009999999960000
000069999999900009999999960000
000069999999900009999999960000
666669999999900009999999966666
600069999999900009999999964006
600069999999900009999999960006
600469999999900009999999960006
666669999999900009999999966666
607069999999900009999999960006
607069999999900009999999960706
600069999999900009999999960706""",
        """42424242000099999999
24242424000099999999
99429942000099999999
99249924000099999999
42424242000099999999
24242424000099999999
42999999000099999999
24999999000099999999
42429999000099999999
24249999000099999999
99429999000099999999
99249999000099999999
99999999000099999999
00000000000000000000
00000000000000330000
00000000000000310000
00000000000033330000
99999999000031319999
99999999000033999933
99999999000031999931
99999999000033333333
99999999000031313131
99999999000099339999
99999999000099319999
99999999000099333333
99999999000099313131
99999999000099999933
99999999000099999931
99999999000099333333
99999999000099313131""",
    ),
    (
        """000000064444444444444444444444
022222064444444444444444444444
023333064444444444444444444444
022222064444444444444444444444
033332064444444444444444444444
022222061111111111111111111111
000000061111111111111111111111
000000061111111111111111111111
000000061111111111111111111111
000000061111111111111111111111
000000061111111111111111111111
000000061111111111111111111111
044444061111111111111111111111
044004064444444444444444444444
040004064444444444444444444444
044400064444444444444444444444
000000064444444444444444444444
000000064444444444444444444444
000000064444444444444444444444
000000064444444444444444444444
000000064444444444444444444444
000000064444444444444444444444
666666661111111111111111111111
660006661111111111111111111111
660006661111111111111111111111
664006661111111111111111111111
666666661111111111111111111111
660006661111111111111111111111
667706661111111111111111111111
660006661111111111111111111111""",
        """4444444444444444444444
4444444444444444444444
4444444444444444444444
4444444444444444444444
4444444444444444444444
3333333333333331111111
3222232222322221111111
3333333333333331111111
2222322223222231111111
3333333333333331111111
3333311111111111111111
3222211111111111111111
3333311111111111111111
2222344444444444444444
3333344444444444444444
3333344444444443333344
3222244444444443222244
3333344444444443333344
2222344444444442222344
3333344444444443333344
3333333333444443333344
3222232222444443222244
3333333333111113333311
2222322223111112222311
3333333333111113333311
3333333333333333333311
3222232222322223222211
3333333333333333333311
2222322223222232222311
3333333333333333333311""",
    ),
)

_SOURCE_KEYS = frozenset((_grid_key(p["input"]), _grid_key(p["output"])) for p in _source_pairs())
