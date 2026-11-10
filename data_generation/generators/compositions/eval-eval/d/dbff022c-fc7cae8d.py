"""Oriented stencil filling: local side correspondence controls a cavity texture.

No parent generator is imported. The decoder uses only the supplied input.
"""
import json
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(int, copy=False)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(body, bg, diagonal=False):
    """Return background regions that cannot escape the body's bounding box."""
    h, w = body.shape
    regions = []
    for cells in _components(body == bg, diagonal):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            regions.append(cells)
    return regions


def _matrix(source, target):
    assert set(source.values()) == set(target.values()) and len(source) == 2
    matrix = np.zeros((2, 2), dtype=int)
    for side, color in source.items():
        other = next(s for s, v in target.items() if v == color)
        matrix += np.outer(NORMAL[other], NORMAL[side])
    assert np.array_equal(matrix.T @ matrix, np.eye(2, dtype=int))
    return matrix


def _transform(a, matrix):
    positions = matrix @ np.indices(a.shape).reshape(2, -1)
    positions -= positions.min(axis=1, keepdims=True)
    result = np.empty(tuple(positions.max(axis=1) + 1), dtype=a.dtype)
    result[positions[0], positions[1]] = a.ravel()
    return result


def _adjacent(index):
    return SIDES[index], SIDES[(index + 1) % 4]


def _guides(index, colors, reverse=False):
    return dict(zip(_adjacent(index), colors[::-1] if reverse else colors))


def _bar_cells(bounds, side):
    r, c, h, w = bounds
    if side == "T":
        return [(r - 1, x) for x in range(c, c + w)]
    if side == "B":
        return [(r + h, x) for x in range(c, c + w)]
    if side == "L":
        return [(x, c - 1) for x in range(r, r + h)]
    return [(x, c + w) for x in range(r, r + h)]


def _corner(bounds, guides):
    r, c, h, w = bounds
    return (r - 1 if "T" in guides else r + h,
            c - 1 if "L" in guides else c + w)


def _put_key(grid, sources, fills, side, start):
    h, w = grid.shape
    cells = []
    for i, (source, fill) in enumerate(zip(sources, fills)):
        p = start + i
        outer, inner = {"T": ((0, p), (1, p)), "B": ((h - 1, p), (h - 2, p)),
                        "L": ((p, 0), (p, 1)), "R": ((p, w - 1), (p, w - 2))}[side]
        grid[outer], grid[inner] = source, fill
        cells.extend((outer, inner))
    return cells


def _decode(grid):
    grid = _grid(grid)
    H, W = grid.shape
    bg = int(grid[0, 0])
    assert all(grid[q] == bg for q in ((0, -1), (-1, 0), (-1, -1))), "corners name background"
    keys = []
    for cells in _components(grid != bg, True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw:
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 4:
            if r0 == 0 and 0 < c0 <= c1 < W - 1:
                lanes.append(("T", grid[0, c0:c1 + 1], grid[1, c0:c1 + 1]))
            if r1 == H - 1 and 0 < c0 <= c1 < W - 1:
                lanes.append(("B", grid[-1, c0:c1 + 1], grid[-2, c0:c1 + 1]))
        if sw == 2 and 2 <= sh <= 4:
            if c0 == 0 and 0 < r0 <= r1 < H - 1:
                lanes.append(("L", grid[r0:r1 + 1, 0], grid[r0:r1 + 1, 1]))
            if c1 == W - 1 and 0 < r0 <= r1 < H - 1:
                lanes.append(("R", grid[r0:r1 + 1, -1], grid[r0:r1 + 1, -2]))
        for side, outer, inner in lanes:
            if len(set(map(int, outer))) == len(outer):
                keys.append(dict(side=side, sources=tuple(map(int, outer)),
                                 fills=tuple(map(int, inner)), cells=cells))
    assert len(keys) == 1, "one isolated edge key is required"
    key = keys[0]
    mapping = dict(zip(key["sources"], key["fills"]))
    key_colors = set(key["sources"] + key["fills"])
    # Marker colors occur exclusively in isolated, straight, interior bars.
    # The stencil's corner tag is a singleton of its ink color, so ink cannot
    # be mistaken for a marker, even when the sample contains straight strokes.
    bars_by_color = {}
    for color in sorted(set(map(int, np.unique(grid))) - key_colors - {bg}):
        descriptions = []
        for cells in _components(grid == color):
            rr, cc = zip(*cells)
            if len(cells) < 2 or any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
                break
            if len(set(rr)) == 1:
                descriptions.append(dict(axis="H", fixed=rr[0], lo=min(cc), hi=max(cc), cells=cells, color=color))
            elif len(set(cc)) == 1:
                descriptions.append(dict(axis="V", fixed=cc[0], lo=min(rr), hi=max(rr), cells=cells, color=color))
            else:
                break
        else:
            if len(descriptions) >= 2:
                bars_by_color[color] = descriptions
    assert len(bars_by_color) >= 2, "two local marker identities are required"
    bars = [b for descriptions in bars_by_color.values() for b in descriptions]
    candidates = []
    for horizontal in (b for b in bars if b["axis"] == "H"):
        for vertical in (b for b in bars if b["axis"] == "V" and b["color"] != horizontal["color"]):
            hr, hs, he = horizontal["fixed"], horizontal["lo"], horizontal["hi"]
            vc, vs, ve = vertical["fixed"], vertical["lo"], vertical["hi"]
            if vc == hs - 1:
                vside = "L"
            elif vc == he + 1:
                vside = "R"
            else:
                continue
            if hr == vs - 1:
                hside = "T"
            elif hr == ve + 1:
                hside = "B"
            else:
                continue
            r, c, h, w = vs, hs, ve - vs + 1, he - hs + 1
            if r < 2 or c < 2 or r + h > H - 2 or c + w > W - 2:
                continue
            guides = {hside: horizontal["color"], vside: vertical["color"]}
            bounds = (r, c, h, w)
            body = grid[r:r + h, c:c + w].copy()
            colors = set(map(int, np.unique(body))) - {bg}
            if len(colors) != 1 or colors & {horizontal["color"], vertical["color"]}:
                continue
            color = next(iter(colors))
            tag = int(grid[_corner(bounds, guides)])
            # Exact bars and empty opposite sides give the extent; a one-cell
            # background moat separates this local coordinate frame from others.
            occupied = set(horizontal["cells"] + vertical["cells"])
            valid = True
            for side in SIDES:
                if any(grid[q] != guides.get(side, bg) for q in _bar_cells(bounds, side)):
                    valid = False
            er0, er1 = r - ("T" in guides), r + h + ("B" in guides)
            ec0, ec1 = c - ("L" in guides), c + w + ("R" in guides)
            ring = np.concatenate((grid[er0 - 1, ec0 - 1:ec1 + 1], grid[er1, ec0 - 1:ec1 + 1],
                                   grid[er0:er1, ec0 - 1], grid[er0:er1, ec1]))
            if not valid or np.any(ring != bg):
                continue
            corner_cells = ((r - 1, c - 1), (r - 1, c + w), (r + h, c - 1), (r + h, c + w))
            if any(grid[q] != bg for q in corner_cells if q != _corner(bounds, guides)):
                continue
            if tag == color and color not in key_colors:
                kind = "reference"
            elif tag == bg:
                kind = "mold"
                rr, cc = np.nonzero(body != bg)
                if min(rr) != 0 or max(rr) != h - 1 or min(cc) != 0 or max(cc) != w - 1:
                    continue
                if len(_components(body != bg, True)) != 1:
                    continue
            else:
                continue
            candidates.append(dict(kind=kind, bounds=bounds, body=body, color=color,
                                   guides=guides, bar_cells=occupied, holes=_holes(body, bg)))
    references = [item for item in candidates if item["kind"] == "reference"]
    assert len(references) == 1, "one tagged stencil sample"
    reference = references[0]
    markers = set(reference["guides"].values())
    molds = [item for item in candidates if item["kind"] == "mold" and set(item["guides"].values()) == markers]
    assert 1 <= len(molds) <= 6, "one to six molds"
    candidates = [reference] + molds
    bars = [b for color in markers for b in bars_by_color[color]]
    assert bg in reference["body"] and np.any(reference["body"] == reference["color"])
    assert reference["color"] not in {m["color"] for m in molds}
    # Every bar is assigned once. No unidentified content is silently ignored.
    used_bars = [q for item in candidates for q in item["bar_cells"]]
    assert len(used_bars) == len(set(used_bars))
    assert set(used_bars) == {q for b in bars for q in b["cells"]}
    explained = set(key["cells"]) | set(used_bars) | {_corner(reference["bounds"], reference["guides"])}
    for item in candidates:
        r, c, h, w = item["bounds"]
        explained.update((r + int(y), c + int(x)) for y, x in np.argwhere(item["body"] != bg))
    assert explained == set(map(tuple, np.argwhere(grid != bg))), "unidentified ink"
    for mold in molds:
        mold["matrix"] = _matrix(reference["guides"], mold["guides"])
    return dict(grid=grid, background=bg, key=key, mapping=mapping, reference=reference, molds=molds)


def _paint(info, *, use_mapping=True, use_orientation=True, solid=False, restart=False, recursive=False, tight_crop=False, wall_phase=None):
    result = info["grid"].copy()
    bg, mapping, reference = info["background"], info["mapping"], info["reference"]
    stencil = reference["body"] == reference["color"]
    if tight_crop:
        stencil = _crop(stencil)
    for mold in info["molds"]:
        if not mold["holes"] or (use_mapping and mold["color"] not in mapping):
            continue
        fill = mapping[mold["color"]] if use_mapping else mold["color"]
        if recursive:
            seen = {mold["color"]}
            while fill in mapping and fill not in seen:
                seen.add(fill)
                fill = mapping[fill]
        tile = _transform(stencil, mold["matrix"]) if use_orientation else stencil
        all_cells = [q for hole in mold["holes"] for q in hole]
        origin = (min(q[0] for q in all_cells), min(q[1] for q in all_cells))
        if wall_phase is not None:
            origin = (wall_phase, wall_phase)
        r, c, _, _ = mold["bounds"]
        for hole in mold["holes"]:
            phase = (min(q[0] for q in hole), min(q[1] for q in hole)) if restart else origin
            for y, x in hole:
                if solid or tile[(y - phase[0]) % tile.shape[0], (x - phase[1]) % tile.shape[1]]:
                    result[r + y, c + x] = fill
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the local key, reference and coordinate frames from this input."""
    return _paint(_decode(input_grid))


def _dilate(mask, diagonal=False):
    result = mask.copy()
    steps = STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    for dr, dc in steps:
        r0, r1 = max(0, dr), min(mask.shape[0], mask.shape[0] + dr)
        c0, c1 = max(0, dc), min(mask.shape[1], mask.shape[1] + dc)
        result[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return result


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[min(rr):max(rr) + 1, min(cc):max(cc) + 1].copy()


def _shape(rng, kind, small=False):
    if kind == "teaching":
        # Two unlike holes share one lattice. The left diamond seals in four-
        # connectivity through diagonal wall contacts; the right hole is concave.
        rr, cc = np.indices((5, 5))
        diamond = abs(rr - 2) + abs(cc - 2) == 2
        ph, pw = (3, 3) if small else (int(rng.integers(2, 4)), int(rng.integers(2, 5)))
        hole = np.zeros((ph + 2, pw + 2), dtype=bool)
        hole[1:-1, 1:-1] = True
        hole[ph, pw] = False
        if pw >= 3 and rng.random() < .5:
            hole[ph, pw - 1] = False
        outline = _dilate(hole) & ~hole
        gap = 1 if small else int(rng.integers(1, 4))
        offset = 0 if small else int(rng.integers(2))
        mask = np.zeros((max(5, outline.shape[0] + offset), 5 + gap + outline.shape[1]), dtype=bool)
        mask[:5, :5] = diamond
        mask[offset:offset + outline.shape[0], 5 + gap:] = outline
        mask[2, 4:6 + gap] = True
    elif kind == "diamond":
        rad = int(rng.integers(1, 3 if small else 5))
        rr, cc = np.indices((2 * rad + 1, 2 * rad + 1))
        mask = abs(rr - rad) + abs(cc - rad) == rad
    elif kind == "multi":
        h, w = (5, 6) if small else (int(rng.integers(5, 10)), int(rng.integers(5, 12)))
        mask = np.ones((h, w), dtype=bool)
        if rng.random() < .5:
            mask[1:-1, 1:-1] = False
            mask[h // 2, :] = True
        else:
            mask[1:-1, 1:-1] = False
            mask[:, w // 2] = True
        if not small and min(h, w) >= 7 and rng.random() < .5:
            mask[h // 2, :] = True
            mask[:, w // 2] = True
        if rng.random() < .7:
            mask[1, 1] = True  # Unequal cavity geometry and lattice phase.
    else:
        h = int(rng.integers(1, 3 if small else 7))
        w = int(rng.integers(1, 4 if small else 8))
        hole = np.ones((h, w), dtype=bool)
        if kind == "profile" and h >= 2 and w >= 2:
            left, right = 0, w - 1
            for y in range(h):
                anchor = int(rng.integers(left, right + 1))
                left, right = int(rng.integers(anchor + 1)), int(rng.integers(anchor, w))
                hole[y, :] = False
                hole[y, left:right + 1] = True
        thickness = 1 if small else int(rng.choice([1, 1, 1, 2]))
        hole = np.pad(hole, thickness)
        mask = hole.copy()
        for _ in range(thickness):
            mask = _dilate(mask, kind in ("rectangle", "open") or rng.random() < .3)
        mask &= ~hole
        if kind == "open":
            y, x = np.argwhere(hole)[0]
            mask[:y + 1, x] = False
    mask = _crop(mask)
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    return mask.copy()


def _sample_stencil(rng, blank_edge=False):
    for _ in range(100):
        shape = (int(rng.integers(2, 6)), int(rng.integers(2, 6)))
        mask = rng.random(shape) < rng.uniform(.3, .7)
        if blank_edge:
            side = str(rng.choice(SIDES))
            if side == "T":
                mask[0, :] = False
            elif side == "B":
                mask[-1, :] = False
            elif side == "L":
                mask[:, 0] = False
            else:
                mask[:, -1] = False
        if not (np.any(mask) and np.any(~mask)):
            continue
        # Compare periodic fields rather than array shapes: a rectangular
        # checkerboard must not pass as asymmetric just because axes swap.
        fields = set()
        yy, xx = np.indices((12, 12))
        for flip in (False, True):
            for turn in range(4):
                tile = np.rot90(np.fliplr(mask) if flip else mask, turn)
                fields.add(tile[yy % tile.shape[0], xx % tile.shape[1]].tobytes())
        if len(fields) == 8:
            return mask
    # An asymmetric, primitive 3-by-4 binary pattern has eight periodic fields.
    return np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]], dtype=bool)


def _pack(rng, sizes, safe=False):
    area = sum(h * w for h, w in sizes)
    for attempt in range(65):
        H = W = 30
        if attempt < 40:
            H, W = int(rng.integers(max(14, max(h for h, w in sizes) + 4), 31)), int(rng.integers(max(14, max(w for h, w in sizes) + 4), 31))
            if rng.random() < .18:
                H = W = max(H, W)
        if (H - 4) * (W - 4) < area * 1.15:
            continue
        blocked = np.zeros((H - 4, W - 4), dtype=bool)
        positions = [None] * len(sizes)
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
        for i in order:
            h, w = sizes[i]
            legal = [(r, c) for r in range(blocked.shape[0] - h + 1)
                     for c in range(blocked.shape[1] - w + 1) if not np.any(blocked[r:r + h, c:c + w])]
            if not legal:
                break
            r, c = legal[int(rng.integers(len(legal)))]
            blocked[r:r + h, c:c + w] = True
            positions[i] = (r + 2, c + 2)
        else:
            return (H, W), positions
    if safe:
        # Last construction uses small bodies; a height-sorted shelf always
        # fits these at most seven panels in the 26-by-26 interior.
        positions = [None] * len(sizes)
        row = col = line_height = 0
        for i in sorted(range(len(sizes)), key=lambda i: -sizes[i][0]):
            h, w = sizes[i]
            if col + w > 26:
                row += line_height
                col = line_height = 0
            assert row + h <= 26, "safe shelf exceeds canvas"
            positions[i] = (row + 2, col + 2)
            col += w
            line_height = max(line_height, h)
        return (30, 30), positions
    return None


def _render(rng, stencil, masks, colors, guides, ref_guides, palette, key_side, sources, fills, safe=False):
    bodies = [stencil] + masks
    all_guides = [ref_guides] + guides
    packed = _pack(rng, [(a.shape[0] + 3, a.shape[1] + 3) for a in bodies], safe)
    if packed is None:
        return None
    shape, places = packed
    bg, ink = palette[0], palette[3]
    grid = np.full(shape, bg, dtype=int)
    for i, (body, guide, (pr, pc)) in enumerate(zip(bodies, all_guides, places)):
        r, c = pr + 1 + ("T" in guide), pc + 1 + ("L" in guide)
        h, w = body.shape
        bounds = (r, c, h, w)
        grid[r:r + h, c:c + w][body] = ink if i == 0 else colors[i - 1]
        for side, color in guide.items():
            for q in _bar_cells(bounds, side):
                grid[q] = color
        if i == 0:
            grid[_corner(bounds, guide)] = ink
    axis = shape[1] if key_side in ("T", "B") else shape[0]
    start = int(rng.integers(1, axis - len(sources)))
    _put_key(grid, sources, fills, key_side, start)
    return grid


def _make_pair(rng, held_corner, teaching=False, test=False):
    attempts = 64 if teaching else 24
    for attempt in range(attempts):
        safe = attempt == attempts - 1
        palette = list(map(int, rng.permutation(10)))
        bg, m1, m2, ink = palette[:4]
        pool = palette[4:]
        n = 3 if teaching else int(rng.integers(2, 5))
        sources, free = pool[:n], pool[n:]
        style = int(rng.integers(5))
        if style == 0:
            fills = list(map(int, rng.choice(sources + free[:-1], n)))
        elif style == 1:
            fills = [int(rng.choice(free[:-1]))] * n
        elif style == 2:
            fills = list(map(int, rng.choice(free[:-1], n)))
        elif style == 3:
            fills = sources[1:] + sources[:1]
        else:
            fills = sources.copy()
        if teaching:
            fills[0], fills[1], fills[2] = sources[1], free[0], sources[2]
        inert = free[0] if rng.random() < .5 else free[-1]
        ref_index = int(rng.integers(4))
        ref_guides = _guides(ref_index, [m1, m2], bool(rng.integers(2)))
        train_corners = [i for i in range(4) if i != held_corner]
        allowed = list(range(4)) if test else train_corners
        if teaching:
            specs = [("teaching", sources[0]), ("open", sources[0]),
                     ("rectangle", inert), ("rectangle", sources[2])]
        else:
            count = int(rng.integers(1, 7))
            first_kind = str(rng.choice(["rectangle", "profile", "diamond", "multi"]))
            specs = [(first_kind, sources[0])]
            for _ in range(count - 1):
                kind = str(rng.choice(["rectangle", "profile", "diamond", "multi", "open"]))
                color = int(rng.choice(sources + [inert]))
                specs.append((kind, color))
        masks = []
        for kind, color in specs:
            if safe and kind != "teaching":
                mask = np.ones((3, 3), dtype=bool)
                mask[1, 1] = False
                if kind == "open":
                    mask[0, 1] = False
            elif teaching and kind == "rectangle":
                # The self-mapped mold has two-cell walls. Its cavity anchor
                # differs from both the wall corner and a fixed one-cell inset.
                thick = color == sources[2]
                size, inset = (7, 2) if thick else (5, 1)
                mask = np.ones((size, size), dtype=bool)
                mask[inset:size - inset, inset:size - inset] = False
                mask = np.rot90(mask, int(rng.integers(4)))
            else:
                mask = _shape(rng, kind, small=attempt >= 3 or len(specs) >= 5)
            masks.append(mask)
        all_guides = []
        for index, mask in enumerate(masks):
            options = [_guides(corner, [m1, m2], reverse) for corner in allowed for reverse in (False, True)]
            if teaching and index == 0:
                options = [g for g in options if int(round(np.linalg.det(_matrix(ref_guides, g)))) == 1
                           and _matrix(ref_guides, g)[0, 0] == 0]
            if test and index == 0:
                options = [_guides(held_corner, [m1, m2], reverse) for reverse in (False, True)]
            all_guides.append(options[int(rng.integers(len(options)))])
        stencil = _sample_stencil(rng, blank_edge=teaching)
        order = rng.permutation(n)
        ordered_sources, ordered_fills = [sources[i] for i in order], [fills[i] for i in order]
        grid = _render(rng, stencil, masks, [c for k, c in specs], all_guides,
                       ref_guides, palette, str(rng.choice(SIDES)), ordered_sources, ordered_fills, safe)
        if grid is None:
            continue
        info = _decode(grid)
        out = _paint(info)
        if np.array_equal(out, grid):
            continue
        if test:
            novel = [m for m in info["molds"] if set(m["guides"]) == set(_adjacent(held_corner))
                     and m["color"] in info["mapping"] and m["holes"]]
            if not any(any(out[m["bounds"][0] + y, m["bounds"][1] + x] != bg
                           for hole in m["holes"] for y, x in hole) for m in novel):
                continue
        if teaching:
            active = next(m for m in info["molds"] if m["color"] == sources[0] and m["holes"])
            # A constructive witness, not a seed-dependent solver convention.
            if np.array_equal(out, _paint(info, recursive=True)) or np.array_equal(out, _paint(info, restart=True)):
                continue
            if np.array_equal(out, _paint(info, use_orientation=False)):
                continue
            if np.array_equal(out, _paint(info, tight_crop=True)):
                continue
            if any(np.array_equal(out, _paint(info, wall_phase=inset)) for inset in (0, 1)):
                continue
            inv_info = dict(info, molds=[dict(m, matrix=m["matrix"].T) for m in info["molds"]])
            if np.array_equal(out, _paint(inv_info)):
                continue
            clone = grid.copy()
            a, b = active["guides"].values()
            for side, color in active["guides"].items():
                for q in _bar_cells(active["bounds"], side):
                    clone[q] = b if color == a else a
            if np.array_equal(out, solve(clone)):
                continue
            # Each component of the teaching mold exhibits paint and a blank,
            # so its diagonal and irregular cavities visibly affect the answer.
            r, c, _, _ = active["bounds"]
            if any(not any(out[r + y, c + x] != bg for y, x in hole)
                   for hole in active["holes"]):
                continue
            self_mold = next(m for m in info["molds"] if m["color"] == sources[2])
            sr, sc, _, _ = self_mold["bounds"]
            if not any(out[sr + y, sc + x] != bg for hole in self_mold["holes"] for y, x in hole):
                continue
        return {"input": grid, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _contrast(rng, pair):
    grid = pair["input"].copy()
    info = _decode(grid)
    mold = next(m for m in info["molds"] if len(m["holes"]) >= 2 and m["color"] in info["mapping"])
    a, b = mold["guides"].values()
    for side, color in mold["guides"].items():
        for q in _bar_cells(mold["bounds"], side):
            grid[q] = b if color == a else a
    for q in info["key"]["cells"]:
        grid[q] = info["background"]
    # Move only the lookup strip to an adjacent edge; its direction stays inward.
    side = SIDES[(SIDES.index(info["key"]["side"]) + 1) % 4]
    axis = grid.shape[1] if side in ("T", "B") else grid.shape[0]
    start = int(rng.integers(1, axis - len(info["mapping"])))
    _put_key(grid, info["key"]["sources"], info["key"]["fills"], side, start)
    result = {"input": grid, "output": solve(grid)}
    assert not np.array_equal(result["output"], pair["output"])
    return result


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _signature(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    held_corner = int(rng.integers(4))
    first = _make_pair(rng, held_corner, teaching=True)
    puzzle = {"train": [first, _contrast(rng, first)], "test": []}
    seen = {_signature(p["input"]) for p in puzzle["train"]}
    for split, extra in (("train", nt - 2), ("test", ne)):
        for _ in range(extra):
            for retry in range(30):
                pair = _make_pair(rng, held_corner, test=split == "test")
                signature = _signature(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    assert out.shape == inp.shape and np.array_equal(_paint(info), out), "wrong stencil fill"
    bg = info["background"]
    assert np.array_equal(out[inp != bg], inp[inp != bg]), "existing ink changed"
    permitted = np.zeros(inp.shape, dtype=bool)
    active, inactive, opened, diagonal, irregular, multihole, self_map = 0, 0, 0, False, False, False, False
    transforms = set()
    for mold in info["molds"]:
        mapped = mold["color"] in info["mapping"]
        r, c, _, _ = mold["bounds"]
        holes = mold["holes"]
        active += mapped and bool(holes)
        inactive += not mapped and bool(holes)
        opened += mapped and not holes
        multihole |= mapped and len(holes) > 1
        diagonal |= mapped and sum(map(len, holes)) > sum(map(len, _holes(mold["body"], bg, True)))
        for hole in holes:
            rr, cc = zip(*hole)
            irregular |= mapped and len(hole) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
            if mapped:
                for y, x in hole:
                    permitted[r + y, c + x] = True
                    if info["mapping"][mold["color"]] == mold["color"] and out[r + y, c + x] != bg:
                        self_map = True
        if mapped and holes:
            transforms.add(tuple(mold["matrix"].ravel()))
    assert not np.any((out != inp) & ~permitted), "paint escaped an eligible cavity"
    assert np.any(out != inp), "scene must visibly paint"
    return dict(info=info, active=active, inactive=inactive, opened=opened, diagonal=diagonal,
                irregular=irregular, multihole=multihole, self_map=self_map, transforms=transforms,
                chain=not np.array_equal(out, _paint(info, recursive=True)),
                shared_phase=not np.array_equal(out, _paint(info, restart=True)),
                orientation=not np.array_equal(out, _paint(info, use_orientation=False)),
                solid=not np.array_equal(out, _paint(info, solid=True)),
                full_extent=not np.array_equal(out, _paint(info, tight_crop=True)),
                cavity_anchor=all(not np.array_equal(out, _paint(info, wall_phase=inset)) for inset in (0, 1)),
                mapping=not np.array_equal(out, _paint(info, use_mapping=False)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            sig = _signature(_grid(pair["input"]))
            assert sig not in seen, "duplicate input"
            seen.add(sig)
            analyses[split].append(_analyze(pair))
    train = analyses["train"]
    for feature in ("inactive", "opened", "diagonal", "irregular", "multihole", "self_map",
                    "chain", "shared_phase", "orientation", "solid", "mapping", "full_extent", "cavity_anchor"):
        assert any(a[feature] for a in train), "missing training witness: " + feature
    matrices = {m for a in train for m in a["transforms"]}
    assert any(m[0] == 0 and m[3] == 0 and m[1] == -m[2] for m in matrices), "no quarter-turn witness"
    assert any(m[0] * m[3] - m[1] * m[2] == -1 for m in matrices), "no reflection witness"
    assert len({a["info"]["key"]["side"] for a in train}) >= 2, "key direction needs two edges"
    # Check a controlled, local correspondence reversal from the visible grids.
    contrasts = False
    for i, a in enumerate(train):
        for j, b in enumerate(train[i + 1:], i + 1):
            ia, ib = a["info"], b["info"]
            if ia["grid"].shape != ib["grid"].shape or ia["mapping"] != ib["mapping"]:
                continue
            aa, bb = ia["grid"].copy(), ib["grid"].copy()
            for info, grid in ((ia, aa), (ib, bb)):
                for q in info["key"]["cells"]:
                    grid[q] = info["background"]
                for mold in info["molds"]:
                    for q in mold["bar_cells"]:
                        grid[q] = info["background"]
            contrasts |= np.array_equal(aa, bb) and not np.array_equal(puzzle["train"][i]["output"], puzzle["train"][j]["output"])
    assert contrasts, "no same-content correspondence contrast"
    sample = train[0]["info"]["reference"]["body"]
    bg = train[0]["info"]["background"]
    assert any(np.all(line == bg) for line in (sample[0], sample[-1], sample[:, 0], sample[:, -1]))
    corners = {frozenset(m["guides"]) for a in train for m in a["info"]["molds"]}
    for a in analyses["test"]:
        info = a["info"]
        output = _paint(info)
        assert any(frozenset(m["guides"]) not in corners and m["color"] in info["mapping"]
                   and any(output[m["bounds"][0] + y, m["bounds"][1] + x] != info["background"]
                           for hole in m["holes"] for y, x in hole) for m in info["molds"]), "missing novel-frame test"


def self_test():
    """Bounded semantic and diversity checks on 42 seeds, plus count boundaries."""
    # Hand-derived regression: transpose the 2-by-3 sample, repeat the resulting
    # 3-by-2 tile, then clip it to a 3-by-5 cavity. This does not call _paint to
    # construct the expected output and checks exact new colors and tile phase.
    stencil = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    ring = np.ones((5, 7), dtype=bool)
    ring[1:-1, 1:-1] = False
    rng = np.random.default_rng(891)
    manual = _render(rng, stencil, [ring], [4], [{"T": 2, "L": 1}],
                     {"T": 1, "L": 2}, [0, 1, 2, 8], "T", [4, 7], [7, 9], True)
    info = _decode(manual)
    r, c, _, _ = info["molds"][0]["bounds"]
    expected = manual.copy()
    impression = np.array([[7, 0, 7, 0, 7], [0, 7, 0, 7, 0], [0, 7, 0, 7, 0]])
    expected[r + 1:r + 4, c + 1:c + 6] = impression
    assert np.array_equal(solve(manual), expected)
    thick = np.ones((7, 9), dtype=bool)
    thick[2:5, 2:7] = False
    thick_input = _render(rng, stencil, [thick], [4], [{"T": 2, "L": 1}],
                          {"T": 1, "L": 2}, [0, 1, 2, 8], "T", [4, 7], [7, 9], True)
    thick_info = _decode(thick_input)
    tr, tc, _, _ = thick_info["molds"][0]["bounds"]
    thick_expected = thick_input.copy()
    thick_expected[tr + 2:tr + 5, tc + 2:tc + 7] = impression
    assert np.array_equal(solve(thick_input), thick_expected)
    # The same lattice crosses a dividing wall; it does not restart on the right.
    divided = manual.copy()
    divided[r:r + 5, c + 3] = 4
    divided_expected = expected.copy()
    divided_expected[r:r + 5, c + 3] = 4
    assert np.array_equal(solve(divided), divided_expected)
    opened = manual.copy()
    opened[r, c + 2] = 0
    assert np.array_equal(solve(opened), opened)
    # Altering only the active key destination changes color, not print geometry.
    recolored = manual.copy()
    key = info["key"]
    for q in key["cells"]:
        recolored[q] = 0
    start = min(q[1] for q in key["cells"])
    _put_key(recolored, [4, 7], [9, 9], "T", start)
    recolored_expected = recolored.copy()
    recolored_expected[r + 1:r + 4, c + 1:c + 6] = np.where(impression == 7, 9, 0)
    assert np.array_equal(solve(recolored), recolored_expected)
    puzzles, shapes, mold_shapes, stencil_shapes, structures, matrices, key_sides = [], set(), set(), set(), set(), set(), set()
    backgrounds, counts, signatures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(np.array(pair["input"])), pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                shapes.add(pair["input"].shape)
                stencil_shapes.add(info["reference"]["body"].shape)
                key_sides.add(info["key"]["side"])
                backgrounds.add(info["background"])
                counts.add(len(info["molds"]))
                roles = {}
                normalized = tuple(roles.setdefault(int(v), len(roles)) for v in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                for mold in info["molds"]:
                    mold_shapes.add((mold["body"].shape, (mold["body"] != info["background"]).tobytes()))
                    matrices.add(tuple(mold["matrix"].ravel()))
        if seed < 3:
            other = generate_puzzle(seed)
            assert all(np.array_equal(puzzle[s][i][k], other[s][i][k])
                       for s in puzzle for i in range(len(puzzle[s])) for k in ("input", "output"))
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 30 and len(mold_shapes) >= 70 and len(stencil_shapes) >= 10
    assert len(matrices) == 8 and key_sides == set(SIDES) and backgrounds == set(range(10))
    assert counts == set(range(1, 7))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(2000 + 10 * nt + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A full color permutation preserves roles. Rotated input layouts remain
    # decodable, although the display-relative lattice phase is recomputed.
    for turn in range(4):
        pair = puzzles[0]["test"][0]
        inp = np.rot90(pair["input"], turn)
        # Lattice phase is defined in the displayed mold frame, so rotating
        # the whole scene need not rotate its prior phase. Check a fresh solve
        # and its invariant rather than asserting an unjustified covariance.
        expected = solve(inp)
        _analyze({"input": inp, "output": expected})
        permutation = np.random.default_rng(190 + turn).permutation(10)
        assert np.array_equal(solve(permutation[inp]), permutation[expected])
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed noncavity output accepted")
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes), "mold_silhouettes": len(mold_shapes),
            "stencil_shapes": len(stencil_shapes), "symmetries": len(matrices), "mold_counts": sorted(counts),
            "color_normalized_inputs": len(structures)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
