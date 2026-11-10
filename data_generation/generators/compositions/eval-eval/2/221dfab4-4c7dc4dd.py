"""Scanned restoration: locally demonstrated stamps selected by an edge ruler.

Only NumPy and the standard library are used. No parent code or saved data is
needed. See task_spec.md for the input language and precedence of operations.
"""
from collections import Counter

import numpy as np


# Eight rigid symmetries, acting on (row, column) offsets.
_D4 = (((1, 0), (0, 1)), ((0, 1), (-1, 0)),
       ((-1, 0), (0, -1)), ((0, -1), (1, 0)),
       ((1, 0), (0, -1)), ((0, -1), (-1, 0)),
       ((-1, 0), (0, 1)), ((0, 1), (1, 0)))
_SIDES = ("top", "right", "bottom", "left")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    parts = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        parts.append(cells)
    return parts


def _offsets(mask, anchor):
    return {(int(r) - anchor[0], int(c) - anchor[1]) for r, c in np.argwhere(mask)}


def _transform(points, symmetry):
    m = _D4[symmetry]
    return {(m[0][0] * r + m[0][1] * c,
             m[1][0] * r + m[1][1] * c) for r, c in points}


def _place(points, anchor, symmetry):
    return {(anchor[0] + r, anchor[1] + c) for r, c in _transform(points, symmetry)}


def _canonical(a, side):
    if side == "top":
        return a
    if side == "bottom":
        return a[::-1]
    if side == "left":
        return a.T
    return a.T[::-1]


def _uncanonical(a, side):
    if side in ("top", "bottom"):
        return _canonical(a, side).copy()
    return a.T.copy() if side == "left" else a[::-1].T.copy()


def _bar(a, color):
    points = np.argwhere(a == color)
    if not len(points):
        return None
    rr, cc = points.T
    candidates = []
    for side, fixed, boundary, positions in (
        ("top", rr, 0, cc), ("bottom", rr, len(a) - 1, cc),
        ("left", cc, 0, rr), ("right", cc, a.shape[1] - 1, rr),
    ):
        if np.all(fixed == boundary):
            lo, hi = int(positions.min()), int(positions.max())
            if hi - lo + 1 == len(points):
                candidates.append((side, lo, len(points)))
    return candidates[0] if len(candidates) == 1 else None


def _frames(a, background):
    candidates = []
    for color in np.unique(a):
        color = int(color)
        if color == background:
            continue
        parts = _components(a == color)
        if len(parts) != 3:
            continue
        boxes = []
        for cells in parts:
            rr, cc = np.array(cells).T
            r, c = int(rr.min()), int(cc.min())
            h, w = int(rr.max()) - r + 1, int(cc.max()) - c + 1
            perimeter = {(y, x) for y in range(r, r + h) for x in range(c, c + w)
                         if y in (r, r + h - 1) or x in (c, c + w - 1)}
            if min(h, w) < 5 or set(cells) != perimeter:
                break
            boxes.append((r, c, h, w))
        if len(boxes) == 3:
            candidates.append((color, sorted(boxes)))
    if len(candidates) != 1:
        raise ValueError("expected one color forming exactly three hollow frames")
    return candidates[0]


def _roles(tile, background):
    colors, counts = np.unique(tile[tile != background], return_counts=True)
    if len(colors) != 2 or sorted(counts.tolist())[0] != 1 or max(counts) < 3:
        raise ValueError("a motif has one anchor and at least three body cells")
    anchor_color = int(colors[np.argmin(counts)])
    body_color = int(colors[np.argmax(counts)])
    anchor = tuple(map(int, np.argwhere(tile == anchor_color)[0]))
    return anchor_color, body_color, anchor


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if max(counts) <= a.size / 2:
        raise ValueError("background must be a strict majority of the input")
    background = int(colors[np.argmax(counts)])
    frame, boxes = _frames(a, background)
    interiors = [a[r + 1:r + h - 1, c + 1:c + w - 1].copy() for r, c, h, w in boxes]
    candidates = []
    for index, tile in enumerate(interiors):
        isolated_colors = {int(tile[cells[0]]) for cells in _components(tile != background)
                           if len({int(tile[p]) for p in cells}) == 1}
        for color in np.unique(tile):
            color = int(color)
            if color in (background, frame) or color not in isolated_colors:
                continue
            bar = _bar(tile, color)
            if bar is not None and np.count_nonzero(a == color) == np.count_nonzero(tile == color):
                candidates.append((index, color, bar))
    if len(candidates) != 1:
        raise ValueError("expected one exclusive boundary ruler bar")
    board_index, marker, (side, start, length) = candidates[0]
    board = interiors[board_index]
    references = [tile for i, tile in enumerate(interiors) if i != board_index]
    if references[0].shape != references[1].shape:
        raise ValueError("the two reference interiors must have equal dimensions")
    references.sort(key=lambda tile: int(np.count_nonzero(tile != background)))
    cue, blueprint = references
    anchor_color, body_color, ref_anchor = _roles(cue, background)
    if _roles(blueprint, background) != (anchor_color, body_color, ref_anchor):
        raise ValueError("reference states must preserve their color roles and anchor")
    if (np.count_nonzero(cue != background) >= np.count_nonzero(blueprint != background)
            or np.any((cue != background) & (cue != blueprint))):
        raise ValueError("reference completion must strictly extend the cue")
    if len(_components(cue != background)) != 1:
        raise ValueError("reference cue must be connected")
    cue_offsets = _offsets(cue != background, ref_anchor)
    full_offsets = _offsets(blueprint != background, ref_anchor)
    if sum(_transform(cue_offsets, k) == cue_offsets for k in range(8)) != 1:
        raise ValueError("the anchored cue must distinguish all eight rigid symmetries")
    motif_mask = (board != background) & (board != marker)
    objects = []
    occupied_potential = set()
    for cells in _components(motif_mask):
        local_counts = Counter(int(board[p]) for p in cells)
        singletons = [color for color, count in local_counts.items() if count == 1]
        if len(local_counts) != 2 or len(singletons) != 1:
            raise ValueError("each damaged motif has one anchor and one body role")
        ac = singletons[0]
        bc = next(color for color in local_counts if color != ac)
        anchor = next(p for p in cells if int(board[p]) == ac)
        if anchor[0] in (0, len(board) - 1) or anchor[1] in (0, board.shape[1] - 1):
            raise ValueError("anchors stay off the board boundary")
        target_offsets = {(r - anchor[0], c - anchor[1]) for r, c in cells}
        matches = [k for k in range(8) if _transform(cue_offsets, k) == target_offsets]
        if len(matches) != 1:
            raise ValueError("damaged motif must uniquely match the reference cue")
        symmetry = matches[0]
        full = _place(full_offsets, anchor, symmetry)
        if any(not (0 <= r < len(board) and 0 <= c < board.shape[1]) for r, c in full):
            raise ValueError("registered blueprint must fit inside the board")
        if any(board[p] == marker for p in full):
            raise ValueError("a potential blueprint may not cover the original bar")
        moat = {q for r, c in full for q in ((r, c), (r - 1, c), (r + 1, c),
                                            (r, c - 1), (r, c + 1))}
        if occupied_potential & moat:
            raise ValueError("potential blueprints must be disjoint with a one-cell moat")
        occupied_potential.update(full)
        distance = {"top": anchor[0], "bottom": len(board) - 1 - anchor[0],
                    "left": anchor[1], "right": board.shape[1] - 1 - anchor[1]}[side]
        objects.append(dict(anchor=anchor, anchor_color=ac, body_color=bc,
                            symmetry=symmetry, cells=set(cells), full=full,
                            distance=distance, active=distance % 6 == 4))
    if len(objects) < 2:
        raise ValueError("the board needs at least two damaged motifs")
    outside = np.ones(a.shape, dtype=bool)
    for r, c, h, w in boxes:
        outside[r:r + h, c:c + w] = False
    if np.any(a[outside] != background):
        raise ValueError("outside the three frames is plain background")
    return dict(input=a, background=background, frame=frame, boxes=boxes,
                board_index=board_index, board=board, marker=marker, side=side,
                start=start, length=length, cue=cue, blueprint=blueprint,
                ref_anchor=ref_anchor, cue_offsets=cue_offsets,
                full_offsets=full_offsets, objects=objects)


def _apply_ruler(board, background, marker, pulse, side, start, length):
    original = _canonical(board, side)
    out = original.copy()
    foreground = (original != background) & (original != marker)
    out[:, start:start + length] = background
    for d in range(len(out)):
        if d % 6 in (0, 2):
            out[d, start:start + length] = marker
        elif d % 6 == 4:
            out[d, start:start + length] = pulse
            out[d, foreground[d]] = pulse
    return _uncanonical(out, side)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read reference geometry, local palettes, registration, and phase from pixels."""
    info = _analyze(input_grid)
    restored = info["board"].copy()
    for obj in info["objects"]:
        if obj["active"]:
            for p in obj["full"]:
                restored[p] = obj["anchor_color"] if p == obj["anchor"] else obj["body_color"]
    return _apply_ruler(restored, info["background"], info["marker"], info["frame"],
                        info["side"], info["start"], info["length"])


def _pixel_oracle(info):
    """Check each output coordinate without the strip-painting implementation."""
    stamps = {}
    for obj in info["objects"]:
        if obj["active"]:
            for p in obj["full"]:
                stamps[p] = obj["anchor_color"] if p == obj["anchor"] else obj["body_color"]
    board = info["board"]
    out = np.empty_like(board)
    for r in range(len(board)):
        for c in range(board.shape[1]):
            d = {"top": r, "bottom": len(board) - 1 - r,
                 "left": c, "right": board.shape[1] - 1 - c}[info["side"]]
            lateral = c if info["side"] in ("top", "bottom") else r
            value = stamps.get((r, c), int(board[r, c]))
            if info["start"] <= lateral < info["start"] + info["length"]:
                value = (info["background"] if d % 2 else
                         info["frame"] if d % 6 == 4 else info["marker"])
            elif d % 6 == 4 and value not in (info["background"], info["marker"]):
                value = info["frame"]
            out[r, c] = value
    return out


def _grow(rng, mask, size):
    mask = mask.copy()
    h, w = mask.shape
    while mask.sum() < size:
        frontier = sorted({(rr, cc) for r, c in np.argwhere(mask)
                           for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                           if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]})
        if not frontier:
            break
        mask[frontier[int(rng.integers(len(frontier)))]] = True
    return mask


def _complete_mask(rng, cue):
    h, w = cue.shape
    rr, cc = np.indices(cue.shape)
    mode = int(rng.integers(5))
    mask = cue.copy()
    if mode == 0:
        return _grow(rng, mask, int(rng.integers(int(mask.sum()) + 3, h * w + 1)))
    r0, r1 = sorted(rng.choice(h + 1, 2, replace=False).tolist())
    c0, c1 = sorted(rng.choice(w + 1, 2, replace=False).tolist())
    if mode == 1:
        extra = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    elif mode == 2:
        extra = abs(rr - h // 2) + abs(cc - w // 2) <= int(rng.integers(2, max(h, w)))
    elif mode == 3:
        widths = rng.integers(1, w + 1, h)
        extra = abs(2 * cc - (w - 1)) < widths[:, None]
    else:
        extra = (((rr == r0) | (rr == r1 - 1)) & (cc >= c0) & (cc < c1)
                 | ((cc == c0) | (cc == c1 - 1)) & (rr >= r0) & (rr < r1))
    mask |= extra
    # Connect every proposed piece to the cue by varied orthogonal paths.
    root = tuple(map(int, np.argwhere(cue)[int(rng.integers(int(cue.sum())))]))
    for part in _components(mask):
        p = part[int(rng.integers(len(part)))]
        if rng.integers(2):
            mask[min(root[0], p[0]):max(root[0], p[0]) + 1, root[1]] = True
            mask[p[0], min(root[1], p[1]):max(root[1], p[1]) + 1] = True
        else:
            mask[root[0], min(root[1], p[1]):max(root[1], p[1]) + 1] = True
            mask[min(root[0], p[0]):max(root[0], p[0]) + 1, p[1]] = True
    return _grow(rng, mask, max(8, int(cue.sum()) + 3))


def _templates(rng, large=False):
    h, w = [int(rng.integers(5 if large else 4, 8 if large else 7)) for _ in range(2)]
    for _ in range(100):
        anchor = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
        cue = np.zeros((h, w), dtype=bool)
        cue[anchor] = True
        cue = _grow(rng, cue, int(rng.integers(4, 8)))
        offsets = _offsets(cue, anchor)
        if sum(_transform(offsets, k) == offsets for k in range(8)) == 1:
            break
    else:
        # Asymmetric anchored L, valid in every supported template size.
        anchor = (1, 1)
        cue[:] = False
        cue[1, 1:4] = True
        cue[2, 1] = True
    full1 = _complete_mask(rng, cue)
    for _ in range(80):
        full2 = _complete_mask(rng, cue)
        if np.count_nonzero(full1 != full2) >= 3:
            return cue, (full1, full2), anchor
    # A deterministic different extension, retaining the same meaningful key.
    full2 = np.ones_like(cue)
    if np.count_nonzero(full1 != full2) < 3:
        full2 = _grow(rng, cue, int(cue.sum()) + 3)
    return cue, (full1, full2), anchor


def _layout(rng, shape, teaching):
    ph, pw = shape[0] + 2, shape[1] + 2
    orientation = int(rng.integers(2))
    gap = int(rng.integers(1, 4))
    offset = int(rng.integers(3))
    between = int(rng.integers(1, 3))
    if orientation == 0:
        lh, lw = ph + offset, 2 * pw + gap
        depth = int(rng.integers(13, 30 - lh - between - 2 + 1))
        span = int(rng.integers(17 if teaching else 13, 28))
        min_h, min_w = lh + between + depth + 2, max(lw, span + 2)
    else:
        lh, lw = 2 * ph + gap, pw + offset
        depth = int(rng.integers(17 if teaching else 13, 28))
        span = int(rng.integers(13, 30 - lw - between - 2 + 1))
        min_h, min_w = max(lh, depth + 2), lw + between + span + 2
    height, width = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    reverse = bool(rng.integers(2))
    if orientation == 0:
        top = int(rng.integers(height - min_h + 1))
        legend_r = top if not reverse else top + depth + 2 + between
        board_r = top + lh + between if not reverse else top
        legend_c = int(rng.integers(width - lw + 1))
        board_c = int(rng.integers(width - span - 2 + 1))
        refs = [(legend_r, legend_c, ph, pw),
                (legend_r + offset, legend_c + pw + gap, ph, pw)]
    else:
        left = int(rng.integers(width - min_w + 1))
        legend_c = left if not reverse else left + span + 2 + between
        board_c = left + lw + between if not reverse else left
        legend_r = int(rng.integers(height - lh + 1))
        board_r = int(rng.integers(height - depth - 2 + 1))
        refs = [(legend_r, legend_c, ph, pw),
                (legend_r + ph + gap, legend_c + offset, ph, pw)]
    return (height, width), refs, (board_r, board_c, depth + 2, span + 2)


def _targets(rng, cue_offsets, potential_offsets, shape, palette, teaching):
    depth, span = shape
    objects, used = [], set()
    desired = int(rng.integers(3 if teaching else 2, 7))
    for index in range(desired):
        for _ in range(250):
            symmetry = int(rng.integers(8))
            if index == 0:
                rows = [r for r in range(1, depth - 1) if r % 6 == 4]
            elif index == 1:
                rows = [r for r in range(1, depth - 1) if r % 6 in (3, 5)]
            else:
                rows = list(range(1, depth - 1))
            anchor = (int(rng.choice(rows)), int(rng.integers(1, span - 1)))
            full = _place(potential_offsets, anchor, symmetry)
            if any(not (0 <= r < depth and 0 <= c < span) for r, c in full):
                continue
            moat = {q for r, c in full for q in ((r, c), (r - 1, c), (r + 1, c),
                                                (r, c - 1), (r, c + 1))}
            if used & moat:
                continue
            cells = _place(cue_offsets, anchor, symmetry)
            if any(r == 0 for r, c in full):
                continue  # keep the seed edge free to sample any bar later
            ac, bc = rng.choice(palette, 2, replace=False).tolist()
            objects.append(dict(anchor=anchor, anchor_color=int(ac), body_color=int(bc),
                                symmetry=symmetry, cells=cells, potential=full,
                                active=anchor[0] % 6 == 4))
            used.update(full)
            break
        else:
            break
    if len(objects) < 2:
        return None
    board = np.full(shape, -1, dtype=np.int8)
    for obj in objects:
        for p in obj["cells"]:
            board[p] = obj["anchor_color"] if p == obj["anchor"] else obj["body_color"]
    if teaching and not any(not o["active"] and any(r % 6 == 4 for r, c in o["cells"])
                            for o in objects):
        return None
    return board, objects


def _outline(a, box, color):
    r, c, h, w = box
    a[r, c:c + w] = a[r + h - 1, c:c + w] = color
    a[r:r + h, c] = a[r:r + h, c + w - 1] = color


def _construct(rng, teaching=False, large=False, rotation=None):
    """Return reference twins; fields and all pixels except the dense tile agree."""
    for _ in range(150):
        cue, fulls, ref_anchor = _templates(rng, large)
        canvas_shape, refs, board_box = _layout(rng, cue.shape, teaching)
        depth, span = board_box[2] - 2, board_box[3] - 2
        colors = rng.permutation(10).tolist()
        background, frame, marker, ref_ac, ref_bc = colors[:5]
        target_palette = colors[3:]
        cue_offsets = _offsets(cue, ref_anchor)
        potential = _offsets(fulls[0] | fulls[1], ref_anchor)
        result = _targets(rng, cue_offsets, potential, (depth, span), target_palette, teaching)
        if result is None:
            continue
        board, objects = result
        board[board == -1] = background
        # Objects are sampled before choosing an informative corridor.
        possible = []
        for length in range(1, min(6, span - 1)):
            for start in range(1, span - length):
                if np.any(board[1, start:start + length] != background):
                    continue  # original boundary bar is an isolated component
                strip = {(r, c) for r in range(depth) for c in range(start, start + length)}
                fg = set().union(*(o["cells"] for o in objects))
                if teaching:
                    if not any(r % 2 for r, c in fg & strip):
                        continue
                    if not any(r > 0 and r % 6 in (0, 2) for r, c in fg & strip):
                        continue
                    if not any(o["active"] and not (o["cells"] & strip) for o in objects):
                        continue
                possible.append((start, length))
        if not possible:
            continue
        start, length = possible[int(rng.integers(len(possible)))]
        board[0, start:start + length] = marker
        ordered_refs = refs if rng.integers(2) else refs[::-1]
        rotation_value = int(rng.integers(4)) if rotation is None else rotation
        pairs = []
        for full in fulls:
            canvas = np.full(canvas_shape, background, dtype=np.int8)
            for box, mask in zip(ordered_refs, (cue, full)):
                _outline(canvas, box, frame)
                r, c, h, w = box
                tile = np.where(mask, ref_bc, background).astype(np.int8)
                tile[ref_anchor] = ref_ac
                canvas[r + 1:r + h - 1, c + 1:c + w - 1] = tile
            _outline(canvas, board_box, frame)
            r, c, h, w = board_box
            canvas[r + 1:r + h - 1, c + 1:c + w - 1] = board
            # Labels use known stamps; solve independently rediscovers them.
            restored = board.copy()
            full_offsets = _offsets(full, ref_anchor)
            for obj in objects:
                if obj["active"]:
                    for p in _place(full_offsets, obj["anchor"], obj["symmetry"]):
                        restored[p] = obj["anchor_color"] if p == obj["anchor"] else obj["body_color"]
            output = _apply_ruler(restored, background, marker, frame, "top", start, length)
            pairs.append(dict(input=np.rot90(canvas, rotation_value).copy(),
                              output=np.rot90(output, rotation_value).copy()))
        if np.array_equal(pairs[0]["output"], pairs[1]["output"]):
            continue
        try:
            infos = [_analyze(p["input"]) for p in pairs]
        except ValueError:
            continue
        if teaching and not all(_evidence(info)["growth"] and _evidence(info)["inactive_crossing"]
                                and _evidence(info)["remote_growth"] for info in infos):
            continue
        return pairs
    raise RuntimeError("bounded construction exhausted; no arbitrary fallback rule")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    base_rotation = int(rng.integers(4))
    puzzle = {"train": _construct(rng, teaching=True, rotation=base_rotation), "test": []}
    seen = {_key(p["input"]) for p in puzzle["train"]}
    for split, count in (("train", nt - 2), ("test", ns)):
        for index in range(count):
            for _ in range(20):
                # Tests rotate the scene away from the twin demonstrations and
                # sample larger reference blueprints. Nothing latent changes.
                rotation = ((base_rotation + 1 + index % 3) % 4 if split == "test"
                            else int(rng.integers(4)))
                pair = _construct(rng, large=split == "test", rotation=rotation)[0]
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate rejection exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info):
    board = _canonical(info["board"], info["side"])
    fg = (board != info["background"]) & (board != info["marker"])
    strip = np.zeros_like(fg)
    strip[:, info["start"]:info["start"] + info["length"]] = True
    rows = np.arange(len(board))[:, None]
    growth = remote = inactive_crossing = restored_strip = False
    for obj in info["objects"]:
        distances = lambda p: {"top": p[0], "bottom": len(info["board"]) - 1 - p[0],
                               "left": p[1], "right": info["board"].shape[1] - 1 - p[1]}[info["side"]]
        in_strip = lambda p: info["start"] <= (p[1] if info["side"] in ("top", "bottom")
                                               else p[0]) < info["start"] + info["length"]
        if obj["active"]:
            additions = obj["full"] - obj["cells"]
            visible = any(not in_strip(p) and distances(p) % 6 != 4 for p in additions)
            growth |= visible
            remote |= visible and not any(in_strip(p) for p in obj["cells"])
            restored_strip |= any(in_strip(p) for p in additions)
        else:
            inactive_crossing |= any(distances(p) % 6 == 4 and not in_strip(p)
                                     for p in obj["cells"]) and any(
                                         not in_strip(p) and distances(p) % 6 != 4
                                         for p in obj["full"] - obj["cells"])
    return dict(growth=growth, remote_growth=remote, inactive_crossing=inactive_crossing,
                restored_strip=restored_strip,
                clearing=bool(np.any(fg & strip & (rows % 2 == 1))),
                ordinary_overwrite=bool(np.any(fg & strip & (rows > 0) & np.isin(rows % 6, (0, 2)))),
                remote_highlight=bool(np.any(fg & ~strip & (rows % 6 == 4))),
                outside_preserved=bool(np.any(fg & ~strip & (rows % 6 != 4))))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ns, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("a pair must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            expected = _pixel_oracle(info)
            if not np.array_equal(out, expected) or not np.array_equal(solve(inp), expected):
                raise ValueError("output disagrees with registered, phase-gated restoration")
            if out.shape != info["board"].shape:
                raise ValueError("output must be the board interior")
            if split == "train":
                training.append(_evidence(info))
    for witness in ("growth", "remote_growth", "inactive_crossing", "clearing",
                    "ordinary_overwrite", "remote_highlight", "outside_preserved"):
        if not any(e[witness] for e in training):
            raise ValueError(f"training lacks {witness} evidence")
    return None


def self_test():
    """Bounded semantic diagnostics and observable diversity on exactly 42 seeds."""
    shapes, board_shapes, cues, blueprints, sides, symmetries, spans, counts = [set() for _ in range(8)]
    structures, role_values = set(), set()
    overlap, boundary, pairs_checked = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        twins = puzzle["train"][:2]
        x, y = [_analyze(p["input"]) for p in twins]
        assert np.array_equal(x["board"], y["board"])
        assert np.array_equal(x["cue"], y["cue"])
        assert not np.array_equal(twins[0]["output"], twins[1]["output"])
        # Drop B's reconstruction, or drop A's anchor gate while retaining
        # everything else. These are causal diagnostics, not just shape hashes.
        for info, pair in zip((x, y), twins):
            no_blueprint = _apply_ruler(info["board"], info["background"], info["marker"],
                                       info["frame"], info["side"], info["start"], info["length"])
            every_object = info["board"].copy()
            touched_object = info["board"].copy()
            for obj in info["objects"]:
                touched = any(({"top": r, "bottom": len(every_object) - 1 - r,
                                "left": c, "right": every_object.shape[1] - 1 - c}[info["side"]]
                               % 6 == 4) for r, c in obj["cells"])
                for p in obj["full"]:
                    value = obj["anchor_color"] if p == obj["anchor"] else obj["body_color"]
                    every_object[p] = value
                    if touched:
                        touched_object[p] = value
            no_gate = _apply_ruler(every_object, info["background"], info["marker"],
                                   info["frame"], info["side"], info["start"], info["length"])
            wrong_gate = _apply_ruler(touched_object, info["background"], info["marker"],
                                      info["frame"], info["side"], info["start"], info["length"])
            assert not np.array_equal(no_blueprint, pair["output"])
            assert not np.array_equal(no_gate, pair["output"])
            assert not np.array_equal(wrong_gate, pair["output"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                pairs_checked += 1
                shapes.add(pair["input"].shape)
                board_shapes.add(info["board"].shape)
                cues.add((info["cue"].shape, (info["cue"] != info["background"]).tobytes()))
                blueprints.add((info["blueprint"].shape,
                                (info["blueprint"] != info["background"]).tobytes()))
                sides.add(info["side"])
                symmetries.update(o["symmetry"] for o in info["objects"])
                spans.add(info["length"])
                counts.add(len(info["objects"]))
                roles = {}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                role_values.update((info["background"], info["frame"], info["marker"]))
                overlap += _evidence(info)["restored_strip"]
                h, w = info["board"].shape
                boundary += any(r in (0, h - 1) or c in (0, w - 1)
                                for o in info["objects"] for r, c in o["full"])
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutated its input"
    assert len(shapes) >= 30 and len(board_shapes) >= 50
    assert len(cues) >= 100 and len(blueprints) >= 150 and len(structures) >= 180
    assert sides == set(_SIDES) and symmetries == set(range(8))
    assert spans == set(range(1, 6)) and len(counts) >= 3
    assert role_values == set(range(10)) and overlap > 0 and boundary > 0
    a, b = generate_puzzle(99, 2, 1), generate_puzzle(99, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ("input", "output"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(9, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # Rotation covariance, including extraction and registration, on a real scene.
    pair = a["train"][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
    assert np.array_equal(solve(pair["input"][:, ::-1]), pair["output"][:, ::-1])
    # Swapping reference slots cannot select a different operation or direction.
    info = _analyze(pair["input"])
    boxes = [b for i, b in enumerate(info["boxes"]) if i != info["board_index"]]
    swapped = pair["input"].copy()
    tiles = [swapped[r + 1:r + h - 1, c + 1:c + w - 1].copy() for r, c, h, w in boxes]
    for box, tile in zip(boxes, tiles[::-1]):
        r, c, h, w = box
        swapped[r + 1:r + h - 1, c + 1:c + w - 1] = tile
    assert np.array_equal(solve(swapped), pair["output"])
    # A hand-built family member beyond the sampler's size/span distribution:
    # depth six, only one special tick, an eight-cell bar, and boundary growth.
    cue = np.zeros((4, 5), dtype=np.int8)
    cue[1, 1:4] = 2
    cue[2, 1] = 2
    cue[1, 1] = 3
    full = np.zeros_like(cue)
    full[:3, :4] = 2
    full[1, 1] = 3
    board = np.zeros((6, 24), dtype=np.int8)
    board[0, 5:13] = 1
    for (r, c), ac, bc in (((4, 4), 6, 7), ((2, 17), 8, 9)):
        board[r, c] = ac
        board[r, c + 1:c + 3] = bc
        board[r + 1, c] = bc
    canvas = np.zeros((16, 26), dtype=np.int8)
    for box, tile in zip(((0, 0, 6, 7), (0, 9, 6, 7), (8, 0, 8, 26)), (cue, full, board)):
        _outline(canvas, box, 5)
        r, c, h, w = box
        canvas[r + 1:r + h - 1, c + 1:c + w - 1] = tile
    expected = board.copy()
    expected[3:6, 3:7] = 7
    expected[:, 5:13] = 0
    expected[0, 5:13] = expected[2, 5:13] = 1
    expected[4, 3:13] = 5
    for k in range(4):
        assert np.array_equal(solve(np.rot90(canvas, k)), np.rot90(expected, k))
    return dict(seeds=42, pairs=pairs_checked, input_shapes=len(shapes), board_shapes=len(board_shapes),
                cue_structures=len(cues), blueprint_structures=len(blueprints),
                normalized_scenes=len(structures), sides=sorted(sides),
                rigid_symmetries=len(symmetries), ruler_lengths=sorted(spans),
                object_counts=sorted(counts), restoration_strip_intersections=overlap,
                potential_boundary_contacts=boundary,
                semantic_verification="generator-owned checks only")
