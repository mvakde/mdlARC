"""Torn stencil sheets: jointly fit pieces and complete symbolic stamps.

The dictionary is visible on the pinned piece in EVERY input. The solver never
receives a seed, training examples, construction records, or expected output.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return g.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)
             if (a or b) and (diagonal or abs(a) + abs(b) == 1)]
    while pending:
        p = min(pending)
        pending.remove(p)
        cells = [p]
        for r, c in cells:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r + dr, c), (r, c + dc), (r + dr, c + dc))


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= g.size // 2:
        raise ValueError("background must occupy more than half the input")
    markers = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        q = np.argwhere(g == color)
        r, c = q.min(0)
        y, x = q.max(0)
        if y - r != 1 or x - c != 1:
            continue
        missing = ({(int(r), int(c)), (int(r), int(x)),
                    (int(y), int(c)), (int(y), int(x))}
                   - set(map(tuple, q.tolist())))
        point = missing.pop()
        if g[point] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL",
                  (0, 0): "BR"}[(point[0] - r, point[1] - c)]
        markers.append((color, point, corner))
    if len(markers) != 1:
        raise ValueError("need one distinct external three-cell L")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.asarray(cells)
        r, c = q.min(0)
        y, x = q.max(0) + 1
        m = np.full((y - r, x - c), -1, dtype=np.int8)
        for a, b in cells:
            m[a - r, b - c] = g[a, b]
        if point in cells:
            anchor = len(pieces)
        pieces.append((m, (int(r), int(c))))
    if anchor is None or not 2 <= len(pieces) <= 6:
        raise ValueError("need 2..6 pieces and a pinned piece")
    a, pos = pieces[anchor]
    expected = (pos[0] + (a.shape[0] - 1 if corner[0] == "B" else 0),
                pos[1] + (a.shape[1] - 1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("L must mark a bounding-box corner of its piece")
    payload = np.concatenate([m[m >= 0] for m, _ in pieces])
    carrier = int(np.bincount(payload, minlength=10).argmax())
    if np.count_nonzero(payload == carrier) <= payload.size // 2:
        raise ValueError("the sheet carrier must be a strict majority of payload")
    book = _read_book(a, carrier)
    if set(book) | {v[0] for v in book.values()} | {carrier, bg, marker} != set(
            map(int, np.unique(g))):
        raise ValueError("unexplained color or colliding palette roles")
    return bg, carrier, marker, point, corner, pieces, anchor, book


def _read_book(anchor, carrier):
    """Enclosed, two-color reference components identify code and ink roles."""
    book = {}
    for cells in _components((anchor >= 0) & (anchor != carrier), True):
        # A swatch has a complete one-cell carrier moat within the pinned piece.
        if any(not (0 < r < anchor.shape[0] - 1 and 0 < c < anchor.shape[1] - 1)
               or np.any(anchor[r - 1:r + 2, c - 1:c + 2] < 0)
               for r, c in cells):
            continue
        colors, counts = np.unique([anchor[p] for p in cells], return_counts=True)
        if len(colors) != 2 or np.count_nonzero(counts == 1) != 1:
            continue
        code = int(colors[counts == 1][0])
        ink = int(colors[counts != 1][0])
        origin = (min(r for r, _ in cells), min(c for _, c in cells))
        if anchor[origin] != code:
            continue
        h = max(r for r, _ in cells) - origin[0] + 1
        w = max(c for _, c in cells) - origin[1] + 1
        if not (3 <= h <= 5 and 3 <= w <= 5):
            continue
        mask = anchor[origin[0]:origin[0] + h, origin[1]:origin[1] + w] == ink
        if (mask[0, 0] or not mask[0].any() or not mask[-1].any()
                or not mask[:, 0].any() or not mask[:, -1].any()
                or len(_components(mask, True)) != 1):
            continue
        if code in book and (ink != book[code][0] or not np.array_equal(mask, book[code][1])):
            raise ValueError("conflicting swatches")
        book[code] = (ink, mask.copy())
    if not 1 <= len(book) <= 2:
        raise ValueError("pinned piece must contain 1..2 enclosed full swatches")
    codes = set(book)
    inks = {ink for ink, _ in book.values()}
    if len(inks) != len(book) or codes & inks or carrier in codes | inks:
        raise ValueError("code and ink roles must be distinct")
    return book


def _material(sheet, carrier, book):
    """Check an assembly's observations, then reconstruct only live stamps.

    Missing ink is allowed in a live component, never extra or wrong-color ink.
    Bare and lower-right-attached material must be complete, uncut swatches.
    """
    codes = set(book)
    inverse = {ink: code for code, (ink, _) in book.items()}
    output = np.full(sheet.shape, carrier, dtype=np.int8)
    used = np.zeros(sheet.shape, dtype=bool)
    live, bare, rejected = [], [], []
    for cells in _components(sheet != carrier, True):
        points = [p for p in cells if int(sheet[p]) in codes]
        origin = (min(r for r, _ in cells), min(c for _, c in cells))
        if len(points) == 1 and points[0] == origin:
            r, c = origin
            code = int(sheet[origin])
            ink, mask = book[code]
            h, w = min(mask.shape[0], sheet.shape[0] - r), min(mask.shape[1], sheet.shape[1] - c)
            if used[r:r + h, c:c + w].any():
                return None
            window = sheet[r:r + h, c:c + w]
            allowed = np.full((h, w), carrier, dtype=np.int8)
            allowed[mask[:h, :w]] = ink
            allowed[0, 0] = code
            if np.any((window != carrier) & (window != allowed)):
                return None
            if any(sheet[p] != ink or p[0] - r >= h or p[1] - c >= w
                   or not mask[p[0] - r, p[1] - c] for p in cells if p != origin):
                return None
            used[r:r + h, c:c + w] = True
            output[r:r + h, c:c + w][mask[:h, :w]] = ink
            observed = sum(int(sheet[p] == ink) for p in cells)
            live.append((r, c, code, observed, int(mask[:h, :w].sum())))
        else:
            ink_colors = {int(sheet[p]) for p in cells if int(sheet[p]) in inverse}
            if len(ink_colors) != 1 or len(points) > 1:
                return None
            ink = ink_colors.pop()
            code = inverse[ink]
            mask = book[code][1]
            r, c = origin
            expected = {(r + int(y), c + int(x)): ink for y, x in np.argwhere(mask)}
            if points:
                y, x = max(map(tuple, np.argwhere(mask).tolist()))
                if points[0] != (r + y + 1, c + x + 1) or int(sheet[points[0]]) != code:
                    return None
                expected[points[0]] = code
                rejected.append((r, c, code))
            else:
                bare.append((r, c, code))
            if {p: int(sheet[p]) for p in cells} != expected:
                return None
    return output, live, bare, rejected


def _packings(pieces, anchor, point, corner, canvas):
    """All translation-only rectangular packings; incomplete search raises."""
    area = sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces)
    nodes = 0
    for h in range(1, canvas[0] + 1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if w > canvas[1] or oy < 0 or ox < 0 or oy + h > canvas[0] or ox + w > canvas[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        configs = []
        for m, _ in pieces:
            coords = list(map(tuple, np.argwhere(m >= 0).tolist()))
            bits = sum(1 << (y * w + x) for y, x in coords)
            configs.append((m.shape, coords[0][1], bits))
        ar, ac = pieces[anchor][1][0] - oy, pieces[anchor][1][1] - ox
        ah, aw = pieces[anchor][0].shape
        if ar < 0 or ac < 0 or ar + ah > h or ac + aw > w:
            continue
        positions = [None] * len(pieces)
        positions[anchor] = (ar, ac)
        full = (1 << area) - 1
        initial = configs[anchor][2] << (ar * w + ac)

        def visit(occupied, remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 100000:
                raise ValueError("packing search exceeds the supported family budget")
            if not remaining:
                yield (h, w), (oy, ox), tuple(positions)
                return
            free = full ^ occupied
            index = (free & -free).bit_length() - 1
            r, c = divmod(index, w)
            for i in remaining:
                (ph, pw), first_col, bits = configs[i]
                x = c - first_col
                if x < 0 or r + ph > h or x + pw > w:
                    continue
                placed = bits << (r * w + x)
                if not placed & occupied:
                    positions[i] = (r, x)
                    yield from visit(occupied | placed, tuple(j for j in remaining if j != i))
        yield from visit(initial, tuple(i for i in range(len(pieces)) if i != anchor))


def _assemble(pieces, positions, shape):
    sheet = np.full(shape, -1, dtype=np.int8)
    labels = np.full(shape, -1, dtype=np.int8)
    for i, ((m, _), (r, c)) in enumerate(zip(pieces, positions)):
        present = m >= 0
        sheet[r:r + m.shape[0], c:c + m.shape[1]][present] = m[present]
        labels[r:r + m.shape[0], c:c + m.shape[1]][present] = i
    return sheet, labels


def _infer(g):
    bg, carrier, marker, point, corner, pieces, anchor, book = _extract(g)
    winner = None
    geometry_count, valid_count = 0, 0
    geometric_sheets = set()
    for shape, origin, positions in _packings(pieces, anchor, point, corner, g.shape):
        geometry_count += 1
        sheet, labels = _assemble(pieces, positions, shape)
        geometric_sheets.add((shape, origin, sheet.tobytes()))
        material = _material(sheet, carrier, book)
        if material is None:
            continue
        valid_count += 1
        rendered, live, bare, rejected = material
        key = (shape, origin, sheet.tobytes())
        if winner is not None and winner[0] != key:
            raise ValueError("ambiguous assembled instruction sheets")
        winner = (key, shape, origin, positions, sheet, labels, rendered, live, bare, rejected)
    if winner is None:
        raise ValueError("no packing agrees with the visible stencil dictionary")
    _, shape, (r, c), positions, sheet, labels, rendered, live, bare, rejected = winner
    out = np.full(g.shape, bg, dtype=np.int8)
    out[r:r + shape[0], c:c + shape[1]] = rendered
    absolute = tuple((y + r, x + c) for y, x in positions)
    return out, dict(bg=bg, carrier=carrier, marker=marker, point=point, corner=corner,
                     pieces=pieces, anchor=anchor, book=book, shape=shape, origin=(r, c),
                     positions=absolute, sheet=sheet, labels=labels, live=live, bare=bare,
                     rejected=rejected, geometry_count=geometry_count,
                     distinct_geometric_sheets=len(geometric_sheets), valid_count=valid_count)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _infer(_grid(input_grid))[0]


def _render_records(shape, carrier, records, book):
    """Construction labels come from chosen live requests, not segmentation."""
    out = np.full(shape, carrier, dtype=np.int8)
    for r, c, code in records:
        ink, mask = book[code]
        h, w = min(mask.shape[0], shape[0] - r), min(mask.shape[1], shape[1] - c)
        out[r:r + h, c:c + w][mask[:h, :w]] = ink
    return out


def _features(info):
    edges = []
    cross, partial_cross, max_span = 0, 0, 1
    live = {(r, c): (code, n, full) for r, c, code, n, full in info["live"]}
    for cells in _components(info["sheet"] != info["carrier"], True):
        origin = (min(r for r, _ in cells), min(c for _, c in cells))
        if origin not in live:
            continue
        touched = {int(info["labels"][p]) for p in cells}
        if len(touched) > 1:
            edges.append(touched)
            cross += 1
            _, n, full = live[origin]
            partial_cross += int(n < full)
            max_span = max(max_span, len(touched))
    reached = {info["anchor"]}
    while True:
        old = reached.copy()
        for edge in edges:
            if edge & reached:
                reached |= edge
        if old == reached:
            break
    boundaries = set()
    for r, c, code, _, _ in info["live"]:
        mask = info["book"][code][1]
        bottom, right = r + mask.shape[0] > info["shape"][0], c + mask.shape[1] > info["shape"][1]
        if bottom:
            boundaries.add("bottom")
        if right:
            boundaries.add("right")
        if bottom and right:
            boundaries.add("corner")
    all_codes = [(int(r), int(c), int(info["sheet"][r, c]))
                 for r, c in np.argwhere(np.isin(info["sheet"], list(info["book"])))]
    shortcut = _render_records(info["shape"], info["carrier"], all_codes, info["book"])
    correct = _render_records(info["shape"], info["carrier"],
                              [(r, c, code) for r, c, code, _, _ in info["live"]], info["book"])
    return dict(cross=cross, partial_cross=partial_cross, max_span=max_span,
                joined=len(reached) == len(info["pieces"]), boundaries=boundaries,
                all_codes_fails=not np.array_equal(shortcut, correct),
                stepped=any(np.any(m < 0) for m, _ in info["pieces"]))


def _sample_mask(rng, h, w):
    # Blank upper-left anchor, tight bounds, and eight-connected ink. Start
    # with a bent outline, then add an independently sampled connected interior.
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(5))
    if kind == 0:
        mask[0, 1:] = mask[-1, :] = True
        mask[1:, 0] = mask[:, -1] = True
    elif kind == 1:
        mask[0, 1:] = mask[1:, 0] = mask[-1, :] = True
    elif kind == 2:
        mask[0, 1:] = mask[:, -1] = mask[-1, :] = True
        mask[1, 0] = True
    elif kind == 3:
        r, c = 0, 1
        mask[r, c] = mask[1, 0] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
    else:
        mask[:] = True
        mask[0, 0] = False
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if rng.random() < .5:
                    mask[r, c] = False
    for _ in range(int(rng.integers(0, h * w // 3 + 1))):
        y, x = int(rng.integers(h)), int(rng.integers(w))
        if (y or x) and any(mask[a, b] for a in range(max(0, y - 1), min(h, y + 2))
                           for b in range(max(0, x - 1), min(w, x + 2))):
            mask[y, x] = True
    return mask


def _prototype_layout(book, stacked):
    places = []
    offset = 2
    for code, (_, mask) in book.items():
        places.append((offset, 2, code) if stacked else (2, offset, code))
        offset += mask.shape[0 if stacked else 1] + 2
    h = offset if stacked else max(m.shape[0] for _, m in book.values()) + 4
    w = max(m.shape[1] for _, m in book.values()) + 4 if stacked else offset
    return (h, w), places


def _partition(rng, anchor_shape, corner, count, stepped, mode, spacious=False):
    ah, aw = anchor_shape
    low, high = (6, 10) if spacious else (4, 8)
    eh, ew = int(rng.integers(low, high)), int(rng.integers(low, high))
    if mode == "horizontal":
        aw += ew
        ew = 0
    elif mode == "vertical":
        ah += eh
        eh = 0
    h, w = ah + eh, aw + ew
    labels = np.zeros((h, w), dtype=np.int8)
    rects = []
    if eh and ew:
        if rng.integers(2):
            rects = [(0, aw, h, ew), (ah, 0, eh, aw)]
        else:
            rects = [(0, aw, ah, ew), (ah, 0, eh, w)]
    elif eh:
        rects = [(ah, 0, eh, w)]
    else:
        rects = [(0, aw, h, ew)]
    while len(rects) + 1 < count:
        options = [(i, d) for i, rect in enumerate(rects) for d in (0, 1)
                   if rect[2 + d] >= 4]
        if not options:
            return None
        i, d = options[int(rng.integers(len(options)))]
        r, c, ph, pw = rects.pop(i)
        cut = int(rng.integers(2, (ph if d == 0 else pw) - 1))
        rects.extend([(r, c, cut, pw), (r + cut, c, ph - cut, pw)] if d == 0
                     else [(r, c, ph, cut), (r, c + cut, ph, pw - cut)])
    for i, (r, c, ph, pw) in enumerate(rects, 1):
        labels[r:r + ph, c:c + pw] = i
    if stepped:
        choices = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                donor = int(labels[r, c])
                if donor == 0:
                    continue  # The dictionary piece retains its rectangular moat.
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    recipient = int(labels[r + dr, c + dc])
                    if recipient and recipient != donor:
                        choices.append((r, c, recipient))
        rng.shuffle(choices)
        done = 0
        for r, c, recipient in choices:
            donor = int(labels[r, c])
            if donor == recipient:
                continue
            before = labels.copy()
            labels[r, c] = recipient
            if all(len(_components(labels == i)) == 1 for i in (donor, recipient)):
                done += 1
                if done >= int(rng.integers(1, 4)):
                    break
            else:
                labels = before
        if not done:
            return None
    if corner[0] == "B":
        labels = labels[::-1].copy()
    if corner[1] == "R":
        labels = labels[:, ::-1].copy()
    return labels


def _place_record(raw, occupied, r, c, code, book, carrier, kind, rng, partial=False):
    ink, mask = book[code]
    h, w = mask.shape
    H, W = raw.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    hh, ww = min(h, H - r), min(w, W - c)
    area = np.zeros(raw.shape, dtype=bool)
    area[r:r + hh, c:c + ww] = True
    if np.any(area & occupied):
        return None
    # Reserve a halo as well: unrelated raw components may not touch diagonally.
    if np.any(raw[max(0, r - 1):min(H, r + hh + 1), max(0, c - 1):min(W, c + ww + 1)] != carrier):
        return None
    observation = mask[:hh, :ww].copy()
    if kind == "live" and (hh != h or ww != w):
        observation[:] = False  # Boundary requests start as isolated code points.
    if kind != "live" and (hh != h or ww != w):
        return None
    if partial:
        for index in rng.permutation(int(observation.sum())):
            coords = np.argwhere(observation)
            if not len(coords):
                break
            y, x = coords[int(index) % len(coords)]
            old = observation.copy()
            observation[y, x] = False
            attached = observation.copy()
            attached[0, 0] = True
            if len(_components(attached, True)) != 1:
                observation = old
            if int(observation.sum()) <= max(2, int(mask.sum()) // 2):
                break
    block = np.full((hh, ww), carrier, dtype=np.int8)
    block[observation] = ink
    if kind == "live":
        block[0, 0] = code
    elif kind == "attached":
        y, x = max(map(tuple, np.argwhere(mask).tolist()))
        pr, pc = r + y + 1, c + x + 1
        if pr >= H or pc >= W or raw[pr, pc] != carrier or occupied[pr, pc]:
            return None
        if not mask[:H - pr, :W - pc].any():
            return None  # The incorrectly attached code must change an all-code shortcut.
        area[pr, pc] = True
        if np.any(raw[max(0, pr - 1):min(H, pr + 2), max(0, pc - 1):min(W, pc + 2)] != carrier):
            return None
        raw[pr, pc] = code
    raw[r:r + hh, c:c + ww] = block
    occupied |= area
    return observation


def _cut(labels, raw):
    result = []
    for i in range(int(labels.max()) + 1):
        q = np.argwhere(labels == i)
        r, c = q.min(0)
        y, x = q.max(0) + 1
        m = np.where(labels[r:y, c:x] == i, raw[r:y, c:x], -1).astype(np.int8)
        result.append((m, (int(r), int(c))))
    return result


def _scatter(rng, pieces, canvas, final, anchor, marks):
    def separated(a, b):
        r, c, h, w = a
        y, x, ph, pw = b
        return r + h < y or y + ph < r or c + w < x or x + pw < c
    for _ in range(12):
        positions = [None] * len(pieces)
        positions[anchor] = final[anchor]
        placed = [(*final[anchor], *pieces[anchor][0].shape)]
        order = sorted((i for i in range(len(pieces)) if i != anchor),
                       key=lambda i: -pieces[i][0].size)
        for i in order:
            h, w = pieces[i][0].shape
            options = [(r, c) for r in range(canvas[0] - h + 1)
                       for c in range(canvas[1] - w + 1)]
            rng.shuffle(options)
            for r, c in options:
                if any(r - 1 <= y <= r + h and c - 1 <= x <= c + w for y, x in marks):
                    continue
                if all(separated((r, c, h, w), p) for p in placed):
                    positions[i] = (r, c)
                    placed.append((r, c, h, w))
                    break
            if positions[i] is None:
                break
        if all(p is not None for p in positions):
            return positions
    return None


def _make_pair(rng, book, roles, corner, count, stepped, partial, artifacts, clipped):
    bg, carrier, marker = roles
    artifact_kinds = tuple(kind for bit, kind in ((1, "bare"), (2, "attached")) if artifacts & bit)
    for attempt in range(160):
        anchor_shape, proto = _prototype_layout(book, bool(rng.integers(2)))
        mode = "corner" if artifacts else str(rng.choice(("corner", "corner", "horizontal", "vertical")))
        labels = _partition(rng, anchor_shape, corner, count, stepped, mode,
                            spacious=artifacts)
        if labels is None or max(labels.shape) > 22 or labels.size > 400:
            continue
        H, W = labels.shape
        raw = np.full((H, W), carrier, dtype=np.int8)
        occupied = np.zeros(raw.shape, dtype=bool)
        q = np.argwhere(labels == 0)
        ar, ac = q.min(0)
        # Prototype orientation does not follow the L: all anchors remain top-left.
        records = []
        for r, c, code in proto:
            r, c = int(ar) + r, int(ac) + c
            observed = _place_record(raw, occupied, r, c, code, book, carrier, "live", rng)
            if observed is None:
                break
            records.append((r, c, code))
        if len(records) != len(book):
            continue
        # Connect every piece to the anchored piece using OBSERVED motif fragments.
        reached = {0}
        cross_count = 0
        for _ in range(count + 3):
            if len(reached) == count:
                break
            options = [(r, c, code) for code, (_, mask) in book.items()
                       for r in range(H - mask.shape[0] + 1)
                       for c in range(W - mask.shape[1] + 1)
                       if len(set(map(int, np.unique(labels[r:r + mask.shape[0], c:c + mask.shape[1]])))) > 1]
            rng.shuffle(options)
            placed = False
            for r, c, code in options:
                mask = book[code][1]
                ph, pw = mask.shape
                ink_labels = set(map(int, labels[r:r + ph, c:c + pw][mask]))
                ink_labels.add(int(labels[r, c]))
                if not (ink_labels & reached and ink_labels - reached):
                    continue
                old, old_occupied = raw.copy(), occupied.copy()
                observation = _place_record(raw, occupied, r, c, code, book, carrier,
                                            "live", rng, partial=partial)
                if observation is None:
                    continue
                observed_labels = set(map(int, labels[r:r + ph, c:c + pw][observation]))
                observed_labels.add(int(labels[r, c]))
                if not (observed_labels & reached and observed_labels - reached):
                    raw, occupied = old, old_occupied
                    continue
                reached |= observed_labels
                records.append((r, c, code))
                cross_count += 1
                placed = True
                break
            if not placed:
                break
        if len(reached) != count or not cross_count:
            continue
        if artifacts:
            success = True
            for kind in artifact_kinds:
                options = [(r, c, code) for code, (_, mask) in book.items()
                           for r in range(H - mask.shape[0] + 1)
                           for c in range(W - mask.shape[1] + 1)]
                rng.shuffle(options)
                if not any(_place_record(raw, occupied, r, c, code, book, carrier, kind, rng)
                           is not None for r, c, code in options):
                    success = False
                    break
            if not success:
                continue
        if clipped:
            clip_ok = True
            for boundary in ("corner", "bottom", "right"):
                options = [(r, c, code) for code, (_, mask) in book.items()
                           for r in range(H) for c in range(W)
                           if ((r + mask.shape[0] > H, c + mask.shape[1] > W)
                               == {"bottom": (True, False), "right": (False, True),
                                   "corner": (True, True)}[boundary])
                           and mask[:H - r, :W - c].any()]
                rng.shuffle(options)
                for r, c, code in options:
                    if _place_record(raw, occupied, r, c, code, book, carrier, "live", rng) is not None:
                        records.append((r, c, code))
                        break
                else:
                    clip_ok = False
                    break
            if not clip_ok:
                continue
        material = _material(raw, carrier, book)
        if material is None or not material[1]:
            continue
        rendered = _render_records(raw.shape, carrier, records, book)
        if not np.array_equal(rendered, material[0]):
            continue
        pieces = _cut(labels, raw)
        try:
            learned = _read_book(pieces[0][0], carrier)
        except ValueError:
            continue
        if set(learned) != set(book) or any(not np.array_equal(learned[c][1], book[c][1]) for c in book):
            continue
        # Canvas size responds to material area, with independent extra margins.
        base = max(21, int(np.ceil(np.sqrt(H * W * 2.4))))
        canvas = (min(30, max(H + 2, base + int(rng.integers(0, 5)))),
                  min(30, max(W + 2, base + int(rng.integers(0, 5)))))
        oy_low, ox_low = int(corner[0] == "T"), int(corner[1] == "L")
        oy_high = canvas[0] - H - int(corner[0] == "B")
        ox_high = canvas[1] - W - int(corner[1] == "R")
        if oy_high < oy_low or ox_high < ox_low:
            continue
        oy = int(rng.integers(oy_low, oy_high + 1))
        ox = int(rng.integers(ox_low, ox_high + 1))
        point = (oy + (H - 1 if corner[0] == "B" else 0),
                 ox + (W - 1 if corner[1] == "R" else 0))
        marks = _marker_cells(point, corner)
        final = [(oy + r, ox + c) for _, (r, c) in pieces]
        positions = _scatter(rng, pieces, canvas, final, 0, marks)
        if positions is None:
            continue
        g = np.full(canvas, bg, dtype=np.int8)
        for (m, _), (r, c) in zip(pieces, positions):
            g[r:r + m.shape[0], c:c + m.shape[1]][m >= 0] = m[m >= 0]
        for p in marks:
            g[p] = marker
        expected = np.full(canvas, bg, dtype=np.int8)
        expected[oy:oy + H, ox:ox + W] = rendered
        try:
            predicted, info = _infer(g)
        except ValueError:
            continue
        if not np.array_equal(predicted, expected):
            continue
        features = _features(info)
        if partial and not features["partial_cross"]:
            continue
        if "attached" in artifact_kinds and not features["all_codes_fails"]:
            continue
        # At least one ambiguous geometric packing makes B's template evidence
        # necessary for placement, beyond its always-necessary rendering effect.
        if not partial and info["distinct_geometric_sheets"] < 2:
            continue
        return {"input": g, "output": expected}
    raise RuntimeError("bounded stencil-sheet construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    classes = int(rng.integers(1, 3))
    book = {}
    for i in range(classes):
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        book[palette[2 * i]] = (palette[2 * i + 1], _sample_mask(rng, h, w))
    available = palette[2 * classes:]
    roles = tuple(available[:3])
    corners = list(rng.choice(_CORNERS, 2, replace=False))
    plans = [(corners[0], 3, False, False, 0, False),
             (corners[1], 4, False, True, 3, True)]
    for _ in range(num_train - 2):
        plans.append((str(rng.choice(corners)), int(rng.integers(3, 6)),
                      bool(rng.random() < .45), True, int(rng.integers(0, 4)),
                      bool(rng.random() < .5)))
    rng.shuffle(plans)
    puzzle = {"train": [], "test": []}
    seen = set()
    max_train_pieces = max(plan[1] for plan in plans)
    taught_steps = any(plan[2] for plan in plans)
    for split, plans_split in (("train", plans), ("test", [
            (str(rng.choice(_CORNERS)), int(rng.integers(max_train_pieces + 1, 7)),
             taught_steps and bool(rng.random() < .5), True,
             int(rng.integers(0, 4)), True) for _ in range(num_test)])):
        for plan in plans_split:
            # Roles are fixed on some puzzles, locally reassigned on others.
            local_roles = roles if rng.random() < .6 else tuple(map(int, rng.choice(available, 3, replace=False)))
            for _ in range(8):
                pair = _make_pair(rng, book, local_roles, *plan)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not create distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _book_key(book):
    return tuple((c, ink, m.shape, m.tobytes()) for c, (ink, m) in sorted(book.items()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, dictionary, infos = set(), None, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            key = (g.shape, g.tobytes())
            assert key not in seen
            seen.add(key)
            predicted, info = _infer(g)
            assert np.array_equal(predicted, out), "output disagrees with input-only reconstruction"
            bk = _book_key(info["book"])
            if dictionary is None:
                dictionary = bk
            assert dictionary == bk, "dictionary changes within the puzzle"
            bg, carrier, marker = info["bg"], info["carrier"], info["marker"]
            assert not np.any(out == marker) and not np.isin(out, list(info["book"])).any()
            assert all(out[p] == bg for p in _marker_cells(info["point"], info["corner"]))
            r, c = info["origin"]
            h, w = info["shape"]
            assert np.count_nonzero(out != bg) == h * w
            assert np.all(out[r:r + h, c:c + w] != bg)
            assert sum(int(np.count_nonzero(m >= 0)) for m, _ in info["pieces"]) == h * w
            assert info["positions"][info["anchor"]] == info["pieces"][info["anchor"]][1]
            assert any(pos != old for (_, old), pos in zip(info["pieces"], info["positions"]))
            assert set(map(int, np.unique(out))) <= {bg, carrier} | {v[0] for v in info["book"].values()}
            assert info["live"]
            assert len(_components((g != bg) & (g != marker), True)) == len(info["pieces"]), \
                "distinct source pieces must have an eight-neighbor blank gap"
            features = _features(info)
            assert features["joined"] and features["cross"], \
                "every piece must participate in the observed cross-cut motif network"
            # A-ablation: render at scattered positions, without joining the sheet.
            unassembled = np.full(g.shape, bg, dtype=np.int8)
            for m, old in info["pieces"]:
                local = np.where(m >= 0, m, carrier)
                result = _material(local, carrier, info["book"])
                tile = np.full(m.shape, carrier, dtype=np.int8) if result is None else result[0]
                rr, cc = old
                unassembled[rr:rr + m.shape[0], cc:cc + m.shape[1]][m >= 0] = tile[m >= 0]
            assert not np.array_equal(unassembled, out), "assembly contribution has no visible effect"
            # B-ablation: a correctly joined raw sheet is still the wrong answer.
            unrendered = np.full(g.shape, bg, dtype=np.int8)
            unrendered[r:r + h, c:c + w] = info["sheet"]
            assert not np.array_equal(unrendered, out), "symbolic rendering has no visible effect"
            infos[split].append(info)
    training = infos["train"]
    assert any(i["distinct_geometric_sheets"] > 1 for i in training)
    assert any(_features(i)["partial_cross"] for i in training)
    assert any(i["bare"] and i["rejected"] for i in training)
    assert any(_features(i)["all_codes_fails"] for i in training)
    train_boundaries = set().union(*(_features(i)["boundaries"] for i in training))
    assert train_boundaries == {"bottom", "right", "corner"}
    for i in infos["test"]:
        features = _features(i)
        assert features["partial_cross"] and features["boundaries"] <= train_boundaries
        assert len(i["pieces"]) > max(len(t["pieces"]) for t in training)
        if features["stepped"]:
            assert any(_features(t)["stepped"] for t in training)


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, books, shapes, partitions, corners, piece_counts = set(), set(), set(), set(), set(), set()
    stepped = 0
    mask_shapes, binary_masks, class_counts, sheet_shapes = set(), set(), set(), set()
    spans = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        serial = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in p[s]] for s in p}
        fingerprints.add(hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest())
        for pairs in p.values():
            for pair in pairs:
                _, i = _infer(_grid(pair["input"]))
                books.add(_book_key(i["book"]))
                shapes.add(pair["input"].shape)
                partitions.add((i["labels"].shape, i["labels"].tobytes()))
                corners.add(i["corner"])
                piece_counts.add(len(i["pieces"]))
                stepped += any(np.any(m < 0) for m, _ in i["pieces"])
                sheet_shapes.add(i["shape"])
                class_counts.add(len(i["book"]))
                spans.add(_features(i)["max_span"])
                for _, mask in i["book"].values():
                    mask_shapes.add(mask.shape)
                    binary_masks.add((mask.shape, mask.tobytes()))
    assert len(fingerprints) == 42 and len(books) >= 35
    assert len(shapes) >= 15 and len(partitions) >= 60
    assert corners == set(_CORNERS) and piece_counts >= {3, 4, 5, 6}
    assert stepped >= 5
    assert class_counts == {1, 2} and len(mask_shapes) >= 7 and len(binary_masks) >= 25
    assert len(sheet_shapes) >= 35 and max(spans) >= 3
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(100 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Same input, reversed segmentation order: no construction order can break ties.
    pair = puzzles[0]["test"][0]
    _, i = _infer(pair["input"])
    order = list(reversed(range(len(i["pieces"]))))
    pieces = [i["pieces"][j] for j in order]
    outputs, sheets = set(), set()
    for shape, origin, positions in _packings(pieces, order.index(i["anchor"]), i["point"], i["corner"], pair["input"].shape):
        sheet, _ = _assemble(pieces, positions, shape)
        material = _material(sheet, i["carrier"], i["book"])
        if material is not None:
            out = np.full(pair["input"].shape, i["bg"], dtype=np.int8)
            r, c = origin
            out[r:r + shape[0], c:c + shape[1]] = material[0]
            outputs.add(out.tobytes())
            sheets.add((shape, origin, sheet.tobytes()))
    assert outputs == {pair["output"].tobytes()} and len(sheets) == 1
    # Two quiet equal-height strips can exchange places; reject distinct sheets.
    ambiguous = np.zeros((22, 14), dtype=np.int8)
    ambiguous[1:8, 1:8] = 5
    ref = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    ambiguous[3:6, 3:6][ref] = 3
    ambiguous[3, 3] = 2
    ambiguous[15:17, 1:8] = ambiguous[18:20, 1:8] = 5
    ambiguous[15, 1] = 2
    for p in _marker_cells((1, 1), "TL"):
        ambiguous[p] = 9
    try:
        solve(ambiguous)
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("ambiguous assembly accepted")
    for kind in ("ink", "marker", "shift", "duplicate"):
        p = deepcopy(puzzles[0])
        pair = p["test"][0]
        _, i = _infer(pair["input"])
        if kind == "ink":
            ink = next(iter(i["book"].values()))[0]
            pair["output"][tuple(np.argwhere(pair["output"] == ink)[0])] = i["carrier"]
        elif kind == "marker":
            pair["output"][_marker_cells(i["point"], i["corner"])[0]] = i["marker"]
        elif kind == "shift":
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        else:
            p["train"][1] = deepcopy(p["train"][0])
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted corruption: " + kind)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
