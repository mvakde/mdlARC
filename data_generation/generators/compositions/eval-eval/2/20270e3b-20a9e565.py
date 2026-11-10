"""Zippered continuation: register interleaved evidence and a distributed query.

Only NumPy and the standard library are used. No files, RNG, demonstrations,
metadata or hidden palette conventions are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


STYLES = ("filled", "hollow", "comb", "ladder", "F", "zigzag")
HOST_SEEDS = (0, 204330584, 956251119, 458355229, 898170691, 300724982,
              676820928, 613354724, 240694801, 425220731, 876180782,
              534276108, 389683576, 588575696, 481053104, 95484257,
              653324727, 686019428, 144342974, 768930524, 807892969,
              679979048, 821913571, 42570895, 873410437, 361564268,
              22303087, 111380582, 504502212, 958463787, 348650354,
              531369624, 573437776, 49636052, 149407882, 793582735,
              871277499, 73473611, 911207955, 507481922, 252424177,
              302910311)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size"
    assert np.issubdtype(a.dtype, np.integer), "integer colors required"
    assert np.all((0 <= a) & (a <= 9)), "color range"
    return a


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    cells = {tuple(map(int, q)) for q in np.argwhere(mask)}
    parts = []
    while cells:
        root = min(cells)
        cells.remove(root)
        part, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in cells:
                        cells.remove(q)
                        part.add(q)
                        stack.append(q)
        parts.append(part)
    return parts


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _corners(bounds):
    t, b, l, r = bounds
    return ({(t, l), (t, l + 1), (t + 1, l)},
            {(b, r), (b, r - 1), (b - 1, r)})


def _cards(grid):
    """Zero is the exterior; two filled rectangles identify ownership exactly."""
    parts = _components(grid != 0)
    assert len(parts) == 2, "exactly two separated cards"
    cards = []
    for part in parts:
        t, b, l, r = _box(part)
        assert len(part) == (b - t + 1) * (r - l + 1), "solid card silhouette"
        a = grid[t:b + 1, l:r + 1].copy()
        border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
        assert len(np.unique(border)) == 1, "unannotated paper border"
        paper = int(border[0])
        assert paper != 0
        cards.append((a, (t, l), paper))
    assert cards[0][2] == cards[1][2], "shared transparent paper role"
    return cards


def _parse(grid, seam=-1):
    cards = _cards(grid)
    paper = cards[0][2]
    colors = set(map(int, np.unique(grid))) - {0, paper}
    corners = []
    for color in sorted(colors):
        records = []
        for a, origin, _ in cards:
            pos = {tuple(map(int, q)) for q in np.argwhere(a == color)}
            if len(pos) != 3:
                break
            t, b, l, r = _box(pos)
            if b - t != 1 or r - l != 1:
                break
            ul, br = _corners((t, b, l, r))
            if pos == ul:
                records.append(("anchor", pos, a, origin))
            elif pos == br:
                records.append(("mover", pos, a, origin))
            else:
                break
        if len(records) == 2 and {x[0] for x in records} == {"anchor", "mover"}:
            corners.append((color, records))
    assert len(corners) == 1, "one upper-left and one lower-right L, on separate cards"
    marker, records = corners[0]
    guides = []
    for color in sorted(colors - {marker}):
        bars = [{tuple(map(int, q)) for q in np.argwhere(a == color)}
                for _, _, a, _ in records]
        if len(bars[0]) != len(bars[1]) or not 2 <= len(bars[0]) <= 5:
            continue
        for transpose in (False, True):
            bs = [{(c, r) if transpose else (r, c) for r, c in bar} for bar in bars]
            if all(len({c for r, c in bar}) == 1 and
                   max(r for r, c in bar) - min(r for r, c in bar) + 1 == len(bar)
                   for bar in bs):
                guides.append((color, transpose, bs))
    assert len(guides) == 1, "unique congruent registration bars"
    guide, transpose, bars = guides[0]
    cells, annotations, by_owner = {}, {}, {}
    raw_annotations = set()
    translations = {}
    for (role, annotation, a, origin), bar in zip(records, bars):
        oriented = a.T if transpose else a
        gr, _, gc, _ = _box(bar)
        # Congruent guides agree tangentially; mover guide goes one normal cell
        # to the left of the anchor guide. No individual card is rotated.
        shift = (-gr, -gc + (seam if role == "mover" else 0))
        translations[role] = shift
        own = {}
        for rr, cc in np.argwhere((oriented != paper) & (oriented != guide) & (oriented != marker)):
            q = int(rr) + shift[0], int(cc) + shift[1]
            color = int(oriented[rr, cc])
            assert color != 0, "exterior may not occur inside paper"
            assert q not in cells or cells[q] == color, "conflicting ink collision"
            cells[q] = own[q] = color
        by_owner[role] = own
        moved = {(c + shift[0], r + shift[1]) if transpose else
                 (r + shift[0], c + shift[1]) for r, c in annotation}
        annotations[role] = moved
        raw_annotations.update((r + origin[0], c + origin[1]) for r, c in annotation)
    assert cells and all(by_owner.values())
    t, b, l, r = _box(annotations["anchor"] | annotations["mover"])
    bounds = t, b, l, r
    ul, br = _corners(bounds)
    assert annotations["anchor"] == ul and annotations["mover"] == br, "aligned inclusive query"
    assert b - t >= 1 and r - l >= 1 and (b - t) + (r - l) >= 4
    assert max(b - t + 1, r - l + 1) <= 30, "query size"
    assert all(not (t <= y <= b and l <= x <= r) for y, x in cells), "query is erased"
    return dict(paper=paper, marker=marker, guide=guide, transpose=transpose,
                cells=cells, by_owner=by_owner, bounds=bounds,
                bar_length=len(bars[0]), raw_bounds=_box(raw_annotations),
                translations=translations, cards=[a.shape for a, _, _ in cards])


def _template(style, h, w, spacing):
    """Cell predicates used by inference; -1 is blank, other values are depths."""
    if h < 3 or w < 1:
        return None
    y, x = np.indices((h, w))
    if style == "filled":
        mask = np.ones((h, w), dtype=bool)
    elif style == "hollow":
        if w < 3:
            return None
        mask = (y == 0) | (y == h - 1) | (x == 0) | (x == w - 1)
    elif style == "comb":
        if w < 2 or h < spacing + 1:
            return None
        mask = (x == 0) | (y % spacing == 0)
    elif style == "ladder":
        if w < 3 or h < spacing + 1:
            return None
        mask = (x == 0) | (x == w - 1) | (y % spacing == 0)
    elif style == "F":
        if w < 2 or h < spacing + 2:
            return None
        mask = (x == 0) | (y == 0) | (y == spacing)
    elif style == "zigzag":
        if w < 2 or h < w:
            return None
        phase = y % (2 * w - 2)
        mask = x == np.minimum(phase, 2 * w - 2 - phase)
    else:
        return None
    return np.where(mask, y // spacing, -1)


def _line(values):
    delta = values[1] - values[0]
    if any(v != values[0] + i * delta for i, v in enumerate(values)):
        return None
    return int(values[0]), int(delta)


def _segment(parsed):
    cells, paper = parsed["cells"], parsed["paper"]
    columns = sorted({c for r, c in cells})
    groups = []
    for c in columns:
        if not groups or c > groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(c)
    assert 4 <= len(groups) <= 8, "four to eight consecutive interleaved observations"
    patches, starts, tops, owners = [], [], [], []
    for group in groups:
        sub = {q: v for q, v in cells.items() if q[1] in group}
        t, b, l, r = _box(sub)
        a = np.full((b - t + 1, r - l + 1), paper, dtype=np.int8)
        for (y, x), v in sub.items():
            a[y - t, x - l] = v
        patches.append(a)
        starts.append(l)
        tops.append(t)
        present = tuple(role for role, own in parsed["by_owner"].items() if any(q in own for q in sub))
        # A specimen is either on one card or duplicated identically on both.
        assert all(all(q in parsed["by_owner"][role] for q in sub) for role in present)
        owners.append(frozenset(present))
    return patches, starts, tops, owners


def _geometry(model, i):
    h = model["h"][0] + i * model["h"][1]
    w = model["w"][0] + i * model["w"][1]
    y = model["top"][0] + i * model["top"][1]
    x = model["x"] + i * (model["w"][0] + model["gap"]) + i * (i - 1) // 2 * model["w"][1]
    return y, x, h, w


def _predict(model, bounds, paper, **changes):
    m = dict(model)
    m.update(changes)
    t, b, l, r = bounds
    result = np.full((b - t + 1, r - l + 1), paper, dtype=np.int8)
    # The finite family bounds both observed and traversed object indices.
    forward = l > m["last"]
    ids = range(m["n"], 17) if forward else range(-1, -17, -1)
    for i in ids:
        y, x, h, w = _geometry(m, i)
        if (forward and x > r) or (not forward and x + w <= l):
            break
        if min(h, w) < 1 or max(h, w) > 60:
            return None
        template = _template(m["style"], h, w, m["spacing"])
        if template is None:
            return None
        template = template[::m["flip_r"], ::m["flip_c"]]
        for rr in range(max(t, y), min(b + 1, y + h)):
            for cc in range(max(l, x), min(r + 1, x + w)):
                depth = int(template[rr - y, cc - x])
                if depth >= 0:
                    phase = (i * m["advance"] + depth * m["banded"]) % len(m["palette"])
                    result[rr - t, cc - l] = m["palette"][phase]
    else:
        return None
    return result


def _models(parsed):
    patches, starts, tops, owners = _segment(parsed)
    n = len(patches)
    hl = _line([a.shape[0] for a in patches])
    wl = _line([a.shape[1] for a in patches])
    tl = _line(tops)
    assert hl is not None and wl is not None and tl is not None, "arithmetic geometry"
    gaps = [starts[i + 1] - starts[i] - patches[i].shape[1] for i in range(n - 1)]
    assert len(set(gaps)) == 1 and 1 <= gaps[0] <= 4, "constant positive inter-object gap"
    t, b, l, r = parsed["bounds"]
    assert l > starts[-1] + patches[-1].shape[1] - 1 or r < starts[0], "external continuation query"
    colors = sorted({int(v) for a in patches for v in np.unique(a) if v != parsed["paper"]})
    assert 2 <= len(colors) <= 3 and n >= len(colors) + 1, "observed cyclic phase and repetition"
    base = dict(h=hl, w=wl, top=tl, x=starts[0], gap=gaps[0], n=n,
                last=starts[-1] + patches[-1].shape[1] - 1, advance=1)
    answers = []
    for style in STYLES:
        for spacing in (2, 3):
            for flip_r in (1, -1):
                for flip_c in (1, -1):
                    templates = [_template(style, *a.shape, spacing) for a in patches]
                    if any(a is None for a in templates):
                        continue
                    templates = [a[::flip_r, ::flip_c] for a in templates]
                    if any(not np.array_equal(a != parsed["paper"], z >= 0)
                           for a, z in zip(patches, templates)):
                        continue
                    for banded in (0, 1):
                        mapping = {}
                        good = True
                        for i, (a, z) in enumerate(zip(patches, templates)):
                            for depth in np.unique(z[z >= 0]):
                                values = np.unique(a[z == depth])
                                residue = (i + int(depth) * banded) % len(colors)
                                if len(values) != 1 or (residue in mapping and mapping[residue] != int(values[0])):
                                    good = False
                                    break
                                mapping[residue] = int(values[0])
                            if not good:
                                break
                        if not good or len(mapping) != len(colors) or set(mapping.values()) != set(colors):
                            continue
                        m = dict(base, style=style, spacing=spacing, flip_r=flip_r,
                                 flip_c=flip_c, banded=banded,
                                 palette=tuple(mapping[k] for k in range(len(colors))))
                        pred = _predict(m, parsed["bounds"], parsed["paper"])
                        if pred is not None and np.any(pred != parsed["paper"]):
                            answers.append((pred, m))
    assert answers, "unsupported sequence"
    assert all(np.array_equal(answers[0][0], a) for a, _ in answers), "ambiguous extrapolation"
    return answers, (patches, starts, tops, owners)


def _analyze(input_grid):
    parsed = _parse(_grid(input_grid))
    answers, segmented = _models(parsed)
    prediction, model = answers[0]
    result = prediction.T.copy() if parsed["transpose"] else prediction.copy()
    patches, starts, tops, owners = segmented
    # Exactly the alternating members are split. An optional shared specimen
    # supplies a visible same-color collision, with transparent union semantics.
    shared = [i for i, o in enumerate(owners) if len(o) == 2]
    assert len(shared) <= 1 and set(shared) <= {0, len(owners) - 1}, "only one shared endpoint specimen"
    exclusive = [next(iter(o)) for o in owners if len(o) == 1]
    assert len(exclusive) >= 3 and all(a != b for a, b in zip(exclusive, exclusive[1:])), "zippered ownership"
    assert set(exclusive) == {"anchor", "mover"}
    t, b, l, r = parsed["bounds"]
    raw = parsed["raw_bounds"]
    raw_shape = raw[1] - raw[0] + 1, raw[3] - raw[2] + 1
    constant = _predict(model, parsed["bounds"], parsed["paper"],
                        h=(model["h"][0], 0), w=(model["w"][0], 0),
                        top=(model["top"][0], 0))
    phase_reset = _predict(model, parsed["bounds"], parsed["paper"], advance=0)
    nearest = None
    target_ids = []
    cuts = []
    forward = l > starts[-1]
    for i in (range(model["n"], 17) if forward else range(-1, -17, -1)):
        y, x, h, w = _geometry(model, i)
        if (forward and x > r) or (not forward and x + w <= l):
            break
        if min(h, w) < 1:
            break
        if x <= r and x + w > l and y <= b and y + h > t:
            z = _template(model["style"], h, w, model["spacing"])
            if z is None:
                break
            z = z[::model["flip_r"], ::model["flip_c"]]
            if np.any(z[max(0, t - y):min(h, b - y + 1),
                        max(0, l - x):min(w, r - x + 1)] >= 0):
                target_ids.append(i)
                cuts.append((l > x or r < x + w - 1, t > y or b < y + h - 1))
    assert target_ids
    nearest = min(target_ids) if forward else max(target_ids)
    info = dict(style=model["style"], growth=(model["h"][1], model["w"][1]),
                drift=model["top"][1], gap=model["gap"], visible=len(patches),
                banded=model["banded"], cycle=len(model["palette"]),
                palette=model["palette"], transpose=parsed["transpose"],
                direction=("down" if forward else "up") if parsed["transpose"] else
                ("right" if forward else "left"), bounds=parsed["bounds"],
                skip=nearest - model["n"] if forward else -nearest - 1,
                bar_length=parsed["bar_length"], paper=parsed["paper"],
                marker=parsed["marker"], guide=parsed["guide"],
                duplicate=sum(len(o) == 2 for o in owners),
                target_count=len(target_ids), axial_cut=any(x[0] for x in cuts),
                transverse_cut=any(x[1] for x in cuts),
                shape=result.shape, raw_shape=raw_shape,
                constant_distinct=constant is None or not np.array_equal(constant, prediction),
                phase_distinct=phase_reset is None or not np.array_equal(phase_reset, prediction),
                candidate_models=len(answers))
    return result, info, parsed, model


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive ownership, registration, sequence and crop entirely from pixels."""
    return _analyze(np.asarray(input_grid))[0]


def _paint(style, h, w, spacing, palette, i, banded, flip_r, flip_c, paper):
    """Label-side explicit strokes, independent of the inference predicates."""
    a = np.full((h, w), paper, dtype=np.int8)
    for y in range(h):
        stroke = set()
        if style == "filled":
            stroke.update(range(w))
        elif style == "hollow":
            stroke.update(range(w) if y in (0, h - 1) else (0, w - 1))
        elif style in ("comb", "ladder"):
            stroke.add(0)
            if style == "ladder":
                stroke.add(w - 1)
            if y % spacing == 0:
                stroke.update(range(w))
        elif style == "F":
            stroke.update(range(w) if y in (0, spacing) else (0,))
        elif style == "zigzag":
            phase = y % (2 * w - 2)
            stroke.add(phase if phase < w else 2 * w - 2 - phase)
        color = palette[(i + (y // spacing if banded else 0)) % len(palette)]
        for x in stroke:
            a[y, x] = color
    return a[::flip_r, ::flip_c].copy()


def _render_card(ink, annotation, guide_col, guide_length, palette):
    paper, guide, marker = palette[:3]
    guide_cells = {(r, guide_col) for r in range(guide_length)}
    all_cells = set(ink) | annotation | guide_cells
    t, b, l, r = _box(all_cells)
    if min(b - t + 3, r - l + 3) < 3 or max(b - t + 3, r - l + 3) > 30:
        return None
    assert not set(ink) & (annotation | guide_cells)
    assert not annotation & guide_cells
    grid = np.full((b - t + 3, r - l + 3), paper, dtype=np.int8)
    for cells, color in ((guide_cells, guide), (annotation, marker)):
        for y, x in cells:
            grid[y - t + 1, x - l + 1] = color
    for (y, x), color in ink.items():
        grid[y - t + 1, x - l + 1] = color
    return grid


def _stage(rng, cards, axis):
    order = list(rng.permutation(2))
    a, b = [cards[int(i)] for i in order]
    # Fit all feasible staging directions; the content axis is independent.
    modes = []
    for vertical in (True, False):
        ah, aw = a.shape
        bh, bw = b.shape
        base_h = ah + bh + 1 if vertical else max(ah, bh)
        base_w = max(aw, bw) if vertical else aw + bw + 1
        if max(base_h, base_w) <= 30:
            modes.append(vertical)
    if not modes:
        return None
    vertical = bool(rng.choice(modes))
    ah, aw = a.shape
    bh, bw = b.shape
    gap = int(rng.integers(1, 4))
    if vertical:
        gap = min(gap, 30 - ah - bh)
        spare = 30 - max(aw, bw)
        jitter = int(rng.integers(min(4, spare) + 1))
        origins = [(0, 0), (ah + gap, jitter)]
    else:
        gap = min(gap, 30 - aw - bw)
        spare = 30 - max(ah, bh)
        jitter = int(rng.integers(min(4, spare) + 1))
        origins = [(0, 0), (jitter, aw + gap)]
    height = max(y + g.shape[0] for (y, x), g in zip(origins, (a, b)))
    width = max(x + g.shape[1] for (y, x), g in zip(origins, (a, b)))
    if max(height, width) > 30:
        return None
    padt = int(rng.integers(min(2, 30 - height) + 1))
    padb = int(rng.integers(min(2, 30 - height - padt) + 1))
    padl = int(rng.integers(min(2, 30 - width) + 1))
    padr = int(rng.integers(min(2, 30 - width - padl) + 1))
    grid = np.zeros((height + padt + padb, width + padl + padr), dtype=np.int8)
    for g, (y, x) in zip((a, b), origins):
        grid[y + padt:y + padt + g.shape[0], x + padl:x + padl + g.shape[1]] = g
    return grid.T.copy() if axis else grid


def _candidate(rng, profile):
    style = profile.get("style") or str(rng.choice(STYLES))
    spacing = int(rng.integers(2, 4))
    n = int(rng.integers(4, 7))
    p = int(rng.integers(2, 4))
    palette = profile["palette"]
    content = palette[3:3 + p]
    banded = int(rng.integers(2))
    flip_r, flip_c = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    # Simple rectangles also permit width growth; thin stroke families mostly
    # grow across their transverse axis to leave space for extrapolation.
    dw = int(rng.choice([0, 0, 1])) if style in ("filled", "hollow") else 0
    dh = int(rng.choice([-1, 0, 1]))
    drift = int(rng.choice([-1, 0, 0, 0, 1]))
    if profile["teaching"]:
        dh = int(rng.choice([-1, 1]))
        dw = 0
        drift = 0
        p = 2
        content = palette[3:5]
        banded = 0
    gap = int(rng.integers(1, 4))
    base_w = int(rng.integers(1 if style == "filled" else 3 if style in ("hollow", "ladder") else 2, 5))
    base_h = int(rng.integers(4, 8))
    if profile.get("width_case"):
        style, n, gap, dh, drift, banded = "filled", 4, 1, 0, 0, 0
        dw = profile["width_case"]
        base_w = 1 if dw > 0 else 5
        p, content = 2, palette[3:5]
    side = profile["side"]
    skip = profile["skip"]
    span = profile["span"]
    target = list(range(n + skip, n + skip + span)) if side == 1 else list(range(-skip - span, -skip))
    all_ids = list(range(min(0, min(target)), max(n - 1, max(target)) + 1))
    # Ensure shrinking and backward queries do not cross a degenerate object.
    min_h = max(3, spacing + 2 if style == "F" else spacing + 1 if style in ("comb", "ladder") else 3)
    base_h += max(0, min_h - min(base_h + i * dh for i in all_ids))
    base_w += max(0, (3 if style in ("hollow", "ladder") else 2 if style in ("comb", "F", "zigzag") else 1) -
                  min(base_w + i * dw for i in all_ids))
    objects = {}
    x0 = 3  # Relative to the anchor registration bar, never a solver constant.
    for i in all_ids:
        h, w = base_h + i * dh, base_w + i * dw
        if max(h, w) > 13 or _template(style, h, w, spacing) is None:
            return None
        x = x0 + i * (base_w + gap) + i * (i - 1) // 2 * dw
        y = 2 + i * drift
        a = _paint(style, h, w, spacing, content, i, banded, flip_r, flip_c, palette[0])
        if not np.all(np.any(a != palette[0], axis=0)):
            return None
        objects[i] = (y, x, a)
    tys = [objects[i][0] for i in target]
    txs = [objects[i][1] for i in target]
    bottom = max(objects[i][0] + objects[i][2].shape[0] - 1 for i in target)
    right = max(objects[i][1] + objects[i][2].shape[1] - 1 for i in target)
    top, left = min(tys), min(txs)
    if profile["partial"]:
        # Partial transverse/axial windows preserve global object and color phase.
        top += int(rng.integers(max(1, (bottom - top) // 2 + 1)))
        if right - left >= 3:
            left += int(rng.integers(min(3, right - left - 1)))
        if rng.random() < .4:
            bottom -= int(rng.integers(max(1, (bottom - top) // 2 + 1)))
    if bottom - top < 1:
        top = bottom - 1
    if right - left < 1:
        right = left + 1
    if bottom - top + right - left < 4:
        top -= 4 - (bottom - top + right - left)
    bounds = top, bottom, left, right
    ul, br = _corners(bounds)
    if max(bottom - top + 1, right - left + 1) > 30:
        return None
    answer = np.full((bottom - top + 1, right - left + 1), palette[0], dtype=np.int8)
    for i in target:
        y, x, a = objects[i]
        for rr, cc in np.argwhere(a != palette[0]):
            gy, gx = y + int(rr), x + int(cc)
            if top <= gy <= bottom and left <= gx <= right:
                answer[gy - top, gx - left] = a[rr, cc]
    if not np.any(answer != palette[0]):
        return None
    inks = [{}, {}]
    start_owner = int(rng.integers(2))
    duplicate = int(rng.choice([0, n - 1])) if profile["duplicate"] else -1
    # Duplicating one interior specimen must not destroy the alternating proof.
    for i in range(n):
        y, x, a = objects[i]
        owners = (0, 1) if i == duplicate else ((i + start_owner) % 2,)
        for owner in owners:
            for rr, cc in np.argwhere(a != palette[0]):
                # In assembled coordinates mover guide is at x=-1. Before
                # staging its whole card is one column to the right.
                q = y + int(rr), x + int(cc) + owner
                inks[owner][q] = int(a[rr, cc])
    length = int(rng.integers(2, 6))
    # A guide may be placed to either side or above/below the evidence. Its
    # origin is the only registration datum; the paper border is irrelevant.
    # Canonical guide is x=0; collisions with backward content are rejected.
    shifted_br = {(y, x + 1) for y, x in br}
    try:
        cards = [_render_card(inks[0], ul, 0, length, palette),
                 _render_card(inks[1], shifted_br, 0, length, palette)]
    except AssertionError:
        return None
    if any(a is None for a in cards):
        return None
    grid = _stage(rng, cards, profile["axis"])
    if grid is None:
        return None
    if profile["axis"]:
        answer = answer.T.copy()
    try:
        prediction, info, _, _ = _analyze(grid)
    except AssertionError:
        return None
    if not np.array_equal(prediction, answer):
        return None
    if info["raw_shape"] == answer.shape:
        return None  # Teaching witness, not a restriction in the input-only solver.
    if profile["teaching"] and not (info["constant_distinct"] and info["phase_distinct"]):
        return None
    # Observe skip and multi-object targets from pixels, rather than trusting
    # a sampled profile whose crop might miss its first nominal object.
    if info["skip"] != skip or info["target_count"] != span:
        return None
    if profile["partial"] and not (info["axial_cut"] or info["transverse_cut"]):
        return None
    return {"input": grid, "output": answer}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = tuple(map(int, rng.permutation(np.arange(1, 10))))
    varying_palette = bool(rng.integers(2))
    main_axis = int(rng.integers(2))
    shift = str(rng.choice(["distance", "span", "direction"]))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            test = split == "test"
            palette = tuple(map(int, rng.permutation(np.arange(1, 10)))) if varying_palette else stable_palette
            profile = dict(palette=palette, teaching=not test and index == 0,
                           side=-1 if (shift == "direction" and test) else 1,
                           skip=1 if (shift == "distance" and test) else 0,
                           span=2 if (shift == "span" and test) else 1,
                           partial=(index > 0 or test), axis=main_axis,
                           duplicate=(index > 0 and bool(rng.random() < .3)),
                           width_case=(1 if int(seed) % 8 == 1 else -1) if
                           not test and index == 1 and int(seed) % 4 == 1 else 0)
            # Both axes are taught before a mixed-axis test is used. This is
            # variation inside a rule, not a mandatory horizontal-to-vertical split.
            if int(seed) % 3 == 0:
                profile["axis"] = (main_axis + index % 2) % 2
            for attempt in range(240):
                candidate = _candidate(rng, profile)
                if candidate is not None and _key(candidate["input"]) not in used:
                    used.add(_key(candidate["input"]))
                    puzzle[split].append(candidate)
                    break
            else:
                raise RuntimeError("bounded geometric sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "duplicate input"
            seen.add(_key(a))
            expected, features, parsed, model = _analyze(a)
            assert np.array_equal(expected, b), "incorrect registered continuation"
            assert set(map(int, np.unique(b))) <= {parsed["paper"], *model["palette"]}
            assert np.any(b != parsed["paper"]), "continuation is meaningful"
            assert not np.any(b == parsed["guide"]) and not np.any(b == parsed["marker"])
            # Ignoring continuation leaves the requested region empty. Ignoring
            # registration produces a different-sized raw L bounding rectangle.
            t, bottom, left, right = parsed["bounds"]
            assert not any(t <= y <= bottom and left <= x <= right for y, x in parsed["cells"])
            assert features["raw_shape"] != b.shape
            info[split].append(features)
    assert any(i["constant_distinct"] for i in info["train"]), "growth teaching witness"
    assert any(i["phase_distinct"] for i in info["train"]), "color-phase teaching witness"
    train_axes = {i["transpose"] for i in info["train"]}
    assert {i["transpose"] for i in info["test"]} <= train_axes, "untaught axis"
    # A real held-out property: farther target, multi-object span, or backward
    # query. All input-side fitting parameters remain established locally.
    assert all(i["skip"] == 0 and i["target_count"] == 1 and
               i["direction"] in ("right", "down") for i in info["train"])
    assert all(i["axial_cut"] or i["transverse_cut"]
               for i in info["train"][1:] + info["test"]), "genuine partial crop"
    assert (all(i["skip"] >= 1 for i in info["test"]) or
            all(i["direction"] in ("left", "up") for i in info["test"]) or
            all(i["target_count"] >= 2 for i in info["test"]))


def self_test() -> dict:
    """Bounded, no disk I/O; structural and semantic checks across 42 seeds."""
    features = []
    signatures, input_shapes, output_shapes, normalized = set(), set(), set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        h = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a.copy()), b)
                h.update(a.tobytes())
                h.update(b.tobytes())
                input_shapes.add(a.shape)
                output_shapes.add(b.shape)
                roles = {}
                normalized.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                features.append(_analyze(a)[1])
        signatures.add(h.hexdigest())
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(input_shapes) >= 20 and len(output_shapes) >= 20
    assert {i["style"] for i in features} == set(STYLES)
    assert {i["cycle"] for i in features} == {2, 3}
    assert {i["direction"] for i in features} == {"left", "right", "up", "down"}
    assert {i["bar_length"] for i in features} == {2, 3, 4, 5}
    assert any(i["growth"] == (0, 0) for i in features)
    assert any(i["growth"][0] > 0 for i in features) and any(i["growth"][0] < 0 for i in features)
    assert any(i["growth"][1] > 0 for i in features)
    assert any(i["growth"][1] < 0 for i in features)
    assert any(i["duplicate"] for i in features)
    assert any(i["target_count"] == 2 for i in features)
    assert any(i["banded"] for i in features) and any(i["skip"] > 0 for i in features)
    assert len({i["paper"] for i in features}) >= 6
    # Exact seed/count reproducibility, all legal explicit count combinations.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17, nt, ne)
            q = generate_puzzle(17, nt, ne)
            assert all(_key(a[k]) == _key(b[k]) for s in p
                       for a, b in zip(p[s], q[s]) for k in ("input", "output"))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(signatures),
                input_shapes=len(input_shapes), output_shapes=len(output_shapes),
                normalized_inputs=len(normalized),
                styles={s: sum(i["style"] == s for i in features) for s in STYLES},
                directions={s: sum(i["direction"] == s for i in features) for s in ("left", "right", "up", "down")},
                growth=sum(i["growth"] != (0, 0) for i in features),
                width_growth=sum(i["growth"][1] != 0 for i in features),
                banded=sum(bool(i["banded"]) for i in features),
                skipped=sum(i["skip"] > 0 for i in features),
                duplicated=sum(i["duplicate"] > 0 for i in features),
                multi_object=sum(i["target_count"] > 1 for i in features),
                partial=sum(i["axial_cut"] or i["transverse_cut"] for i in features))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
