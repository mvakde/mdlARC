"""Register two paper masks and sample one clockwise pinwheel ink field.

No parent imports, fixture bank, seed-dependent solver, or learned convention.
The solver uses interval membership; label construction uses a path walker.
"""
from __future__ import annotations

import json

import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _shift(cells, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _components(cells, diagonal=True):
    remaining, parts = set(cells), []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _cross(cells):
    """Return (length, center) only for an exact hollow equal-spoke cross."""
    if not cells or len(cells) % 4:
        return None
    length = len(cells) // 4
    if not 1 <= length <= 4:
        return None
    t, b, l, r = _bbox(cells)
    if b - t != 2 * length or r - l != 2 * length:
        return None
    center = (t + length, l + length)
    y, x = center
    wanted = {(y + d * dy, x + d * dx)
              for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1))
              for d in range(1, length + 1)}
    return (length, center) if cells == wanted else None


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a


def _parse(input_grid):
    a = _grid(input_grid)
    colors = [int(c) for c in np.unique(a)]
    if len(colors) != 4:
        raise ValueError("registered printing needs exactly four color roles")
    by_color = {c: {tuple(map(int, q)) for q in np.argwhere(a == c)} for c in colors}
    candidates = []
    for ink in colors:
        cross = _cross(by_color[ink])
        if cross is None:
            continue
        length, center = cross
        for marker in colors:
            if marker == ink:
                continue
            bars = _components(by_color[marker], False)
            if len(bars) != 2 or len(bars[0]) != len(bars[1]):
                continue
            k = len(bars[0])
            if not 2 <= k <= 5:
                continue
            for axis in (0, 1):
                if any(len({q[axis] for q in bar}) != 1 or
                       max(q[1-axis] for q in bar) - min(q[1-axis] for q in bar) + 1 != k
                       for bar in bars):
                    continue
                normal = (1, 0) if axis == 0 else (0, 1)
                for background in colors:
                    if background in (ink, marker):
                        continue
                    paper = next(c for c in colors if c not in (ink, marker, background))
                    material = by_color[paper] | by_color[ink]
                    components = _components(material | by_color[marker])
                    if len(components) != 2 or center not in by_color[paper]:
                        continue
                    records = []
                    for component in components:
                        own = [bar for bar in bars if bar <= component]
                        if len(own) != 1:
                            break
                        bar = own[0]
                        neg = any(q in material for q in _shift(bar, (-normal[0], -normal[1])))
                        pos = any(q in material for q in _shift(bar, normal))
                        if neg == pos:
                            break
                        records.append((neg, component & material, min(bar), bar))
                    if len(records) != 2 or records[0][0] == records[1][0]:
                        continue
                    records.sort(key=lambda record: not record[0])
                    _, anchor, qa, bar_a = records[0]
                    _, mover, qb, bar_b = records[1]
                    seed_owner = 0 if by_color[ink] <= anchor else 1
                    if not by_color[ink] <= (anchor if seed_owner == 0 else mover):
                        continue
                    y, x = center
                    seed_square = {(r, c) for r in range(y-length, y+length+1)
                                   for c in range(x-length, x+length+1)}
                    if not seed_square <= (anchor if seed_owner == 0 else mover):
                        continue
                    delta = (qa[0] - qb[0] - normal[0], qa[1] - qb[1] - normal[1])
                    candidates.append(dict(colors=(background, paper, ink, marker),
                                           anchor=anchor, mover=mover, delta=delta,
                                           center=center, length=length, seed_owner=seed_owner,
                                           axis=axis, bar_length=k, bars=(bar_a, bar_b)))
    if len(candidates) != 1:
        raise ValueError(f"input must have one geometric role assignment, found {len(candidates)}")
    return candidates[0]


def _field(shape, center, length):
    """Infinite-plane winding membership from closed-form line intervals.

    Coordinates are tested in the four rotational phases of the north arm.
    All potential re-entries are kept; canvas/support edges do not stop a path.
    """
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], xx - center[1]
    mask = np.zeros(shape, dtype=bool)
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    for _ in range(4):
        mask |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(
                    starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                run = 16*q + 4*s + 2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        yy, xx = xx, -yy
    return mask


def _trace_field(shape, center, length, first_exit=False, support=None):
    """Separate unit-step construction used for labels and cross-checking."""
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    mask = np.zeros(shape, dtype=bool)
    bound = max(cy, cx, h-1-cy, w-1-cx)
    for radial in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        y, x = cy, cx
        dy, dx = radial
        segments = [(dy, dx, radius)]
        dy, dx = dx, -dy
        for k in range(4 * (bound // 8 + 3)):
            segments.extend(((dy, dx, 4*k+2), (dx, -dy, radius-2), (dy, dx, radius)))
            dy, dx = dx, -dy
        stopped = False
        for dy, dx, count in segments:
            for _ in range(count):
                y, x = y + dy, x + dx
                if 0 <= y < h and 0 <= x < w and (support is None or support[y, x]):
                    mask[y, x] = True
                elif first_exit or support is not None:
                    stopped = True
                    break
            if stopped:
                break
    return mask


def _print_surface(material, center, length, colors, field=_field):
    bg, paper, ink, _ = colors
    t, b, l, r = _bbox(material)
    if max(b-t+1, r-l+1) > 30:
        raise ValueError("assembled paper exceeds ARC dimensions")
    support = np.zeros((b-t+1, r-l+1), dtype=bool)
    for y, x in material:
        support[y-t, x-l] = True
    winding = field(support.shape, (center[0]-t, center[1]-l), length)
    return np.where(support, np.where(winding, ink, paper), bg).astype(np.int8)


def _assembled(parsed, seam=-1):
    dr, dc = parsed["delta"]
    normal = (1, 0) if parsed["axis"] == 0 else (0, 1)
    delta = dr + (seam+1)*normal[0], dc + (seam+1)*normal[1]
    mover = _shift(parsed["mover"], delta)
    center = parsed["center"]
    if parsed["seed_owner"] == 1:
        center = center[0]+delta[0], center[1]+delta[1]
    return parsed["anchor"] | mover, center, mover


def solve(input_grid: np.ndarray) -> np.ndarray:
    parsed = _parse(input_grid)
    material, center, _ = _assembled(parsed)
    return _print_surface(material, center, parsed["length"], parsed["colors"])


def _footprint(rng, height, width, left, sign, style):
    """Connected stencil strokes with a full row beside the guide."""
    cells = set()
    cut = int(rng.integers(1, max(2, min(height, width)//2+1)))
    flip = bool(rng.integers(2))
    for d in range(1, height+1):
        for j in range(width):
            x = width-1-j if flip else j
            keep = True
            if style == "chamfer":
                keep = x >= max(0, cut-(height-d))
            elif style == "frame":
                keep = d in (1, height) or x in (0, width-1)
            elif style == "fork":
                keep = d == 1 or x in (0, width//2, width-1)
            elif style == "terrace":
                keep = d == 1 or x < max(2, width-(d-1)//2)
            elif style == "notch":
                keep = not (d > height-cut and x < cut+1)
            if keep or d == 1:
                cells.add((sign*d, left+j))
    return cells


def _piece(rng, length, bar_length, sign, seeded, compact, teaching, reach, overlap, side):
    minimum = 2*length + (5 if teaching else 1) if seeded else 3
    if seeded:
        h = int(rng.integers(minimum, max(minimum+1, 14 if not compact else minimum+2)))
        w = int(rng.integers(minimum, max(minimum+1, 14 if not compact else minimum+2)))
    else:
        h = int(rng.integers(3, 6) if compact else rng.integers(6, 12))
        w = int(rng.integers(max(4, bar_length), 8) if compact else rng.integers(max(7, bar_length), 13))
    w = max(w, bar_length+2)
    left = -int(rng.integers(1, w-bar_length))
    styles = ("solid", "chamfer", "frame", "fork", "terrace", "notch")
    # The other cutout always has a visible window or separated fingers.
    style = str(rng.choice(styles if seeded else ("frame", "fork")))
    material = _footprint(rng, h, w, left, sign, style)
    seed, center = set(), None
    protected = set()
    if seeded:
        margin = 2 if teaching else 0
        distance = int(rng.integers(length+1+margin, h-length-margin+1))
        x = int(rng.integers(left+length+margin, left+w-length-margin))
        center = sign*distance, x
        y = center[0]
        protected = {(r, c) for r in range(y-length, y+length+1)
                     for c in range(x-length, x+length+1)}
        material |= protected
        seed = {(y+d*dy, x+d*dx) for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1))
                for d in range(1, length+1)}
    if overlap:
        depth = int(rng.integers(5, 9) if reach else rng.integers(1, 4))
        thickness = int(rng.integers(1, 4))
        xs = range(-thickness, 0) if side == 0 else range(bar_length, bar_length+thickness)
        # A side tongue bends around, never onto, the guide itself.
        material.update((sign, x) for x in xs)
        material.update((-sign*d, x) for d in range(depth+1) for x in xs)
        # Ensure connection even if the tongue reaches outside the base bbox.
        material.update((sign, x) for x in range(min(left, min(xs)), max(left+w-1, max(xs))+1))
    # Partial backing makes uncovered guides meaningful; leave at least one.
    if overlap and bar_length > 2:
        count = int(rng.integers(1, bar_length))
        for x in rng.choice(bar_length, count, replace=False):
            if (sign, int(x)) not in protected:
                material.discard((sign, int(x)))
    return material, seed, center


def _stage(rng, pieces, markers, seeds, colors, mode):
    normalized = []
    for cells, bar, seed in zip(pieces, markers, seeds):
        t, b, l, r = _bbox(cells | bar)
        normalized.append((_shift(cells, (-t, -l)), _shift(bar, (-t, -l)),
                           _shift(seed, (-t, -l)), b-t+1, r-l+1))
    if rng.integers(2):
        normalized.reverse()
    af, am, ai, ah, aw = normalized[0]
    bf, bm, bi, bh, bw = normalized[1]
    gap = int(rng.integers(1, 5))
    if mode == "horizontal":
        delta = int(rng.integers(-bh+1, ah)), aw+gap
    elif mode == "vertical":
        delta = ah+gap, int(rng.integers(-bw+1, aw))
    else:
        delta = ah+gap, aw+gap
    material = af | _shift(bf, delta)
    guides = am | _shift(bm, delta)
    ink = ai | _shift(bi, delta)
    t, b, l, r = _bbox(material | guides)
    h, w = b-t+1, r-l+1
    if max(h, w) > 30:
        return None
    pad_top = int(rng.integers(min(3, 30-h)+1))
    pad_bottom = int(rng.integers(min(3, 30-h-pad_top)+1))
    pad_left = int(rng.integers(min(3, 30-w)+1))
    pad_right = int(rng.integers(min(3, 30-w-pad_left)+1))
    a = np.full((h+pad_top+pad_bottom, w+pad_left+pad_right), colors[0], dtype=np.int8)
    for cells, color in ((material, colors[1]), (ink, colors[2]), (guides, colors[3])):
        for y, x in cells:
            a[y-t+pad_top, x-l+pad_left] = color
    return a


def _alternative(parsed, kind):
    """Named plausible shortcuts, used solely for teaching checks."""
    material, center, mover = _assembled(parsed)
    colors, length = parsed["colors"], parsed["length"]
    if kind == "no_registration":
        material = parsed["anchor"] | parsed["mover"]
        center = parsed["center"]
    elif kind == "wrong_seam":
        material, center, mover = _assembled(parsed, 1)
    elif kind == "same_line":
        material, center, mover = _assembled(parsed, 0)
    elif kind == "seed_sheet_only":
        material = parsed["anchor"] if parsed["seed_owner"] == 0 else mover
    elif kind == "no_winding":
        def seed_only(shape, pos, n):
            mask = np.zeros(shape, dtype=bool)
            y, x = pos
            for d in range(1, n+1):
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    mask[y+d*dy, x+d*dx] = True
            return mask
        return _print_surface(material, center, length, colors, seed_only)
    elif kind in ("erase_guides", "paint_guides", "xor", "opaque_anchor", "opaque_mover"):
        if kind == "erase_guides":
            material -= parsed["bars"][0] | _shift(parsed["bars"][1], parsed["delta"])
        elif kind == "paint_guides":
            material |= parsed["bars"][0] | _shift(parsed["bars"][1], parsed["delta"])
        elif kind == "xor":
            material = parsed["anchor"] ^ mover
        else:
            front, back = (parsed["anchor"], mover) if kind == "opaque_anchor" else (mover, parsed["anchor"])
            t, b, l, r = _bbox(front)
            material = front | {(y, x) for y, x in back if not (t <= y <= b and l <= x <= r)}
        # Retain a seed even if the deliberately wrong material mask erased it.
        # Ink belongs to the field, so a missing support pixel remains missing.
    elif kind == "full_rectangle":
        t, b, l, r = _bbox(material)
        material = {(y, x) for y in range(t, b+1) for x in range(l, r+1)}
    elif kind == "opposite_hand":
        def reflected(shape, pos, n):
            return np.fliplr(_field(shape, (pos[0], shape[1]-1-pos[1]), n))
        return _print_surface(material, center, length, colors, reflected)
    elif kind == "first_exit":
        return _print_surface(material, center, length, colors,
                              lambda shape, pos, n: _trace_field(shape, pos, n, True))
    elif kind == "stop_at_gap":
        t, b, l, r = _bbox(material)
        support = np.zeros((b-t+1, r-l+1), dtype=bool)
        for y, x in material:
            support[y-t, x-l] = True
        return _print_surface(material, center, length, colors,
                              lambda shape, pos, n: _trace_field(shape, pos, n, support=support))
    elif kind == "ink_crop":
        result = _print_surface(material, center, length, colors)
        t, b, l, r = _bbox({tuple(map(int, q)) for q in np.argwhere(result == colors[2])})
        return result[t:b+1, l:r+1].copy()
    return _print_surface(material, center, length, colors)


_SHORTCUTS = ("no_registration", "wrong_seam", "same_line", "seed_sheet_only", "no_winding",
              "erase_guides", "paint_guides", "xor", "opaque_anchor", "opaque_mover",
              "full_rectangle", "opposite_hand", "first_exit", "stop_at_gap", "ink_crop")


def _different(parsed, output, kind):
    try:
        alternative = _alternative(parsed, kind)
    except (ValueError, IndexError):
        return True  # A shortcut producing an illegal crop is already refuted.
    return not np.array_equal(alternative, output)


def _make_pair(rng, profile, colors, used):
    bar = {(0, x) for x in range(profile["bar_length"])}
    for attempt in range(240):
        side = int(rng.integers(2))
        a, ai, ac = _piece(rng, profile["length"], profile["bar_length"], -1,
                          profile["owner"] == 0, profile["compact"], profile["teaching"],
                          profile["reach"], profile["overlap"], side)
        b, bi, bc = _piece(rng, profile["length"], profile["bar_length"], 1,
                          profile["owner"] == 1, profile["compact"], profile["teaching"],
                          profile["reach"], profile["overlap"], side)
        if len(_components(a | bar)) != 1 or len(_components(b | bar)) != 1:
            continue
        center = ac if ac is not None else (bc[0]-1, bc[1])
        material = a | _shift(b, (-1, 0))
        try:
            output = _print_surface(material, center, profile["length"], colors, _trace_field)
        except ValueError:
            continue
        mode = profile["mode"]
        # The preferred staging has plentiful feasible shapes; try another
        # layout only after a bounded search, preserving every semantic axis.
        if attempt >= 160:
            mode = "horizontal" if attempt % 2 == 0 else "vertical"
        input_grid = _stage(rng, (a, b), (bar, bar), (ai, bi), colors, mode)
        if input_grid is None:
            continue
        pair = {"input": np.rot90(input_grid, profile["rotation"]).copy(),
                "output": np.rot90(output, profile["rotation"]).copy()}
        if _key(pair["input"]) in used:
            continue
        try:
            parsed = _parse(pair["input"])
        except ValueError:
            continue
        joined, cy, moved = _assembled(parsed)
        if not np.array_equal(solve(pair["input"]), pair["output"]):
            raise AssertionError("interval solver and constructed printing disagree")
        t, bottom, left, right = _bbox(joined)
        f = _field((bottom-t+1, right-left+1), (cy[0]-t, cy[1]-left), profile["length"])
        other = moved if parsed["seed_owner"] == 0 else parsed["anchor"]
        own = parsed["anchor"] if parsed["seed_owner"] == 0 else moved
        visible_other_ink = sum(bool(f[y-t, x-left]) for y, x in other-own)
        if visible_other_ink < 4 or int(np.sum(pair["output"] == colors[2])) <= 4*profile["length"]+8:
            continue
        if profile["overlap"] and (not parsed["anchor"] & moved or not
                (parsed["bars"][0] | _shift(parsed["bars"][1], parsed["delta"])) - joined):
            continue
        if profile["teaching"] and not all(_different(parsed, pair["output"], s)
                                           for s in profile["distinguish"]):
            continue
        return pair
    raise RuntimeError("bounded cutout construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.random() < .45)
    rotation = int(rng.integers(4))
    mixed_axis = bool(rng.random() < .5)
    regime = str(rng.choice(("staging", "compact", "reach")))
    if regime == "staging":
        mixed_axis = False  # Hold out staging in a common visible coordinate frame.
    training_mode = str(rng.choice(("horizontal", "vertical")))
    lengths = [2, 3]
    if rng.integers(2):
        lengths.reverse()
    lengths.extend(int(rng.choice((1, 4))) if i == 2 else int(rng.integers(1, 5))
                   for i in range(2, ntrain))
    lengths.extend(int(rng.choice(lengths[:ntrain])) for _ in range(ntest))
    first_owner = int(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for i, length in enumerate(lengths):
        training = i < ntrain
        teaching = training
        colors = palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
        owner = first_owner if i == 0 else 1-first_owner if i == 1 else int(rng.integers(2))
        overlap = i == 1 or (i > 1 and bool(rng.random() < .65))
        if not training and regime == "reach":
            overlap = True
        mode = training_mode
        if regime == "staging" and not training:
            mode = "vertical" if training_mode == "horizontal" else "horizontal"
        elif i > 1 and regime != "staging":
            mode = str(rng.choice(("horizontal", "vertical", "diagonal")))
        distinguish = ("no_registration", "wrong_seam", "same_line", "seed_sheet_only", "no_winding", "erase_guides",
                       "full_rectangle", "opposite_hand", "first_exit", "stop_at_gap")
        if i == 1:
            distinguish += ("xor", "opaque_anchor", "opaque_mover", "paint_guides", "ink_crop")
        scene_rotation = (rotation + i % 2) % 4 if mixed_axis else rotation
        # Half turns reverse which backing side is the canonical anchor.
        canonical_owner = owner ^ int(scene_rotation >= 2)
        profile = dict(length=length, bar_length=int(rng.integers(3, 6)) if overlap
                       else int(rng.integers(2, 6)), owner=canonical_owner, teaching=teaching,
                       overlap=overlap, compact=not training and regime == "compact",
                       reach=not training and regime == "reach", mode=mode,
                       rotation=scene_rotation,
                       distinguish=distinguish)
        pair = _make_pair(rng, profile, colors, used)
        used.add(_key(pair["input"]))
        puzzle["train" if training else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    p = _parse(pair["input"])
    surface, center, moved = _assembled(p)
    bars = p["bars"][0] | _shift(p["bars"][1], p["delta"])
    t, b, l, r = _bbox(surface)
    bg, paper, ink, marker = p["colors"]
    return dict(length=p["length"], axis=p["axis"], bar_length=p["bar_length"],
                owner=p["seed_owner"], palette=p["colors"], overlap=len(p["anchor"] & moved),
                uncovered=len(bars-surface), covered=len(bars & surface),
                missing=(b-t+1)*(r-l+1)-len(surface), material=len(surface),
                ink=int(np.sum(np.asarray(pair["output"]) == ink)),
                input_shape=tuple(np.asarray(pair["input"]).shape),
                output_shape=tuple(np.asarray(pair["output"]).shape),
                delta=p["delta"], center=center)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, info = set(), {"train": [], "test": []}
    exposed = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            p = _parse(a)
            surface, center, _ = _assembled(p)
            expected = _print_surface(surface, center, p["length"], p["colors"])
            assert np.array_equal(expected, b), "incorrect registration, support, or winding"
            bg, paper, ink, marker = p["colors"]
            assert set(map(int, np.unique(b))) <= {bg, paper, ink}
            t, bottom, left, right = _bbox(surface)
            assert b.shape == (bottom-t+1, right-left+1)
            assert int(np.sum(b != bg)) == len(surface), "paper union must be conserved"
            assert b[center[0]-t, center[1]-left] == paper, "hollow center must remain paper"
            for d in range(1, p["length"]+1):
                for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                    assert b[center[0]-t+d*dy, center[1]-left+d*dx] == ink
            f = _features(pair)
            assert f["ink"] > 4*p["length"]+8 and f["missing"] > 0
            assert _different(p, b, "no_registration") and _different(p, b, "no_winding")
            info[split].append(f)
            if split == "train":
                exposed |= {kind for kind in _SHORTCUTS if _different(p, b, kind)}
    assert set(_SHORTCUTS) <= exposed, "training must distinguish all named shortcuts"
    assert {2, 3} <= {f["length"] for f in info["train"]}
    assert {0, 1} == {f["owner"] for f in info["train"]}
    assert any(f["overlap"] == 0 and f["uncovered"] == 0 for f in info["train"])
    assert any(f["overlap"] > 0 and f["uncovered"] > 0 for f in info["train"])
    assert {f["length"] for f in info["test"]} <= {f["length"] for f in info["train"]}


def self_test():
    # Procedural and interval calculations, symmetries, short-spoke behavior,
    # and oversized-plane crops check geometry beyond sampled cutout shapes.
    for length in range(1, 5):
        for shape, center in (((30, 30), (length, length)),
                              ((17, 29), (12, 7)), ((29, 13), (9, 6))):
            f = _field(shape, center, length)
            assert np.array_equal(f, _trace_field(shape, center, length))
            assert np.array_equal(np.rot90(f), _field(shape[::-1],
                                  (shape[1]-1-center[1], center[0]), length))
            big = _trace_field((80, 80), (40, 40), length)
            t, l = 40-center[0], 40-center[1]
            assert np.array_equal(f, big[t:t+shape[0], l:l+shape[1]])
    assert np.array_equal(_field((30, 30), (15, 15), 1), _field((30, 30), (15, 15), 2))
    shapes, palettes, layouts, signatures = set(), set(), set(), set()
    lengths, axes, bars, owners, overlaps, gaps = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        signature = tuple(_key(pair[role]) for split in ("train", "test")
                          for pair in p[split] for role in ("input", "output"))
        assert signature == tuple(_key(pair[role]) for split in ("train", "test")
                                  for pair in q[split] for role in ("input", "output"))
        signatures.add(signature)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                shapes.add((f["input_shape"], f["output_shape"]))
                palettes.add(f["palette"])
                lengths.add(f["length"])
                axes.add(f["axis"])
                bars.add(f["bar_length"])
                owners.add(f["owner"])
                overlaps.add(f["overlap"])
                gaps.add(f["missing"])
                a = np.asarray(pair["input"])
                remap = {}
                layouts.add(tuple(remap.setdefault(int(c), len(remap)) for c in a.ravel()))
                assert np.array_equal(solve(a.copy()), pair["output"])
    assert len(signatures) == 42 and len(shapes) > 100 and len(layouts) > 180
    assert len(palettes) > 50 and lengths == {1, 2, 3, 4} and axes == owners == {0, 1}
    assert bars == {2, 3, 4, 5} and 0 in overlaps and max(overlaps) >= 6 and len(gaps) > 40
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1701+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Verify validator rejects an answer mutation, rather than just schema.
    bad = generate_puzzle(0, 2, 1)
    a = bad["test"][0]["output"]
    a[0, 0] = (int(a[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong ink pixel accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shape_pairs=len(shapes),
                normalized_layouts=len(layouts), palettes=len(palettes),
                lengths=sorted(lengths), bar_lengths=sorted(bars),
                overlap_counts=sorted(overlaps), missing_paper_counts=len(gaps))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
