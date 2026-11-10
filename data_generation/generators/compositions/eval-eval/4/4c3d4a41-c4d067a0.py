"""A miniature-controlled loom: expand, lift whole columns on contact, clip.

The enlarged clue calibrates the loom, including the comb's insertion location.
All semantic information is recovered from pixels by solve; no parent module,
training labels, random state, or generation metadata is consulted.
"""

from collections import Counter

import numpy as np


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _unorient(a, code):
    b = np.rot90(a, -(code % 4))
    return (np.fliplr(b) if code >= 4 else b).copy()


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    found = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        todo, cells = [(y, x)], []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not visited[nr, nc]):
                    visited[nr, nc] = True
                    todo.append((nr, nc))
        found.append(cells)
    return found


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _matches(plan, sequence):
    n = len(sequence)
    return [(r, c) for r in range(plan.shape[0])
            for c in range(plan.shape[1] - n + 1)
            if tuple(int(v) for v in plan[r, c:c + n]) == sequence]


def _input_pixels(f):
    a = np.full(f["shape"], f["bg"], dtype=np.int8)
    t, b, l, r, base, s = (f[k] for k in ("roof", "floor", "left", "right", "base", "structure"))
    a[t, l:r + 1] = a[b, l:r + 1] = s
    a[t:b + 1, l] = a[t:b + 1, r] = s
    a[base - 1:base + 1, l] = f["bg"]
    kl, width, side, stride = (f[k] for k in ("key_left", "width", "side", "stride"))
    a[base, kl:kl + width] = s
    for j, h in enumerate(f["heights"]):
        x = kl + j * stride
        a[base - h:base, x:x + side] = s
    mr, mc = f["mini"]
    rows, cols = f["plan"].shape
    a[mr:mr + 2 * rows:2, mc:mc + 2 * cols:2] = f["plan"]
    rr, cc = f["match"]
    oy, ox = f["origin"]
    for k in range(f["length"]):
        y, x = oy + rr * stride, ox + (cc + k) * stride
        a[y:y + side, x:x + side] = f["plan"][rr, cc + k]
    return a


def _shifts(f):
    rows, cols = f["plan"].shape
    bottom_rows = [int(np.flatnonzero(f["plan"][:, j] != f["bg"])[-1]) for j in range(cols)]
    return tuple(max(0, f["origin"][0] + r * f["stride"] + f["side"] - 1
                     - (f["base"] - h - 1))
                 for r, h in zip(bottom_rows, f["heights"]))


def _construct(f):
    """Direct construction of labels from sampled dimensions and cell formulas."""
    inp = _input_pixels(f)
    out = inp.copy()
    s, bg, base = f["structure"], f["bg"], f["base"]
    kl, side, stride, ox = f["key_left"], f["side"], f["stride"], f["origin"][1]
    for j, h in enumerate(f["heights"]):
        out[base - h:base, kl + j * stride:kl + j * stride + side] = bg
    out[base, kl:kl + f["width"]] = bg
    # Remove the unshifted sample before printing its complete, deformed plan.
    rr, cc = f["match"]
    for j in range(cc, cc + f["length"]):
        y, x = f["origin"][0] + rr * stride, ox + j * stride
        out[y:y + side, x:x + side] = bg
    shifts = _shifts(f)
    for (i, j), color in np.ndenumerate(f["plan"]):
        if color == bg:
            continue
        x = ox + j * stride
        for dy in range(side):
            y = f["origin"][0] + i * stride + dy - shifts[j]
            if y > f["roof"]:
                out[y, x:x + side] = color
    out[base, ox:ox + f["width"]] = s
    for j, h in enumerate(f["heights"]):
        x = ox + j * stride
        out[base - h:base, x:x + side] = s
    return {"input": _orient(inp, f["orientation"]), "output": _orient(out, f["orientation"])}


def _walls(a, bg):
    runs = {}
    for y, row in enumerate(a):
        boundaries = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), len(row)]
        for l, stop in zip(boundaries, boundaries[1:]):
            color = int(row[l])
            if color != bg and stop - l >= 7:
                runs.setdefault((color, l, stop - 1), []).append(y)
    return [(s, l, r, t, b) for (s, l, r), ys in runs.items()
            for t in ys for b in ys if b > t]


def _read(a, bg, wall):
    s, l, r, t, b = wall
    assert np.all(a[t:b + 1, r] == s)
    opening = np.flatnonzero(a[t + 1:b, l] != s) + t + 1
    assert len(opening) == 2 and opening[1] == opening[0] + 1
    assert np.all(a[opening, l] == bg)
    base = int(opening[1])
    assert b - base >= 2 and base - t >= 3

    # Color s is a geometric role, not a fixed literal. The remaining objects
    # are precisely isolated miniature swatches and a horizontal square run.
    comps = _components((a != bg) & (a != s))
    swatches = np.array([c[0] for c in comps if len(c) == 1], dtype=int)
    assert len(swatches) >= 3
    mr, mc = (int(v) for v in swatches.min(axis=0))
    er, ec = (int(v) for v in swatches.max(axis=0) + 1)
    assert np.all((swatches - (mr, mc)) % 2 == 0)
    assert np.any(np.diff(np.unique(swatches[:, 0])) == 2)
    assert np.any(np.diff(np.unique(swatches[:, 1])) == 2)
    plan = a[mr:er:2, mc:ec:2].copy()
    rows, cols = plan.shape
    assert min(rows, cols) >= 2 and np.all(np.any(plan != bg, axis=0))
    assert np.count_nonzero(plan != bg) == len(swatches)
    assert ec <= l - 1

    squares = []
    for cells in comps:
        if len(cells) == 1:
            continue
        points = np.array(cells)
        y, x = (int(v) for v in points.min(axis=0))
        ey, ex = (int(v) for v in points.max(axis=0) + 1)
        side = ey - y
        assert side >= 2 and ex - x == side and len(cells) == side * side
        assert np.all(a[y:ey, x:ex] == a[y, x])
        squares.append((y, x, side, int(a[y, x])))
    assert 2 <= len(squares) <= cols
    squares.sort(key=lambda q: q[1])
    sy, sx, side, _ = squares[0]
    assert all(q[0] == sy and q[2] == side for q in squares)
    stride = squares[1][1] - sx
    assert stride > side and all(q[1] == sx + k * stride for k, q in enumerate(squares))
    sequence = tuple(q[3] for q in squares)
    matches = _matches(plan, sequence)
    assert len(matches) == 1, "clue must uniquely locate a plan segment"
    rr, cc = matches[0]
    oy, ox = sy - rr * stride, sx - cc * stride
    width, height = (cols - 1) * stride + side, (rows - 1) * stride + side
    assert t < oy and oy + height <= base
    assert l < ox and ox + width <= r

    key_cells = np.flatnonzero(a[base, :l] == s)
    assert len(key_cells) == width
    kl = int(key_cells[0])
    assert np.array_equal(key_cells, np.arange(kl, kl + width))
    assert kl + width < l
    heights = []
    for j in range(cols):
        x, y = kl + j * stride, base - 1
        while y >= 0 and np.all(a[y, x:x + side] == s):
            y -= 1
        h = base - 1 - y
        assert 1 <= h <= base - t - 2
        heights.append(h)
    assert not _overlap((mr, mc, er, ec), (base - max(heights), kl, base + 1, kl + width))
    f = dict(shape=a.shape, bg=bg, structure=s, roof=t, floor=b, left=l, right=r,
             base=base, key_left=kl, side=side, stride=stride, width=width,
             mini=(mr, mc), plan=plan, origin=(oy, ox), match=(rr, cc),
             length=len(squares), heights=tuple(heights))
    assert np.array_equal(_input_pixels(f), a), "malformed scene or unrelated pixels"
    return f


def _parse(inp):
    counts = np.bincount(inp.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > np.partition(counts, -2)[-2], "background must be uniquely modal"
    candidates = []
    for code in range(8):
        a = _unorient(inp, code)
        for wall in _walls(a, bg):
            try:
                f = _read(a, bg, wall)
            except AssertionError:
                continue
            f["orientation"] = code
            candidates.append(f)
    if len(candidates) != 1:
        raise ValueError("input must have exactly one loom interpretation")
    return candidates[0]


def _simulate(inp, f, *, expand=True, push=True):
    """Input-derived pixel simulation; shift virtual strips one cell per step.

The expand/push switches are explicit concept-omission diagnostics. The ordinary
solver always enables both. This does not call the construction shift formula.
"""
    a = _unorient(inp, f["orientation"])
    out = a.copy()
    bg, s, roof, base = (f[k] for k in ("bg", "structure", "roof", "base"))
    kl, ox, side, stride = f["key_left"], f["origin"][1], f["side"], f["stride"]
    # Translate the actual external key pixels, preserving its silhouette.
    key_pixels = np.argwhere(a[:, kl:kl + f["width"]] == s)
    for y, local_x in key_pixels:
        out[y, kl + local_x] = bg
    for j, h in enumerate(f["heights"]):
        x = ox + j * stride
        strip = np.full((base - roof - 1, side), bg, dtype=np.int8)
        for i in range(f["plan"].shape[0]):
            selected = expand or (i == f["match"][0]
                                  and f["match"][1] <= j < f["match"][1] + f["length"])
            if selected:
                y = f["origin"][0] + i * stride - roof - 1
                strip[y:y + side] = f["plan"][i, j]
        occupied = np.flatnonzero(np.any(strip != bg, axis=1))
        if push and len(occupied):
            bottom = int(occupied[-1]) + roof + 1
            while bottom >= base - h:
                strip[:-1] = strip[1:].copy()
                strip[-1] = bg
                bottom -= 1
        out[roof + 1:base, x:x + side] = strip
    for y, local_x in key_pixels:
        out[y, ox + local_x] = s
    return _orient(out, f["orientation"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _simulate(inp, _parse(inp))


def _alignment_alternatives(f):
    """Plausible locations if clue colors are ignored, but occupancy is used."""
    rr, cc = f["match"]
    sy = f["origin"][0] + rr * f["stride"]
    sx = f["origin"][1] + cc * f["stride"]
    height = (f["plan"].shape[0] - 1) * f["stride"] + f["side"]
    alternatives = []
    for i in range(f["plan"].shape[0]):
        for j in range(f["plan"].shape[1] - f["length"] + 1):
            if (i, j) == (rr, cc) or np.any(f["plan"][i, j:j + f["length"]] == f["bg"]):
                continue
            oy, ox = sy - i * f["stride"], sx - j * f["stride"]
            if (f["roof"] < oy and oy + height <= f["base"]
                    and f["left"] < ox and ox + f["width"] <= f["right"]):
                alternatives.append(dict(f, match=(i, j), origin=(oy, ox)))
    return alternatives


def _geometry_options(side=None, rich=False):
    options = []
    for rows in range(3 if rich else 2, 6):
        for cols in range(3 if rich else 2, 5):
            for sz in ((side,) if side is not None else (2, 3)):
                for gutter in range(1, 4):
                    stride = sz + gutter
                    width, height = (cols - 1) * stride + sz, (rows - 1) * stride + sz
                    if 2 * width + 7 <= 30 and height + 6 <= 30:
                        options.append((rows, cols, sz, stride))
    return options


def _asymmetric(plan):
    variants = [np.flipud(plan), np.fliplr(plan), np.flip(plan)]
    if plan.shape[0] == plan.shape[1]:
        variants.extend((plan.T, np.flip(plan.T), np.rot90(plan), np.rot90(plan, 3)))
    return all(not np.array_equal(plan, other) for other in variants)


def _sample_plan(rng, rows, cols, bg, palette, length, anchor_kind, witness):
    for _ in range(120):
        plan = rng.choice(palette, size=(rows, cols)).astype(np.int8)
        # Blanks are sites, including boundary sites; repeated rows are legal.
        if rng.random() < .2 and rows >= 3:
            i, j = rng.choice(rows, 2, replace=False)
            plan[i] = plan[j]
        hole_count = int(rng.integers(0, max(2, rows * cols // 3) + 1))
        plan.flat[rng.choice(rows * cols, hole_count, replace=False)] = bg
        lift, rest = (int(v) for v in rng.choice(cols, 2, replace=False))
        if witness:
            plan[0, lift], plan[-1, lift] = palette[0], palette[1]
            plan[0, rest], plan[-1, rest] = palette[-1], bg
            # This void is between two occupied blocks in the moving strip.
            plan[int(rng.integers(1, rows - 1)), lift] = bg
        if (np.any(np.all(plan == bg, axis=0)) or np.all(plan[0] == bg)
                or np.all(plan[-1] == bg) or not _asymmetric(plan)):
            continue
        occupied = np.argwhere(plan != bg)
        if any(not np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
            continue
        choices = []
        for r in range(rows):
            if anchor_kind == "first" and r != 0:
                continue
            if anchor_kind == "last" and r != rows - 1:
                continue
            if anchor_kind == "boundary" and r not in (0, rows - 1):
                continue
            if anchor_kind == "interior" and r in (0, rows - 1):
                continue
            for c in range(cols - length + 1):
                seq = tuple(int(v) for v in plan[r, c:c + length])
                if bg not in seq and _matches(plan, seq) == [(r, c)]:
                    choices.append((r, c))
        if choices:
            match = choices[int(rng.integers(len(choices)))]
            return plan, match, lift, rest
    # No silent fallback to a different rule or shift regime.
    raise ValueError("resample geometry: no unique miniature clue")


def _sample(rng, orientation, side, anchor_kind, witness, flat=False, avoid_bg=None,
            need_alignment=False, short_clue=False):
    options = _geometry_options(side, rich=witness or anchor_kind == "interior")
    for _ in range(200):
        rows, cols, sz, stride = options[int(rng.integers(len(options)))]
        width, height = (cols - 1) * stride + sz, (rows - 1) * stride + sz
        colors = rng.permutation(10)
        if int(colors[0]) == avoid_bg:
            colors[0], colors[1] = colors[1], colors[0]
        bg, structure = int(colors[0]), int(colors[1])
        palette = colors[2:2 + int(rng.integers(2, 7))]
        length = 2 if short_clue else int(rng.integers(2, cols + 1))
        try:
            plan, match, lift, rest = _sample_plan(rng, rows, cols, bg, palette,
                                                  length, anchor_kind, witness)
        except ValueError:
            continue
        # Horizontal slack changes the full origin; there are no marked lanes
        # from which geometry alone could recover the clue's plan coordinates.
        budget = 30 - (2 * width + 7)
        kl = int(rng.integers(0, min(2, budget) + 1))
        budget -= kl
        gap = 1 + int(rng.integers(0, min(2, budget) + 1))
        budget -= gap - 1
        inset_l = 1 + int(rng.integers(0, min(2, budget) + 1))
        budget -= inset_l - 1
        inset_r = 1 + int(rng.integers(0, min(2, budget) + 1))
        l = kl + width + gap
        ox, r = l + 1 + inset_l, l + 2 + inset_l + width + inset_r
        roof = int(rng.integers(0, 3))
        top_gap = int(rng.integers(1, 3 if witness else 4))
        lower_gap = int(rng.integers(1, 6))
        oy, base = roof + top_gap, roof + top_gap + height + lower_gap
        floor = base + int(rng.integers(2, 4))
        if floor >= 30:
            continue
        mini_h, mini_w = 2 * rows - 1, 2 * cols - 1
        below_possible = base + 2 + mini_h <= 30
        below = below_possible and rng.random() < .35
        hcap = min(8, base - roof - 2)
        if not below:
            hcap = min(hcap, base - mini_h - 1)
        required_h = lower_gap + top_gap
        if hcap < max(2, required_h) or l - mini_w - 1 < 0:
            continue
        heights = rng.integers(1, hcap + 1, size=cols).tolist()
        if witness:
            heights[lift] = heights[rest] = required_h
            if not flat:
                other = next(j for j in range(cols) if j not in (lift, rest))
                heights[other] = int(rng.choice([h for h in range(1, hcap + 1) if h != required_h]))
        if flat:
            heights = [required_h if witness else int(rng.integers(1, hcap + 1))] * cols
        if below:
            mr = int(rng.integers(base + 2, 31 - mini_h))
        else:
            mr = int(rng.integers(0, base - max(heights) - mini_h))
        mc = int(rng.integers(0, l - mini_w))
        used_h, used_w = max(floor + 1, mr + mini_h), r + 1
        shape = (int(rng.integers(used_h, min(30, used_h + 3) + 1)),
                 int(rng.integers(used_w, min(30, used_w + 3) + 1)))
        f = dict(shape=shape, bg=bg, structure=structure, roof=roof, floor=floor,
                 left=l, right=r, base=base, key_left=kl, side=sz, stride=stride,
                 width=width, mini=(mr, mc), plan=plan, origin=(oy, ox), match=match,
                 length=length, heights=tuple(int(h) for h in heights), orientation=orientation)
        pair = _construct(f)
        # Every pair must actually need missing plan material after clipping.
        if np.array_equal(pair["output"], _simulate(pair["input"], f, expand=False)):
            continue
        if need_alignment and not any(
                not np.array_equal(pair["output"], _simulate(pair["input"], alt))
                for alt in _alignment_alternatives(f)):
            continue
        return f, pair
    raise RuntimeError("could not sample a fitting loom within bounded retries")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("anchor", "flat")))
    orientation = int(rng.integers(8))
    axes = [orientation]
    if rng.random() < .35:
        axes.append((orientation + int(rng.integers(1, 8))) % 8)
    first_side = int(rng.choice((2, 3)))
    first_boundary = str(rng.choice(("first", "last")))
    # With the widest teeth, horizontal slack is too small for two origins;
    # a bottom clue leaves vertical slack to demonstrate color disambiguation.
    if shift == "anchor" and first_side == 3:
        first_boundary = "last"
    result, seen, first_bg = {"train": [], "test": []}, set(), None
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            witness = i < 2 if split == "train" else i == 0
            side = first_side if split == "train" and i == 0 else (
                5 - first_side if split == "train" and i == 1 else None)
            if shift == "anchor":
                anchor = "boundary" if split == "train" else "interior"
                if split == "train" and i < 2:
                    anchor = first_boundary if i == 0 else ("last" if first_boundary == "first" else "first")
            else:
                anchor = "any"
            flat = split == "test" and shift == "flat"
            code = axes[i % len(axes)] if split == "train" else int(rng.choice(axes))
            for _ in range(40):
                f, pair = _sample(rng, code, side, anchor, witness, flat,
                                  first_bg if split == "train" and i == 1 else None,
                                  need_alignment=split == "train" and i == 0,
                                  short_clue=split == "train" and i < 2)
                if shift == "flat" and split == "train" and len(set(f["heights"])) == 1:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not obtain distinct examples")
            seen.add(key)
            result[split].append(pair)
            if split == "train" and i == 0:
                first_bg = f["bg"]
        rng.shuffle(result[split])
    return result


def _features(f):
    shifts = _shifts(f)
    partial, clipped = False, 0
    for (i, j), color in np.ndenumerate(f["plan"]):
        if color == f["bg"]:
            continue
        top = f["origin"][0] + i * f["stride"] - shifts[j]
        loss = min(f["side"], max(0, f["roof"] + 1 - top))
        clipped += loss * f["side"]
        partial |= 0 < loss < f["side"]
    return dict(shape=f["shape"], plan_shape=f["plan"].shape, side=f["side"],
                gutter=f["stride"] - f["side"], length=f["length"], match=f["match"],
                heights=f["heights"], shifts=shifts, orientation=f["orientation"],
                bg=f["bg"], structure=f["structure"], holes=int(np.sum(f["plan"] == f["bg"])),
                clipped=clipped, partial=partial,
                stationary=0 in shifts, moving=any(v > 0 for v in shifts),
                tied_different=any(f["heights"][j] == f["heights"][k] and shifts[j] != shifts[k]
                                   for j in range(len(shifts)) for k in range(j)),
                bottom_holes=bool(np.any(f["plan"][-1] == f["bg"])),
                mini_below=f["mini"][0] > f["base"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, demonstrations = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            f = _parse(inp)
            assert np.array_equal(out, _simulate(inp, f)), "wrong material, movement, or preserved pixel"
            a, b = _unorient(inp, f["orientation"]), _unorient(out, f["orientation"])
            t, l, r = f["roof"], f["left"], f["right"]
            assert np.array_equal(a[t, l:r + 1], b[t, l:r + 1]), "roof cannot be overwritten"
            mr, mc = f["mini"]
            mh, mw = (2 * n - 1 for n in f["plan"].shape)
            assert np.array_equal(a[mr:mr + mh, mc:mc + mw], b[mr:mr + mh, mc:mc + mw])
            assert not np.array_equal(inp, out)
            assert not np.array_equal(out, _simulate(inp, f, expand=False)), "B is redundant in this pair"
            if split == "train":
                meta = _features(f)
                meta["needs_push"] = not np.array_equal(out, _simulate(inp, f, push=False))
                meta["needs_matching"] = any(not np.array_equal(out, _simulate(inp, alt))
                                              for alt in _alignment_alternatives(f))
                demonstrations.append(meta)
    assert any(d["needs_push"] for d in demonstrations), "training must expose A"
    assert any(d["needs_matching"] for d in demonstrations), "colors must resolve alignment"
    assert any(d["partial"] for d in demonstrations), "show roof cutting through a square"
    assert any(d["tied_different"] for d in demonstrations), "show blank-sensitive contact"
    assert any(d["stationary"] and d["moving"] for d in demonstrations)
    assert len({d["side"] for d in demonstrations}) >= 2, "teach measured scale"
    assert len({d["bg"] for d in demonstrations}) >= 2, "teach geometric color roles"


def _focused_checks():
    # A hand-specified loom, outside the sampler's teaching schedule. Equal
    # teeth lift lane 0 by two but leave lane 1 suspended across a gap.
    f = dict(shape=(20, 27), bg=0, structure=8, roof=0, floor=14, left=10,
             right=25, base=12, key_left=0, side=2, stride=3, width=8,
             mini=(0, 0), origin=(2, 12), match=(0, 0), length=2,
             heights=(4, 4, 1), orientation=0,
             plan=np.array([[2, 3, 4], [4, 0, 2], [3, 0, 2]], dtype=np.int8))
    inp = _input_pixels(f)
    expected = inp.copy()
    expected[12, :8] = 0
    expected[8:12, :2] = expected[8:12, 3:5] = 0
    expected[11, 6:8] = 0
    expected[2:4, 12:17] = 0
    expected[1, 12:14] = 2
    expected[3:5, 12:14] = 4
    expected[6:8, 12:14] = 3
    expected[2:4, 15:17] = 3
    expected[2:4, 18:20] = 4
    expected[5:7, 18:20] = expected[8:10, 18:20] = 2
    expected[12, 12:20] = 8
    expected[8:12, 12:14] = expected[8:12, 15:17] = 8
    expected[11, 18:20] = 8
    assert np.array_equal(solve(inp), expected)
    assert _shifts(_parse(inp)) == (2, 0, 0)
    # Every rigid whole-scene transform and a nontrivial bijective recoloring.
    recolor = np.array([(3 * i + 1) % 10 for i in range(10)], dtype=np.int8)
    for code in range(8):
        assert np.array_equal(solve(_orient(recolor[inp], code)), _orient(recolor[expected], code))
    stationary = dict(f, heights=(1, 1, 1))
    assert _shifts(_parse(_input_pixels(stationary))) == (0, 0, 0)
    assert np.array_equal(solve(_input_pixels(stationary)), _construct(stationary)["output"])
    # Deliberately taller than the sampler's cap: only one pixel row of the
    # bottom occupied square survives in each lane. Semantic reading is broader
    # than the routine sampler, and the roof is never treated as a payload cell.
    severe = dict(f, shape=(22, 27), mini=(16, 0), heights=(10, 10, 10))
    result = solve(_input_pixels(severe))
    assert np.array_equal(result[1, 12:20], np.array([3, 3, 0, 3, 3, 0, 2, 2]))
    assert np.all(result[0, 10:26] == 8)
    # A fully empty interior plan row is admitted when adjacent visible rows
    # elsewhere still establish miniature pitch. It moves as a void, not a
    # compression instruction.
    void = dict(f, shape=(22, 27), base=15, floor=17,
                plan=np.array([[2, 3, 4], [4, 4, 3], [0, 0, 0], [3, 0, 2]], dtype=np.int8))
    result = solve(_input_pixels(void))
    assert np.all(result[6:8, 12:14] == 0)
    assert np.array_equal(result, _construct(void)["output"])
    ambiguous = dict(f, plan=f["plan"].copy())
    ambiguous["plan"][1] = ambiguous["plan"][0]
    broken_wall = inp.copy()
    broken_wall[0, 14] = 0
    for bad in (_input_pixels(ambiguous), broken_wall):
        try:
            solve(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("ambiguous or broken loom accepted")


def self_test():
    """Bounded semantic checks and measured diversity on exactly 42 seeds."""
    _focused_checks()
    records, signatures, normalized, shapes = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                assert np.array_equal(solve(inp), pair["output"])
                f = _parse(inp)
                records.append(dict(split=split, **_features(f)))
                shapes.add(inp.shape)
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
    assert len(signatures) == 42 and len(normalized) >= 180 and len(shapes) >= 35
    assert {r["side"] for r in records} == {2, 3}
    assert {r["gutter"] for r in records} == {1, 2, 3}
    assert {r["orientation"] for r in records} == set(range(8))
    assert {r["bg"] for r in records} == set(range(10))
    assert len({r["plan_shape"] for r in records}) >= 9
    assert len({r["heights"] for r in records}) >= 70
    assert len({r["match"] for r in records if r["split"] == "test"}) >= 6
    assert any(r["length"] >= 3 for r in records)
    assert any(r["mini_below"] for r in records)
    assert any(r["holes"] == 0 for r in records)
    assert any(r["holes"] >= 4 for r in records)
    assert any(not r["moving"] for r in records)
    assert any(not r["stationary"] for r in records)
    assert any(r["clipped"] == 0 for r in records)
    for nt, ns in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(987, nt, ns)
        validate_generated_puzzle(p)
        q = generate_puzzle(987, nt, ns)
        assert all(np.array_equal(x[k], y[k]) for s in p
                   for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # A corrupted miniature match, wall, or target pixel must not pass validation.
    p = generate_puzzle(0, 2, 1)
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_shapes": len(shapes),
            "normalized_structures": len(normalized),
            "plan_shapes": sorted({r["plan_shape"] for r in records}),
            "partial_clips": sum(r["partial"] for r in records),
            "tied_teeth_different_motion": sum(r["tied_different"] for r in records),
            "seed_lengths": dict(sorted(Counter(r["length"] for r in records).items()))}
