"""Corner loom: register complementary corner rubbings, then complete copies.

Only standard Python and NumPy are used. No files, examples, RNG state or
generation metadata are consulted by solve. See task_spec.md for the rule.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_HOST_SEEDS = (0, 367475143, 847938840, 808266797, 119699774, 991460576,
               929157402, 349243174, 173397644, 728683517, 628623373,
               680236997, 588167740, 117015124, 937452421, 94631525,
               621275822, 676554204, 454283746, 616363327, 30497557,
               353311438, 561546031, 965981326, 924983664, 165021905,
               209655941, 934667625, 782362811, 552127010, 800916188,
               185750217, 456149982, 205715281, 108917236, 156862206,
               537736613, 179731834, 372667687, 229877077, 402838781,
               341311693)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(points):
    if not points:
        return False
    seen = {min(points)}
    stack = list(seen)
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = (y + dy, x + dx)
                if p in points and p not in seen:
                    seen.add(p)
                    stack.append(p)
    return seen == set(points)


def _shift(points, delta):
    dy, dx = delta
    return {(y + dy, x + dx) for y, x in points}


def _axis(points, axis):
    return {(p[axis], p[1 - axis]) for p in points}


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, l, 2), (b, r, 3))


def _outline(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _local(corner, points):
    y, x, q = corner
    dy, dx = _DIRS[q]
    return {((r - y) * dy, (c - x) * dx) for r, c in points}


def _world(corner, points):
    y, x, q = corner
    dy, dx = _DIRS[q]
    return {(y + u * dy, x + v * dx) for u, v in points}


def _geometry(a, bg, frame):
    """Recover closed frames from two protected pixels on each incident arm.

    Radius-three marks may cover corners, but never the arm cells at distances
    four and five. This routine does not use the intended stencil or output.
    """
    h, w = a.shape
    corners = set()
    for y, x in np.argwhere(a != bg):
        y, x = int(y), int(x)
        for q, (dy, dx) in enumerate(_DIRS):
            if not (0 <= y + 5 * dy < h and 0 <= x + 5 * dx < w):
                continue
            if all(a[y + k * dy, x] == frame and a[y, x + k * dx] == frame
                   for k in (4, 5)):
                if all(a[y + k * dy, x] != bg and a[y, x + k * dx] != bg
                       for k in (0, 1, 2, 3)):
                    corners.add((y, x, q))
    rects = set()
    for t, l, q in sorted(corners):
        if q != 0:
            continue
        for y, r, qr in corners:
            if qr != 1 or y != t or r - l < 10:
                continue
            for b, x, qb in corners:
                if qb == 2 and x == l and b - t >= 10 and (b, r, 3) in corners:
                    rects.add((t, l, b, r))
    assert 1 <= len(rects) <= 2
    expected = {c for rect in rects for c in _corners(rect)}
    assert expected == corners and len(expected) == 4 * len(rects)
    outline = set()
    for rect in rects:
        edge = _outline(rect)
        assert not outline & edge
        assert all(a[p] != bg for p in edge)
        for p in edge:
            if a[p] != frame:
                assert min(max(abs(p[0] - y), abs(p[1] - x))
                           for y, x, _ in _corners(rect)) <= 3
        outline |= edge
    assert {tuple(map(int, p)) for p in np.argwhere(a == frame)} <= outline
    return sorted(rects), sorted(corners), outline


def _bar(points):
    """Return visible normal axis, normal coordinate, first tangent, length."""
    if not 2 <= len(points) <= 4:
        return None
    for axis in (0, 1):
        nt = _axis(points, axis)
        ns, ts = {n for n, _ in nt}, {t for _, t in nt}
        if len(ns) == 1 and max(ts) - min(ts) + 1 == len(points):
            return axis, next(iter(ns)), min(ts), len(points)
    return None


def _interpret(a):
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(np.argmax(counts))
    answers = []
    colors = list(map(int, np.unique(a)))
    for frame in colors:
        if frame == bg or counts[frame] < 12:
            continue
        try:
            rects, corners, edge = _geometry(a, bg, frame)
            assigned = {c: {} for c in corners}
            for y, x in np.argwhere((a != bg) & (a != frame)):
                y, x = int(y), int(x)
                ds = [(y - r) ** 2 + (x - c) ** 2 for r, c, _ in corners]
                best = min(ds)
                assert ds.count(best) == 1
                corner = corners[ds.index(best)]
                assert max(abs(y - corner[0]), abs(x - corner[1])) <= 3
                color = int(a[y, x])
                assigned[corner].setdefault(color, set()).add((y, x))
            for marker in colors:
                if marker in (bg, frame):
                    continue
                try:
                    donors = [c for c in corners if marker in assigned[c]]
                    assert len(donors) == 2
                    records, ink_colors = [], set()
                    for c in donors:
                        parts = assigned[c]
                        assert len(parts) == 2
                        ink = next(k for k in parts if k != marker)
                        ink_colors.add(ink)
                        patch = _local(c, parts[ink])
                        guide = _local(c, parts[marker])
                        bar = _bar(guide)
                        assert bar is not None
                        axis, n, t, length = bar
                        nt = _axis(patch, axis)
                        neg = any((n - 1, x) in nt for x in range(t, t + length))
                        pos = any((n + 1, x) in nt for x in range(t, t + length))
                        assert neg != pos and _connected(patch | guide)
                        records.append({"corner": c, "ink": ink, "patch": patch,
                                        "guide": guide, "bar": bar, "negative": neg})
                    assert len(ink_colors) == 1
                    assert records[0]["negative"] != records[1]["negative"]
                    records.sort(key=lambda d: not d["negative"])
                    anchor, mover = records
                    axis, an, at, length = anchor["bar"]
                    ma, mn, mt, ml = mover["bar"]
                    assert ma == axis and ml == length
                    delta_nt = (an - 1 - mn, at - mt)
                    delta = delta_nt if axis == 0 else delta_nt[::-1]
                    assert delta != (0, 0)
                    p = anchor["patch"]
                    shifted = _shift(mover["patch"], delta)
                    stencil = p | shifted
                    assert p - shifted and shifted - p
                    assert 4 <= len(stencil) <= 24 and _connected(stencil)
                    assert max(max(abs(u), abs(v)) for u, v in stencil) <= 3
                    marks = []
                    for c in corners:
                        parts = assigned[c]
                        if not parts:
                            continue
                        inks = [k for k in parts if k != marker]
                        assert len(inks) == 1
                        ink = inks[0]
                        if c not in donors:
                            assert len(parts) == 1
                            clue = _local(c, parts[ink])
                            assert clue < stencil
                        full = _world(c, stencil)
                        assert all(0 <= y < a.shape[0] and 0 <= x < a.shape[1]
                                   for y, x in full)
                        for y, x in full:
                            ds = [(y - r) ** 2 + (x - s) ** 2 for r, s, _ in corners]
                            assert ds.count(min(ds)) == 1 and ds[corners.index(c)] == min(ds)
                        marks.append((c, ink))
                    assert 3 <= len(marks) <= 8
                    assert len(marks) > len(donors)
                    info = {"background": bg, "frame": frame, "marker": marker,
                            "rectangles": rects, "corners": corners, "edge": edge,
                            "marks": marks, "anchor": anchor, "mover": mover,
                            "delta": delta, "axis": axis, "length": length,
                            "p": p, "q": shifted, "stencil": stencil}
                    answers.append(info)
                except (AssertionError, StopIteration):
                    continue
        except AssertionError:
            continue
    assert len(answers) == 1, "missing or ambiguous frame / registration interpretation"
    return answers[0]


def _render(shape, info, stencil=None, convention="reflection"):
    s = info["stencil"] if stencil is None else stencil
    out = np.full(shape, info["background"], dtype=np.int8)
    for p in info["edge"]:
        out[p] = info["frame"]
    for corner, color in info["marks"]:
        local = s
        if convention == "rotation" and corner[2] in (1, 2):
            local = {(v, u) for u, v in s}
        points = (_shift(local, corner[:2]) if convention == "global"
                  else _world(corner, local))
        for y, x in points:
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                out[y, x] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the full answer from the query pixels alone."""
    a = _grid(input_grid)
    return _render(a.shape, _interpret(a))


def _grow(rng, domain, guide, back, lo, hi):
    choices = sorted(back & domain)
    if not choices:
        return None
    points = {choices[int(rng.integers(len(choices)))]}
    target = int(rng.integers(lo, hi + 1))
    while len(points) < target:
        nearby = points | guide
        frontier = [p for p in sorted(domain - points)
                    if any(max(abs(p[0] - y), abs(p[1] - x)) <= 1 for y, x in nearby)]
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    return points


def _shards(rng, radius, length, axis, tangent_mode, witness):
    """Sample two masks in a virtual chart; do not build a parent puzzle."""
    domain = {(y, x) for y in range(-radius, radius + 1)
              for x in range(-radius, radius + 1)}
    for _ in range(240):
        an = int(rng.integers(-radius + 1, radius))
        mn = int(rng.integers(-radius, radius))
        at = int(rng.integers(-radius, radius - length + 2))
        mt = at if tangent_mode == "zero" else int(rng.integers(-radius, radius - length + 2))
        if tangent_mode == "nonzero" and mt == at:
            continue
        delta = (an - 1 - mn, at - mt)
        if delta == (0, 0):
            continue
        ag = {(an, t) for t in range(at, at + length)}
        mg = {(mn, t) for t in range(mt, mt + length)}
        sg = _shift(mg, delta)
        forbidden_p = {(an + 1, t) for t in range(at, at + length)}
        forbidden_q = {(an - 2, t) for t in range(at, at + length)}
        dp = domain - ag - forbidden_p
        dq = (domain & _shift(domain, delta)) - sg - forbidden_q
        p = _grow(rng, dp, ag, {(an - 1, t) for t in range(at, at + length)},
                  2, min(12, len(dp)))
        q = _grow(rng, dq, sg, {(an, t) for t in range(at, at + length)},
                  2, min(12, len(dq))) if len(dq) >= 2 else None
        if p is None or q is None or not p - q or not q - p:
            continue
        s = p | q
        if not 4 <= len(s) <= 24 or not _connected(s):
            continue
        if witness and s == {(v, u) for u, v in s}:
            continue
        # One demonstration must expose collision union and both covered and
        # uncovered guide positions. These distinguish XOR and unconditional
        # guide recoloring/erasure without a latent sampling convention.
        if witness and (not p & q or not (ag | sg) - s or not (ag | sg) & s):
            continue
        rawq = _shift(q, (-delta[0], -delta[1]))
        if not (_connected(p | ag) and _connected(rawq | mg)):
            continue
        if witness and ((p | rawq) == s or not rawq - s
                        or (p | _shift(q, (1, 0))) == s
                        or (p | _shift(q, (2, 0))) == s):
            continue
        return tuple(_axis(v, axis) for v in (p, rawq, ag, mg, s))
    raise RuntimeError("bounded shard sampling exhausted")


def _frames(rng, radius, count):
    """Pack one or two independently sized frames, with variable staggering."""
    if count == 1:
        h, w = map(int, rng.integers(11, 24 - 2 * radius, size=2))
        ph, pw = h + 2 * radius, w + 2 * radius
        top = int(rng.integers(0, min(4, 31 - ph)))
        left = int(rng.integers(0, min(4, 31 - pw)))
        shape = (ph + top + int(rng.integers(0, min(4, 31 - ph - top))),
                 pw + left + int(rng.integers(0, min(4, 31 - pw - left))))
        return shape, [(top + radius, left + radius,
                        top + radius + h - 1, left + radius + w - 1)]
    assert radius <= 2
    vertical = bool(rng.integers(2))
    short_max = 13 if radius == 1 else 11
    ws = [int(rng.integers(11, short_max + 1)) for _ in range(2)]
    hs = [int(rng.integers(11, 24 - 2 * radius)) for _ in range(2)]
    widths = [w + 2 * radius for w in ws]
    heights = [h + 2 * radius for h in hs]
    slack = 30 - sum(widths)
    gap = int(rng.integers(slack + 1))
    lp = int(rng.integers(slack - gap + 1))
    width = sum(widths) + gap + lp
    width += int(rng.integers(31 - width))
    height = int(rng.integers(max(heights), 31))
    ys = [int(rng.integers(height - h + 1)) for h in heights]
    xs = (lp, lp + widths[0] + gap)
    rects = [(ys[i] + radius, xs[i] + radius,
              ys[i] + radius + hs[i] - 1, xs[i] + radius + ws[i] - 1) for i in range(2)]
    if vertical:
        rects = [(l, t, r, b) for t, l, b, r in rects]
        return (width, height), rects
    return (height, width), rects


def _make_pair(rng, palette, radius, length, axis, tangent_mode, rich, witness, used,
               forbidden_stencils=()):
    for _ in range(100):
        p, rawq, ag, mg, s = _shards(rng, radius, length, axis, tangent_mode, witness)
        if frozenset(s) in forbidden_stencils:
            continue
        frame_count = 2 if rich or (radius <= 2 and rng.random() < .28) else 1
        shape, rects = _frames(rng, radius, frame_count)
        corners = [c for rect in rects for c in _corners(rect)]
        if witness:
            fid = int(rng.integers(frame_count))
            cs = _corners(rects[fid])
            di = [cs[0], cs[3]]
            if rng.integers(2):
                di.reverse()
            other = cs[int(rng.choice((1, 2)))]
            selected = di + [other]
        else:
            order = rng.permutation(len(corners))
            count = int(rng.integers(5, min(8, len(corners)) + 1)) if rich else int(rng.integers(3, 5))
            selected = [corners[int(i)] for i in order[:count]]
            di = selected[:2]
        bg, frame, marker, donor_ink = palette[:4]
        edge = set().union(*(_outline(rect) for rect in rects))
        inp = np.full(shape, bg, dtype=np.int8)
        for point in edge:
            inp[point] = frame
        marks = [(di[0], donor_ink), (di[1], donor_ink)]
        for c in selected[2:]:
            ink = int(rng.choice(palette[3:]))
            marks.append((c, ink))
            # Clues never supply a missing shard as an accidental full reference.
            keep = int(rng.integers(1, min(4, len(p))))
            order = rng.permutation(len(p))[:keep]
            clue = {sorted(p)[int(i)] for i in order}
            for point in _world(c, clue):
                inp[point] = ink
        for c, patch, guide in ((di[0], p, ag), (di[1], rawq, mg)):
            for point in _world(c, patch):
                inp[point] = donor_ink
            for point in _world(c, guide):
                inp[point] = marker
        info = {"background": bg, "frame": frame, "edge": edge,
                "marks": marks, "stencil": s}
        out = _render(shape, info)
        if _key(inp) in used:
            continue
        try:
            parsed = _interpret(inp)
            assert np.array_equal(_render(shape, parsed), out)
            # Both donors must matter, even when receiver clues are included.
            assert not np.array_equal(_render(shape, parsed, parsed["p"]), out)
            assert not np.array_equal(_render(shape, parsed, parsed["q"]), out)
            if witness:
                assert not np.array_equal(_render(shape, parsed, convention="rotation"), out)
                assert not np.array_equal(_render(shape, parsed, convention="global"), out)
                unshifted = parsed["p"] | parsed["mover"]["patch"]
                assert not np.array_equal(_render(shape, parsed, unshifted), out)
                normal = (1, 0) if parsed["axis"] == 0 else (0, 1)
                for step in (normal, (2 * normal[0], 2 * normal[1])):
                    wrong = parsed["p"] | _shift(parsed["q"], step)
                    assert not np.array_equal(_render(shape, parsed, wrong), out)
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded scene sampling exhausted")


def _counts(num_train, num_test):
    for name, n, lo, hi in (("num_train", num_train, 2, 10),
                            ("num_test", num_test, 1, 4)):
        if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                or not lo <= int(n) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, int(rng.integers(4, 8)), replace=False)))
    changing = bool(rng.integers(2))
    mixed_axes = bool(rng.integers(2))
    base_axis = int(rng.integers(2))
    regime = int(rng.integers(3))
    puzzle, used = {"train": [], "test": []}, set()
    first_stencil = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = split == "train" and i == 0
            radius = int(rng.choice((1, 2, 2, 3)))
            rich = False
            tangent_mode = "any"
            if regime == 0:
                radius = int(rng.choice((2, 2, 3)))
                length = (2 if i % 2 == 0 else 4) if split == "train" else 3
            elif regime == 1:
                length = int(rng.integers(2, min(4, 2 * radius) + 1))
                tangent_mode = "zero" if split == "train" else "nonzero"
            else:
                radius = int(rng.choice((1, 1, 2)))
                length = int(rng.integers(2, min(4, 2 * radius) + 1))
                rich = split == "test"
            axis = (base_axis + i) % 2 if mixed_axes else base_axis
            roles = tuple(map(int, rng.choice(10, len(palette), replace=False))) if changing else palette
            forbidden = first_stencil if split == "train" and i == 1 else ()
            pair = _make_pair(rng, roles, radius, length, axis, tangent_mode, rich, witness, used,
                              forbidden)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if witness:
                first_stencil.add(frozenset(_interpret(pair["input"])["stencil"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the schema and semantic invariants using input-derived geometry."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            p = _interpret(inp)
            assert np.array_equal(_render(inp.shape, p), out)
            assert p["marker"] not in np.unique(out)
            assert set(map(int, np.unique(out))) == set(map(int, np.unique(inp))) - {p["marker"]}
            # Only neighborhoods of marked corners may change; frame evidence
            # at distance four/five and every unmarked corner must be preserved.
            allowed = set()
            for c, color in p["marks"]:
                allowed |= _world(c, {(u, v) for u in range(-3, 4) for v in range(-3, 4)})
                full = _world(c, p["stencil"])
                assert all(out[q] == color for q in full)
            assert all(tuple(map(int, q)) in allowed for q in np.argwhere(inp != out))
            assert len(p["p"]) < len(p["stencil"]) and len(p["q"]) < len(p["stencil"])
            assert not np.array_equal(_render(inp.shape, p, p["p"]), out)
            assert not np.array_equal(_render(inp.shape, p, p["q"]), out)
            if split == "train":
                training.append((inp, out, p))
    # Even two demonstrations teach the interpretation rather than depending
    # on all four default examples. First witness has diagonal donors and an
    # adjacent receiver, so rotation and reflection use identical donor charts.
    first_in, first_out, first = training[0]
    assert first["stencil"] != training[1][2]["stencil"]
    assert first["anchor"]["corner"][2] in (0, 3)
    assert first["mover"]["corner"][2] in (0, 3)
    assert not np.array_equal(_render(first_in.shape, first, convention="rotation"), first_out)
    assert not np.array_equal(_render(first_in.shape, first, convention="global"), first_out)
    assert not np.array_equal(_render(first_in.shape, first, first["p"] | first["mover"]["patch"]), first_out)
    guides = first["anchor"]["guide"] | _shift(first["mover"]["guide"], first["delta"])
    assert first["p"] & first["q"] and guides - first["stencil"] and guides & first["stencil"]
    assert first["mover"]["patch"] - first["stencil"]
    normal = (1, 0) if first["axis"] == 0 else (0, 1)
    for wrong in (first["p"] ^ first["q"], first["stencil"] | guides,
                  first["stencil"] - guides,
                  first["p"] | _shift(first["q"], normal),
                  first["p"] | _shift(first["q"], (2 * normal[0], 2 * normal[1]))):
        assert not np.array_equal(_render(first_in.shape, first, wrong), first_out)
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for k in ("input", "output"):
                a = np.asarray(pair[k], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed survey, count coverage, determinism and negative checks.

    Keeps one puzzle at a time and sets of short hashes/features, writes nothing.
    The returned compact summary is convenient for manual review.
    """
    hashes, shapes, stencils, palettes = set(), set(), set(), set()
    lengths, axes, frames, mark_counts, overlaps, offsets = set(), set(), set(), set(), set(), set()
    removed_old = 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        hashes.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                p = _interpret(a)
                assert np.array_equal(solve(a.copy()), out)
                shapes.add(a.shape)
                stencils.add(tuple(sorted(p["stencil"])))
                palettes.add((p["background"], p["frame"], p["marker"]))
                lengths.add(p["length"])
                axes.add(p["axis"])
                frames.add(len(p["rectangles"]))
                mark_counts.add(len(p["marks"]))
                overlaps.add(len(p["p"] & p["q"]))
                offsets.add(p["delta"])
                removed_old += int(bool(p["mover"]["patch"] - p["stencil"]))
        # Exact unrelated-output mutation must fail the semantic validator.
        changed = {s: list(puzzle[s]) for s in puzzle}
        victim = dict(changed["test"][0])
        victim["output"] = victim["output"].copy()
        pos = tuple(np.argwhere(victim["input"] == _interpret(victim["input"])["background"])[0])
        victim["output"][pos] = (int(victim["output"][pos]) + 1) % 10
        changed["test"][0] = victim
        try:
            validate_generated_puzzle(changed)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("output mutation accepted")
    assert len(hashes) == 42 and len(shapes) >= 35 and len(stencils) >= 120
    assert lengths == {2, 3, 4} and axes == {0, 1} and frames == {1, 2}
    assert {3, 4, 5, 6, 7} <= mark_counts and len(offsets) >= 12
    assert 0 in overlaps and max(overlaps) >= 2 and removed_old >= 30
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(0, nt, nq)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, True), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "input_shapes": len(shapes), "stencils": len(stencils),
            "role_triples": len(palettes), "bar_lengths": sorted(lengths),
            "axes": sorted(axes), "frame_counts": sorted(frames),
            "marked_counts": sorted(mark_counts), "overlap_counts": sorted(overlaps),
            "translations": len(offsets), "pairs_erasing_displaced_ink": removed_old}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
