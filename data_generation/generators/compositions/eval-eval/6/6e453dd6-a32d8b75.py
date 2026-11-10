"""Crossing-aperture loom. See task_spec.md for the pixel-defined rule.

No parent modules, saved puzzles, files, or RNG state are used by solve().
All experiments are in memory; this module writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        groups.append(cells)
    return groups


def _arm(arm, bg):
    """Decode an eastward work arm from input pixels, using true enclosure."""
    aligned = np.full_like(arm, bg)
    profiles = {k: np.zeros(len(arm), bool) for k in ("active", "all_holes", "naive", "body")}
    records = []
    for cells in _components(arm != bg):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        color = np.unique(arm[rr, cc])
        _require(len(color) == 1, "monochrome source components")
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
        mask[rr - r0, cc - c0] = True
        holes, active, recessed = [], [], []
        for hole in _components(~mask):
            if any(r in (0, len(mask) - 1) or c in (0, mask.shape[1] - 1) for r, c in hole):
                continue
            holes.append(hole)
            if any(c == mask.shape[1] - 2 for r, c in hole):
                active.append(hole)
                rows = {r for r, c in hole}
                contact = {r for r, c in hole if c == mask.shape[1] - 2}
                _require(rows == contact, "partial-contact cavities are outside this family")
            else:
                recessed.append(hole)
        for kind, regions in (("active", active), ("all_holes", holes)):
            for hole in regions:
                profiles[kind][[r0 + r for r, c in hole]] = True
        profiles["body"][r0:r1 + 1] = True
        if mask.shape[1] >= 3:
            profiles["naive"][r0:r1 + 1] |= ~mask[:, -2]
        shift = arm.shape[1] - 1 - c1
        aligned[rr, cc + shift] = color[0]
        records.append(dict(row=r0, end=r1, shift=shift, color=int(color[0]), mask=mask,
                            active=active, recessed=recessed, cells=cells))
    _require(bool(records), "each work arm must contain objects")
    for a, b in zip(records, records[1:]):
        _require(a["end"] + 1 < b["row"], "source object row bands must be separated")
    return aligned, profiles, records


def _decode(input_grid):
    """Find the unique rail cross and framed-tile quadrant, in any rotation."""
    original = _grid(input_grid)
    candidates = []
    for turn in range(4):
        g = np.rot90(original, turn)
        bg = int(g[0, 0])
        rows = [r for r in range(4, len(g) - 4)
                if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])]
        cols = [c for c in range(4, g.shape[1] - 4)
                if int(g[0, c]) != bg and np.all(g[:, c] == g[0, c])]
        for r in rows:
            for c in cols:
                rail = int(g[r, c])
                if int(g[r, 0]) != rail or int(g[0, c]) != rail:
                    continue
                header = g[:r, :c]
                points = np.argwhere(header != bg)
                if not len(points):
                    continue
                lo, hi = points.min(0), points.max(0) + 1
                y, x = map(int, lo)
                yy, xx = map(int, hi)
                frame = header[y:yy, x:xx]
                if frame.shape[0] != frame.shape[1] or not 4 <= len(frame) <= 6:
                    continue
                if not (y >= 1 and x >= 1 and yy < r and xx < c):
                    continue
                if not all(np.all(edge == rail) for edge in (frame[0], frame[-1], frame[:, 0], frame[:, -1])):
                    continue
                tile = frame[1:-1, 1:-1].copy()
                colors = np.unique(tile)
                if len(colors) != 2 or bg in colors or rail in colors:
                    continue
                n = len(tile)
                h, w = len(g) - r - 1, g.shape[1] - c - 1
                if h % n or w % n or min(h, w) < 3 * n:
                    continue
                expected = np.full_like(header, bg)
                expected[y:yy, x:xx] = frame
                if not np.array_equal(header, expected):
                    continue
                if np.any(g[r + 1:, :c] == rail) or np.any(g[:r, c + 1:] == rail):
                    continue
                try:
                    west = _arm(g[r + 1:, :c], bg)
                    north = _arm(g[:r, c + 1:].T, bg)
                    for arm in (west, north):
                        gate = arm[1]["active"]
                        if not gate.any() or not np.all(gate.reshape(-1, n) == gate.reshape(-1, n)[:, :1]):
                            raise AssertionError("unregistered aperture profile")
                except AssertionError:
                    continue
                candidates.append(dict(g=g, turn=turn, bg=bg, rail=rail, r=r, c=c,
                                       tile=tile, n=n, west=west, north=north))
    _require(len(candidates) == 1, "unique rail cross and tile quadrant required")
    return candidates[0]


def _render_decoded(d, behavior="normal"):
    g, r, c, n, tile = (d[k] for k in ("g", "r", "c", "n", "tile"))
    out = g.copy()
    if behavior != "no_alignment":
        out[r + 1:, :c] = d["west"][0]
        out[:r, c + 1:] = d["north"][0].T
    profile = behavior if behavior in ("all_holes", "naive", "body") else "active"
    a, b = d["west"][1][profile], d["north"][1][profile]
    gate = (a[:, None] | b[None, :]) if behavior == "union" else (a[:, None] & b[None, :])
    if behavior == "ignore_apertures":
        gate[:] = True
    values = np.unique(tile)
    swapped = (values[0] + int(values[1]) - tile).astype(np.int8)
    if behavior == "no_exchange":
        swapped = tile
    elif behavior == "rotate_tile":
        swapped = np.rot90(swapped)
    elif behavior == "solid_crossing":
        swapped = np.full_like(tile, values[0])
    elif behavior == "phase_shift":
        swapped = np.roll(swapped, 1, axis=0)
    if behavior == "one_stamp_per_patch":
        gate[:] = False
        # Only the first tile of each contiguous row/column intersection patch.
        ar = np.flatnonzero(a & ~np.r_[False, a[:-1]])
        bc = np.flatnonzero(b & ~np.r_[False, b[:-1]])
        for y in ar:
            for x in bc:
                gate[y:y + n, x:x + n] = True
    rr, cc = np.indices(gate.shape)
    canvas = out[r + 1:, c + 1:]
    canvas[gate] = swapped[rr[gate] % n, cc[gate] % n]
    return np.rot90(out, -d["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Slide sources to the rails and print swapped tiles at aperture crossings."""
    return _render_decoded(_decode(input_grid))


def _intervals(rng, q, n, kinds, lengths):
    gap = (3 + n - 1) // n
    spare = q - 2 - sum(lengths) - gap * (len(kinds) - 1)
    _require(spare >= 0, "source schedule does not fit")
    extra = rng.multinomial(spare, np.full(len(kinds) + 1, 1 / (len(kinds) + 1)))
    order = rng.permutation(len(kinds))
    slot = 1 + int(extra[0])
    items = []
    for j, i in enumerate(order):
        length = lengths[int(i)]
        items.append((slot * n, (slot + length) * n, kinds[int(i)]))
        slot += length + gap + int(extra[j + 1])
    return items


def _make_arm(rng, q, n, width, bg, colors, items, combine=False, moving=False):
    """Construct masks and forward labels without calling the solver/decoder."""
    groups, group = [], []
    for item in items:
        if group and not combine and rng.random() < .6:
            groups.append(group)
            group = []
        group.append(item)
    groups.append(group)
    src = np.full((q * n, width), bg, np.int8)
    aligned = src.copy()
    gate = np.zeros(q * n, bool)
    for index, group in enumerate(groups):
        top, bottom = group[0][0] - 1, group[-1][1] + 1
        w = int(rng.integers(4, min(7, width - 1) + 1))
        mask = np.ones((bottom - top, w), bool)
        for start, stop, kind in group:
            back = int(rng.integers(1, w - 2))
            for row in range(start - top, stop - top):
                edge = int(rng.integers(1, w - 2)) if rng.random() < .35 else back
                if kind == "active":
                    mask[row, edge:w - 1] = False
                elif kind == "open":
                    mask[row, :w - 1] = False
                elif kind == "recessed":
                    mask[row, 1:w - 2] = False
            if kind == "active":
                gate[start:stop] = True
        if all(kind == "recessed" for _, _, kind in group) and rng.random() < .65:
            # A thin tail, rather than a local cavity wall, defines the extreme.
            mask[:, -1] = False
            mask[int(rng.choice([0, len(mask) - 1])), -1] = True
        elif rng.random() < .4:
            # Corner erosion changes silhouettes without affecting enclosure.
            mask[int(rng.choice([0, len(mask) - 1])), 0] = False
        limit = width - w
        shift = int(rng.integers(limit + 1))
        if moving and index == 0:
            shift = int(rng.integers(1, limit + 1))
        color = int(rng.choice(colors))
        view = src[top:bottom, width - w - shift:width - shift]
        view[mask] = color
        aligned[top:bottom, width - w:][mask] = color
    return src, aligned, gate


def _binary_tile(rng, n, colors):
    for _ in range(32):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if (bits.any() and not bits.all()
                and all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3))
                and not np.array_equal(bits, np.roll(bits, 1, axis=0))):
            return np.where(bits, colors[0], colors[1]).astype(np.int8)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(np.int8)


def _make_pair(rng, index, test, palette, base_turn):
    # Two minimal-count witnesses teach elongation on each axis separately.
    if not test and index < 2:
        n = 2 + index
        long_axis = index
    elif test:
        n = int(rng.choice([2, 3]))
        long_axis = None
    else:
        n = int(rng.choice([2, 3, 4]))
        long_axis = int(rng.integers(2)) if n < 4 else None
    min_work = n + 4
    max_work = min(9, 29 - (7 if n == 3 else 5) * n)
    if test and n == 2:
        max_work = 9
    wr = int(rng.integers(min_work, max(min_work, max_work) + 1))
    wc = int(rng.integers(min_work, max(min_work, max_work) + 1))
    bg, rail, ta, tb, ink = map(int, palette[:5])
    tile = _binary_tile(rng, n, (ta, tb))
    arms = []
    for axis, width in enumerate((wc, wr)):
        if test:
            # Two active slots on each axis; separate holes or one long window.
            separated = bool(rng.integers(2))
            kinds = ["active", "active", str(rng.choice(["open", "recessed"]))] if separated else ["active", str(rng.choice(["open", "recessed"]))]
            lengths = [1, 1, 1] if separated else [2, 1]
        else:
            distractor = ("open" if axis == index % 2 else "recessed") if index < 2 else str(rng.choice(["open", "recessed", "solid"]))
            kinds = ["active", distractor]
            lengths = [2 if axis == long_axis else 1, 1]
        gap = (3 + n - 1) // n
        minimum = 2 + sum(lengths) + gap * (len(kinds) - 1)
        # The perpendicular arm's work depth is the near-side extent on this
        # axis. It may differ from this arm's own work depth.
        near_extent = wr if axis == 0 else wc
        maximum = (29 - near_extent) // n
        q = int(rng.integers(minimum, maximum + 1))
        items = _intervals(rng, q, n, kinds, lengths)
        arm = _make_arm(rng, q, n, width, bg, (ta, tb, ink), items,
                        combine=test and rng.random() < .35, moving=not test and index < 2)
        arms.append(arm)
    west, north = arms
    h, w = len(west[0]), len(north[0])
    g = np.full((wr + 1 + h, wc + 1 + w), bg, np.int8)
    g[wr, :] = rail
    g[:, wc] = rail
    ty = int(rng.integers(1, wr - n - 2))
    tx = int(rng.integers(1, wc - n - 2))
    g[ty:ty + n + 2, tx:tx + n + 2] = rail
    g[ty + 1:ty + n + 1, tx + 1:tx + n + 1] = tile
    g[wr + 1:, :wc] = west[0]
    g[:wr, wc + 1:] = north[0].T
    canvas = g[wr + 1:, wc + 1:]
    if test or index >= 2:
        style = int(rng.integers(4))
        yy, xx = np.indices(canvas.shape)
        if style == 1:
            canvas[(yy // n) % 3 == 1] = ink
        elif style == 2:
            canvas[(xx // n) % 3 == 1] = int(rng.choice([ta, tb, ink]))
        elif style == 3:
            canvas[((yy // n) + 2 * (xx // n)) % 4 == 1] = ink
    out = g.copy()
    out[wr + 1:, :wc] = west[1]
    out[:wr, wc + 1:] = north[1].T
    swapped = (ta + tb - tile).astype(np.int8)
    target = out[wr + 1:, wc + 1:]
    for y in range(h // n):
        for x in range(w // n):
            if west[2][y * n] and north[2][x * n]:
                target[y * n:(y + 1) * n, x * n:(x + 1) * n] = swapped
    turn = base_turn if index < 2 and not test else (base_turn + int(rng.integers(4))) % 4
    return {"input": np.rot90(g, turn).copy(), "output": np.rot90(out, turn).copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    fixed_palette = rng.permutation(10)
    varying = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            palette = rng.permutation(10) if varying else fixed_palette
            pair = _make_pair(rng, i, split == "test", palette, base_turn)
            # A finite 20-element palette orbit guarantees distinctness: the
            # input uses >=4 colors, so all orbit elements are distinct. There
            # are at most 13 earlier inputs; no stochastic retry loop is needed.
            for sign in (1, -1):
                for offset in range(10):
                    candidate = {role: ((sign * a.astype(int) + offset) % 10).astype(np.int8)
                                 for role, a in pair.items()}
                    if _key(candidate["input"]) not in seen:
                        break
                else:
                    continue
                break
            else:
                raise AssertionError("uniqueness orbit exhausted")
            seen.add(_key(candidate["input"]))
            puzzle[split].append(candidate)
    validate_generated_puzzle(puzzle)
    return puzzle


_ALTERNATIVES = ("all_holes", "naive", "body", "union", "ignore_apertures",
                 "no_exchange", "rotate_tile", "solid_crossing", "phase_shift",
                 "one_stamp_per_patch", "no_alignment")


def _exterior_gate(mask):
    """Independent array-reachability check of the leading aperture rows."""
    occupied = np.pad(mask, 1)
    outside = np.zeros_like(occupied)
    outside[0] = outside[-1] = True
    outside[:, 0] = outside[:, -1] = True
    for _ in range(occupied.size):
        grown = outside.copy()
        grown[1:] |= outside[:-1]
        grown[:-1] |= outside[1:]
        grown[:, 1:] |= outside[:, :-1]
        grown[:, :-1] |= outside[:, 1:]
        grown &= ~occupied
        if np.array_equal(grown, outside):
            return (~occupied & ~outside)[1:-1, -3]
        outside = grown
    raise AssertionError("flood fill did not converge")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, distinguished, train_spans, test_spans = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(solve(inp), out), "output disagrees with input-only solve")
            spans = []
            for arm in (d["west"], d["north"]):
                independent = np.zeros(len(arm[0]), bool)
                for rec in arm[2]:
                    m = rec["mask"]
                    _require(len(_components(m)) == 1, "source connectivity")
                    independent[rec["row"]:rec["end"] + 1] |= _exterior_gate(m)
                _require(np.array_equal(independent, arm[1]["active"]), "enclosure algorithms disagree")
                spans.append(int(independent.sum()) // d["n"])
            # Inverse pixel lookup audits full overwrite and transparency.
            r, c, n = d["r"], d["c"], d["n"]
            actual = np.rot90(out, d["turn"])
            tile = d["tile"]
            colors = list(map(int, np.unique(tile)))
            for y in range(len(d["west"][0])):
                for x in range(len(d["north"][0])):
                    expected = int(d["g"][r + 1 + y, c + 1 + x])
                    if d["west"][1]["active"][y] and d["north"][1]["active"][x]:
                        expected = colors[1] if tile[y % n, x % n] == colors[0] else colors[0]
                    _require(actual[r + 1 + y, c + 1 + x] == expected, "crossing pixel audit")
            _require(np.array_equal(actual[:r + 1, :c + 1], d["g"][:r + 1, :c + 1]), "legend or rail changed")
            _require(np.array_equal(actual[r, :], d["g"][r, :]) and
                     np.array_equal(actual[:, c], d["g"][:, c]), "rails changed")
            if split == "train":
                train_spans.append(spans)
                for alt in _ALTERNATIVES:
                    if not np.array_equal(_render_decoded(d, alt), out):
                        distinguished.add(alt)
            else:
                test_spans.append(spans)
    _require(set(_ALTERNATIVES) == distinguished, "training leaves a stated shortcut indistinguishable")
    _require(any(x >= 2 and y == 1 for x, y in train_spans) and
             any(x == 1 and y >= 2 for x, y in train_spans), "missing separate-axis repetition witnesses")
    _require(all(min(s) >= 2 for s in test_spans), "missing combined-axis test shift")


def family_features(pair):
    """Small pixel-derived measurements for bounded diagnostics."""
    d = _decode(pair["input"])
    recs = d["west"][2] + d["north"][2]
    return dict(shape=list(np.asarray(pair["input"]).shape), n=d["n"], turn=d["turn"],
                objects=len(recs), shifts=[o["shift"] for o in recs],
                active=sum(len(o["active"]) for o in recs),
                recessed=sum(len(o["recessed"]) for o in recs),
                open_contrast=any(np.any(a[1]["naive"] & ~a[1]["all_holes"]) for a in (d["west"], d["north"])),
                mixed=any(o["active"] and o["recessed"] for o in recs),
                spans=[int(a[1]["active"].sum()) // d["n"] for a in (d["west"], d["north"])],
                canvas_colors=len(np.unique(d["g"][d["r"] + 1:, d["c"] + 1:])),
                masks=[hashlib.sha256(o["mask"].tobytes() + bytes(o["mask"].shape)).hexdigest() for o in recs])


def self_test():
    # Stream one puzzle at a time: no saved sample payloads or execution traces.
    signatures, shapes, masks, ns, turns, counts, distances = set(), set(), set(), set(), set(), set(), set()
    silent_contrasts = mixed = patterned = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                f = family_features(pair)
                shapes.add(tuple(f["shape"]))
                masks.update(f["masks"])
                ns.add(f["n"])
                turns.add(f["turn"])
                counts.add(f["objects"])
                distances.update(f["shifts"])
                silent_contrasts += f["open_contrast"]
                mixed += f["mixed"]
                patterned += f["canvas_colors"] > 1
                digest.update(pair["input"].tobytes())
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list-input solve")
        signatures.add(digest.hexdigest())
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(masks) >= 120, "structural variation collapsed")
    _require(ns == {2, 3, 4} and turns == {0, 1, 2, 3}, "scale or orientation variation collapsed")
    _require({2, 3, 4, 5}.issubset(counts) and {0, 1, 2}.issubset(distances), "object/travel variation collapsed")
    _require(silent_contrasts >= 84 and mixed >= 10 and patterned >= 40, "contrast or interaction variation collapsed")
    p, q = generate_puzzle(778, 2, 1), generate_puzzle(778, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "determinism")
    # Equivariance exercises solving conventions beyond sampled orientations.
    example = p["test"][0]
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(example["input"], k)), np.rot90(example["output"], k)), "rotation equivariance")
    lookup = np.random.default_rng(552).permutation(10)
    _require(np.array_equal(solve(lookup[example["input"]]), lookup[example["output"]]), "palette equivariance")
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            generate_puzzle(902 + nt * 5 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds; all legal counts; input-only reconstruction")
