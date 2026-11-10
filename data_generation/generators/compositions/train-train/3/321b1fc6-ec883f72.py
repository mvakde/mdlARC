"""Contact-color stamp printer: input-derived solver and bounded generator.

Only NumPy and the Python standard library are used. No parent is imported.
"""

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30")
    return a


def _sig(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, bg):
    h, w = a.shape
    seen, result = set(), []
    for r in range(h):
        for c in range(w):
            if a[r, c] == bg or (r, c) in seen:
                continue
            stack, cells = [(r, c)], []
            seen.add((r, c))
            while stack:
                p, q = stack.pop()
                cells.append((p, q))
                for nr, nc in ((p - 1, q), (p + 1, q), (p, q - 1), (p, q + 1)):
                    if (0 <= nr < h and 0 <= nc < w and a[nr, nc] != bg
                            and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            top = min(x for x, _ in cells)
            left = min(y for _, y in cells)
            mh = max(x for x, _ in cells) - top + 1
            mw = max(y for _, y in cells) - left + 1
            mask = np.zeros((mh, mw), dtype=bool)
            for x, y in cells:
                mask[x - top, y - left] = True
            result.append({"origin": (top, left), "mask": mask, "cells": cells,
                           "colors": {int(a[x, y]) for x, y in cells}})
    return result


def _frame_options(obj, donor):
    """Enumerate exact U decompositions, never relying on the longest run."""
    for axis in (0, 1):
        pts = {(r, c) if axis == 0 else (c, r) for r, c in obj["cells"]}
        ds = [(r, c) if axis == 0 else (c, r) for r, c in donor["cells"]]
        p0, p1 = min(r for r, _ in ds), max(r for r, _ in ds)
        q0, q1 = min(c for _, c in ds), max(c for _, c in ds)
        for row in sorted({r for r, _ in pts}):
            cols = sorted(c for r, c in pts if r == row)
            left, right = cols[0], cols[-1]
            if len(cols) < 4 or cols != list(range(left, right + 1)):
                continue
            for inward in (-1, 1):
                depths = [inward * (p0 - row), inward * (p1 - row)]
                if min(depths) < 2 or q0 < left + 2 or q1 > right - 2:
                    continue
                arms = []
                for end in (left, right):
                    dd = sorted(inward * (r - row) for r, c in pts if c == end and r != row)
                    if not dd or dd != list(range(1, len(dd) + 1)):
                        break
                    arms.append(len(dd))
                if len(arms) != 2 or min(arms) < max(depths):
                    continue
                expected = {(row, c) for c in range(left, right + 1)}
                expected |= {(row + inward * t, end)
                             for end, length in zip((left, right), arms)
                             for t in range(1, length + 1)}
                if pts != expected:
                    continue
                ends = ((row, left), (row, right))
                directions = ((-inward, -1), (-inward, 1))
                normal = (inward, 0)
                if axis:
                    ends = tuple((c, r) for r, c in ends)
                    directions = tuple((c, r) for r, c in directions)
                    normal = normal[::-1]
                yield {"axis": axis, "inward": inward, "ends": ends,
                       "directions": directions, "normal": normal,
                       "arms": tuple(arms), "bar_length": right - left + 1}


def _parse(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    objects = _components(a, bg)
    donors = [o for o in objects if len(o["colors"]) > 1]
    if len(donors) != 1:
        raise ValueError("expected one multicolored donor")
    donor = donors[0]
    mono = [o for o in objects if len(o["colors"]) == 1]
    options = [(o, f) for o in mono for f in _frame_options(o, donor)]
    if len(options) != 1:
        raise ValueError("expected one unambiguous complete U bracket")
    frame, geometry = options[0]
    targets = [o for o in mono if o is not frame]
    if not targets:
        raise ValueError("expected at least one matching silhouette")
    placeholder_colors = set()
    for obj in targets:
        if not np.array_equal(obj["mask"], donor["mask"]):
            raise ValueError("all silhouettes must match the donor mask")
        placeholder_colors |= obj["colors"]
        # All targets are wholly in the outward half-plane, with clearance.
        n, e = geometry["normal"], geometry["ends"][0]
        if any((r - e[0]) * n[0] + (c - e[1]) * n[1] > -2
               for r, c in obj["cells"]):
            raise ValueError("silhouettes must be outside the bracket")
    fc = next(iter(frame["colors"]))
    if placeholder_colors & (donor["colors"] | {bg, fc}) or fc in donor["colors"]:
        raise ValueError("color roles must be disjoint")
    return {"grid": a, "bg": bg, "donor": donor, "frame": frame,
            "targets": targets, **geometry}


def _trace(scene):
    """Trace original occupied cells; empty holes/notches transmit the ray."""
    a, donor = scene["grid"], scene["donor"]
    owner = {p: i for i, o in enumerate(scene["targets"]) for p in o["cells"]}
    hits = []
    for end, d in zip(scene["ends"], scene["directions"]):
        r, c = end[0] + d[0], end[1] + d[1]
        path, hit, step = [], None, 1
        while 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
            if a[r, c] != scene["bg"]:
                if (r, c) not in owner:
                    raise ValueError("unexpected obstacle on an outward ray")
                index = owner[r, c]
                origin = scene["targets"][index]["origin"]
                offset = (r - origin[0], c - origin[1])
                source = donor["origin"]
                hit = {"index": index, "offset": offset, "contact": (r, c),
                       "color": int(a[source[0] + offset[0], source[1] + offset[1]]),
                       "step": step}
                break
            path.append((r, c))
            r, c = r + d[0], c + d[1]
            step += 1
        hits.append({"path": path, "hit": hit})
    selected = [b["hit"]["index"] for b in hits if b["hit"] is not None]
    if len(selected) != len(set(selected)):
        raise ValueError("rays must have distinct recipients")
    return hits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from visible input. No seed or generation state is consulted."""
    s = _parse(input_grid)
    out = s["grid"].copy()
    donor = s["donor"]
    sr, sc = donor["origin"]
    offsets = [tuple(map(int, x)) for x in np.argwhere(donor["mask"])]
    for branch in _trace(s):
        hit = branch["hit"]
        if hit is None:
            continue
        tr, tc = s["targets"][hit["index"]]["origin"]
        for r, c in offsets:
            out[tr + r, tc + c] = s["grid"][sr + r, sc + c]
        for p in branch["path"]:
            out[p] = hit["color"]
    for p in donor["cells"]:
        out[p] = s["bg"]
    return out


def _connected(mask):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not cells:
        return False
    seen, stack = {min(cells)}, [min(cells)]
    while stack:
        r, c = stack.pop()
        for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if p in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == cells


def _sample_mask(rng, style, large=False):
    if style == "solid":
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        return np.ones((h, w), dtype=bool)
    h, w = int(rng.integers(2, 6 if large else 5)), int(rng.integers(2, 6 if large else 5))
    mode = int(rng.integers(5))
    if h >= 3 and w >= 3 and mode == 0:
        m = np.ones((h, w), dtype=bool)
        m[1:-1, 1:-1] = False
    elif h >= 3 and w >= 3 and mode == 1:
        m = np.zeros((h, w), dtype=bool)
        m[int(rng.integers(1, h - 1)), :] = True
        m[:, int(rng.integers(1, w - 1))] = True
    elif h >= 3 and w >= 3 and mode == 2:
        m = np.ones((h, w), dtype=bool)
        m[:int(rng.integers(1, h)), int(rng.integers(1, w - 1))] = False
    else:
        m = np.zeros((h, w), dtype=bool)
        m[int(rng.integers(h)), int(rng.integers(w))] = True
        target = int(rng.integers(max(h, w) + 1, h * w + 1))
        while True:
            rr, cc = np.nonzero(m)
            if (int(m.sum()) >= target and rr.min() == 0 and rr.max() == h - 1
                    and cc.min() == 0 and cc.max() == w - 1):
                break
            frontier = [(r, c) for r in range(h) for c in range(w) if not m[r, c]
                        and any(0 <= nr < h and 0 <= nc < w and m[nr, nc]
                                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))]
            if not frontier:
                break
            m[frontier[int(rng.integers(len(frontier)))]] = True
        if m.all():
            choices = [(r, c) for r in range(h) for c in range(w)]
            rng.shuffle(choices)
            for p in choices:
                m[p] = False
                rr, cc = np.nonzero(m)
                if (_connected(m) and rr.min() == 0 and rr.max() == h - 1
                        and cc.min() == 0 and cc.max() == w - 1):
                    break
                m[p] = True
    if rng.integers(2):
        m = m[::-1].copy()
    if rng.integers(2):
        m = m[:, ::-1].copy()
    return m


def _separate(a, b, shape):
    h, w = shape
    return (a[0] + h < b[0] or b[0] + h < a[0]
            or a[1] + w < b[1] or b[1] + w < a[1])


def _placements(mask, row, left, right, width):
    """All complete outward placements, classified by actual ray intersections."""
    h, w = mask.shape
    rr, cc = np.nonzero(mask)
    rays = [dict(), dict()]
    for side, (end, dc) in enumerate(((left, -1), (right, 1))):
        for t in range(1, row + 1):
            p = (row - t, end + dc * t)
            if 0 <= p[1] < width:
                rays[side][p] = t
    groups = [[], [], []]  # left hit, right hit, no hit
    for r in range(row - h):
        for c in range(width - w + 1):
            contacts = []
            for side in (0, 1):
                occupied = [(rays[side][(r + int(x), c + int(y))], (int(x), int(y)))
                            for x, y in zip(rr, cc) if (r + int(x), c + int(y)) in rays[side]]
                if occupied:
                    t, offset = min(occupied)
                    contacts.append((side, t, offset))
            if len(contacts) > 1:
                continue
            item = {"origin": (r, c)}
            if contacts:
                side, t, offset = contacts[0]
                item.update(side=side, step=t, offset=offset)
                groups[side].append(item)
            else:
                groups[2].append(item)
    return groups


def _payload(rng, mask, colors, hits):
    pts = [tuple(map(int, p)) for p in np.argwhere(mask)]
    n = len(pts)
    k = int(rng.integers(2, min(5, len(colors), n) + 1))
    palette = [int(x) for x in rng.choice(colors, k, replace=False)]
    mode = int(rng.integers(6))
    values = {}
    centers = [pts[int(i)] for i in rng.choice(n, k, replace=False)]
    for p in pts:
        r, c = p
        if mode == 0:
            index = r % k
        elif mode == 1:
            index = c % k
        elif mode == 2:
            index = (r + c) % k
        elif mode == 3:
            index = min(range(k), key=lambda i: abs(r - centers[i][0]) + abs(c - centers[i][1]))
        elif mode == 4:
            index = 0 if rng.random() < 0.8 else int(rng.integers(k))
        else:
            index = int(rng.integers(k))
        values[p] = palette[index]
    # A single-color answer and a single shared trail color are explicit foils.
    values[pts[0]], values[pts[-1]] = palette[0], palette[1]
    if len(hits) == 2:
        values[hits[0]["offset"]] = palette[0]
        values[hits[1]["offset"]] = palette[1]
    elif hits and hits[0]["offset"] != pts[0]:
        values[pts[0]], values[hits[0]["offset"]] = palette[0], palette[1]
    if len(set(values.values())) < 2:
        other = next(p for p in pts if not hits or p != hits[0]["offset"])
        values[other] = palette[0] if values[pts[0]] != palette[0] else palette[1]
    return values


def _make_pair(rng, *, dual, shadow, style, test, rotation, mirror,
               near_witness=False, bbox_witness=False):
    """Build known contacts first; construct the label without using solve()."""
    for attempt in range(500):
        mask = _sample_mask(rng, style, large=test or attempt % 4 == 0)
        h, w = mask.shape
        bar_len = w + 4 + int(rng.integers(0, 3))
        long_margin = int(rng.integers(11 if shadow else 6, 19 if not dual else 16))
        short_margin = int(rng.integers(5 if dual else 0, 10 if dual else 8))
        if long_margin + short_margin + bar_len > 30:
            continue
        long_side = int(rng.integers(2))
        lm, rm = (long_margin, short_margin) if long_side == 0 else (short_margin, long_margin)
        width = lm + bar_len + rm
        left, right = lm, lm + bar_len - 1
        depth = int(rng.integers(2, 4))
        arms = (depth + h - 1 + int(rng.integers(0, 4)),
                depth + h - 1 + int(rng.integers(0, 4)))
        bottom = int(rng.integers(0, 4))
        maxrow = 29 - max(arms) - bottom
        if maxrow < 12:
            continue
        row = int(rng.integers(12, maxrow + 1))
        height = row + max(arms) + bottom + 1
        groups = _placements(mask, row, left, right, width)
        low, high = (5, 7) if test else (2, 4)
        sides = [long_side, 1 - long_side] if dual else [long_side]
        chosen, hits = [], []
        for side in sides:
            candidates = [o for o in groups[side] if low <= o["step"] <= high
                          and all(_separate(o["origin"], p["origin"], mask.shape) for p in chosen)
                          and all(o["offset"] != p["offset"] for p in hits)]
            if bbox_witness:
                dc = -1 if side == 0 else 1
                # One step before contact is inside the bounding box but empty.
                candidates = [o for o in candidates
                              if 0 <= o["offset"][0] + 1 < h
                              and 0 <= o["offset"][1] - dc < w]
            if style != "solid" and len(hits) == 0:
                # Favor entry away from bbox corners, exposing local lookup.
                noncorner = [o for o in candidates if o["offset"] not in
                             ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))]
                if noncorner:
                    candidates = noncorner
            if not candidates:
                break
            hit = candidates[int(rng.integers(len(candidates)))]
            hits.append(hit)
            chosen.append(hit)
        if len(hits) != len(sides):
            continue
        if shadow:
            first = hits[0]
            options = [o for o in groups[long_side] if o["step"] > first["step"]
                       and all(_separate(o["origin"], p["origin"], mask.shape) for p in chosen)]
            if not options:
                continue
            chosen.append(options[int(rng.integers(len(options)))])
        source = (row + depth, int(rng.integers(left + 2, right - w)))
        options = [o for o in groups[2]
                   if all(_separate(o["origin"], p["origin"], mask.shape) for p in chosen)]
        if near_witness:
            distance = lambda o: (o["origin"][0] - source[0]) ** 2 + (o["origin"][1] - source[1]) ** 2
            options = [o for o in options if distance(o) < min(distance(p) for p in hits)]
        if not options:
            continue
        chosen.append(options[int(rng.integers(len(options)))])
        # More off-ray blanks vary counts independently of the two branches.
        for _ in range(int(rng.integers(0, 3))):
            options = [o for o in groups[2]
                       if all(_separate(o["origin"], p["origin"], mask.shape) for p in chosen)]
            if options:
                chosen.append(options[int(rng.integers(len(options)))])
        palette = [int(x) for x in rng.permutation(10)]
        bg, frame_color = palette[:2]
        nplace = int(rng.integers(1, min(3, len(chosen)) + 1))
        placeholder = palette[2:2 + nplace]
        values = _payload(rng, mask, palette[2 + nplace:], hits)
        a = np.full((height, width), bg, dtype=np.int8)
        a[row, left:right + 1] = frame_color
        for end, arm in zip((left, right), arms):
            a[row + 1:row + arm + 1, end] = frame_color
        for (r, c), color in values.items():
            a[source[0] + r, source[1] + c] = color
        for index, obj in enumerate(chosen):
            tr, tc = obj["origin"]
            for r, c in values:
                a[tr + r, tc + c] = placeholder[index % len(placeholder)]
        if int(np.count_nonzero(a == bg)) * 2 <= a.size:
            continue
        out = a.copy()
        for r, c in values:
            out[source[0] + r, source[1] + c] = bg
        for hit in hits:
            tr, tc = hit["origin"]
            for (r, c), color in values.items():
                out[tr + r, tc + c] = color
            end, dc = (left, -1) if hit["side"] == 0 else (right, 1)
            for t in range(1, hit["step"]):
                out[row - t, end + dc * t] = values[hit["offset"]]
        # Rotate/reflection is global, preserving translated mask correspondence.
        if mirror:
            a, out = a[:, ::-1], out[:, ::-1]
        a, out = np.rot90(a, rotation).copy(), np.rot90(out, rotation).copy()
        return {"input": a, "output": out}
    raise RuntimeError("bounded placement search exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_rotation = int(rng.integers(4))
    specs = [dict(dual=True, shadow=True, style="solid", test=False, near_witness=True),
             dict(dual=False, shadow=False, style="irregular", test=False, bbox_witness=True)]
    while len(specs) < num_train:
        specs.append(dict(dual=bool(rng.integers(2)), shadow=bool(rng.integers(2)),
                          style="solid" if rng.random() < 0.25 else "irregular", test=False))
    rotations = [(base_rotation + i) % 4 for i in range(num_train)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            spec = specs[i] if split == "train" else dict(dual=bool(rng.integers(2)), shadow=True,
                                                         style="irregular", test=True)
            rotation = rotations[i] if split == "train" else int(rng.choice(rotations))
            for _ in range(30):
                pair = _make_pair(rng, **spec, rotation=rotation, mirror=bool(rng.integers(2)))
                sig = _sig(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct examples")
    # Demonstration order conveys no extra rule.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene, branches):
    mask = scene["donor"]["mask"]
    selected = {b["hit"]["index"] for b in branches if b["hit"] is not None}
    owner = {p: i for i, o in enumerate(scene["targets"]) for p in o["cells"]}
    shadows, touched, bbox_gaps = set(), set(), 0
    a = scene["grid"]
    for end, d, branch in zip(scene["ends"], scene["directions"], branches):
        hit = branch["hit"]
        r, c = end
        for t in range(1, 31):
            r, c = r + d[0], c + d[1]
            if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
                break
            if (r, c) in owner:
                touched.add(owner[r, c])
                if hit and t > hit["step"] and owner[r, c] not in selected:
                    shadows.add(owner[r, c])
            if hit and t < hit["step"]:
                tr, tc = scene["targets"][hit["index"]]["origin"]
                if tr <= r < tr + mask.shape[0] and tc <= c < tc + mask.shape[1]:
                    bbox_gaps += 1
    offray = set(range(len(scene["targets"]))) - touched
    source = scene["donor"]["origin"]
    distance = lambda i: sum((a - b) ** 2 for a, b in zip(scene["targets"][i]["origin"], source))
    nearer_offray = bool(selected and offray and min(map(distance, offray)) < min(map(distance, selected)))
    return {"hits": len(selected), "selected": selected, "shadows": shadows,
            "offray": offray, "nearer_offray": nearer_offray,
            "bbox_gaps": bbox_gaps, "steps": [b["hit"]["step"] for b in branches if b["hit"]],
            "trail_colors": [b["hit"]["color"] for b in branches if b["hit"]]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _sig(a) not in seen
            seen.add(_sig(a))
            before = a.copy()
            scene, expected = _parse(a), solve(a)
            assert np.array_equal(a, before), "solve mutated its input"
            assert np.array_equal(out, expected), "incorrect output"
            branches = _trace(scene)
            f = _features(scene, branches)
            assert f["hits"] >= 1 and f["offray"] and len(scene["targets"]) > f["hits"]
            lo, hi = (2, 4) if split == "train" else (5, 7)
            assert all(lo <= t <= hi for t in f["steps"])
            assert all(len(b["path"]) >= 1 for b in branches if b["hit"])
            allowed = set(scene["donor"]["cells"])
            for p in scene["donor"]["cells"]:
                assert out[p] == scene["bg"]
            for p in scene["frame"]["cells"]:
                assert out[p] == a[p]
            sr, sc = scene["donor"]["origin"]
            offsets = [tuple(map(int, p)) for p in np.argwhere(scene["donor"]["mask"])]
            without_a, without_b = out.copy(), out.copy()
            for branch in branches:
                hit = branch["hit"]
                if hit is None:
                    continue
                obj = scene["targets"][hit["index"]]
                tr, tc = obj["origin"]
                for r, c in offsets:
                    assert out[tr + r, tc + c] == a[sr + r, sc + c]
                    without_a[tr + r, tc + c] = hit["color"]
                for p in branch["path"]:
                    assert a[p] == scene["bg"] and out[p] == hit["color"]
                allowed |= set(obj["cells"]) | set(branch["path"])
            for i, obj in enumerate(scene["targets"]):
                if i not in f["selected"]:
                    tr, tc = obj["origin"]
                    for r, c in offsets:
                        assert out[tr + r, tc + c] == a[tr + r, tc + c]
                        without_b[tr + r, tc + c] = a[sr + r, sc + c]
            actual_changes = {tuple(map(int, p)) for p in np.argwhere(a != out)}
            assert actual_changes == allowed, "unexpected edit or missing change"
            assert not np.array_equal(without_a, out), "A ablation did not change answer"
            assert not np.array_equal(without_b, out), "B ablation did not change answer"
            details[split].append((scene, f))
    train = details["train"]
    assert any(s["donor"]["mask"].all() and f["hits"] == 2 and f["shadows"] and f["nearer_offray"]
               and len(set(f["trail_colors"])) == 2 for s, f in train)
    assert any(not s["donor"]["mask"].all() and f["hits"] == 1 and f["bbox_gaps"] for s, f in train)
    assert {s["axis"] for s, _ in train} == {0, 1}
    train_facings = {tuple(s["normal"]) for s, _ in train}
    for s, f in details["test"]:
        assert f["shadows"] and tuple(s["normal"]) in train_facings


def _has_hole(mask):
    h, w = mask.shape
    outside = set()
    stack = [(r, c) for r in range(h) for c in range(w)
             if not mask[r, c] and (r in (0, h - 1) or c in (0, w - 1))]
    outside.update(stack)
    while stack:
        r, c = stack.pop()
        for p, q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= p < h and 0 <= q < w and not mask[p, q] and (p, q) not in outside:
                outside.add((p, q))
                stack.append((p, q))
    return int((~mask).sum()) > len(outside)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded structural, counterfactual and metamorphic checks over 42 seeds."""
    shapes, masks, facings, counts, palettes, offsets = set(), set(), set(), set(), set(), set()
    holes = bbox_gaps = noncorner = nonmajority = long_arms = edge_frames = 0
    position_signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                s = _parse(a)
                branches = _trace(s)
                f = _features(s, branches)
                mask = s["donor"]["mask"]
                shapes.add(a.shape)
                masks.add(_sig(mask))
                facings.add(tuple(s["normal"]))
                counts.add(len(s["targets"]))
                palettes.add((s["bg"], next(iter(s["frame"]["colors"])), tuple(sorted(s["donor"]["colors"]))))
                position_signatures.add((a.shape, s["donor"]["origin"], tuple(o["origin"] for o in s["targets"])))
                holes += _has_hole(mask)
                bbox_gaps += f["bbox_gaps"]
                long_arms += max(s["arms"]) >= s["bar_length"]
                edge_frames += any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1) for r, c in s["frame"]["cells"])
                majority = Counter(int(a[p]) for p in s["donor"]["cells"]).most_common(1)[0][0]
                for b in branches:
                    if b["hit"]:
                        p = b["hit"]["offset"]
                        offsets.add((mask.shape, p))
                        noncorner += p not in ((0, 0), (0, mask.shape[1] - 1), (mask.shape[0] - 1, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
                        nonmajority += b["hit"]["color"] != majority
                # These transformations preserve the rule but change all literal roles/axes.
                if split == "test":
                    assert np.array_equal(solve(np.rot90(a)), np.rot90(out))
                    assert np.array_equal(solve(a[:, ::-1]), out[:, ::-1])
                    perm = np.random.default_rng(seed + 7000).permutation(10)
                    assert np.array_equal(solve(perm[a]), perm[out])
    assert len(shapes) >= 35 and len(masks) >= 35 and len(position_signatures) >= 150
    assert facings == {(1, 0), (-1, 0), (0, 1), (0, -1)}
    assert len(counts) >= 4 and len(palettes) >= 100 and len(offsets) >= 25
    assert min(holes, bbox_gaps, noncorner, nonmajority, long_arms, edge_frames) > 0
    assert _equal(generate_puzzle(993), generate_puzzle(993))
    assert not _equal(generate_puzzle(993), generate_puzzle(994))
    # Explicit regression: the diagonal traverses an empty bottom-right
    # bbox corner before contacting the cross's center, not its bbox edge.
    cross = {(0, 1): 2, (1, 0): 3, (1, 1): 4, (1, 2): 2, (2, 1): 3}
    a = np.zeros((16, 22), dtype=np.int8)
    a[10, 9:16] = 8
    a[11:15, 9] = a[11:15, 15] = 8
    for (r, c), color in cross.items():
        a[12 + r, 11 + c] = color
        a[5 + r, 4 + c] = a[1 + r, 11 + c] = 9
    expected = a.copy()
    for (r, c), color in cross.items():
        expected[12 + r, 11 + c] = 0
        expected[5 + r, 4 + c] = color
    for p in ((9, 8), (8, 7), (7, 6)):
        expected[p] = 4
    assert _trace(_parse(a))[0]["hit"]["offset"] == (1, 1)
    assert np.array_equal(solve(a), expected)
    # Both rays may miss; an unmatched ray has no arbitrary fallback color.
    for r, c in cross:
        a[5 + r, 4 + c] = 0
    expected = a.copy()
    for r, c in cross:
        expected[12 + r, 11 + c] = 0
    assert np.array_equal(solve(a), expected)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(50000 + 10 * train + test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {"seeds": 42, "shapes": len(shapes), "masks": len(masks),
            "layouts": len(position_signatures), "target_counts": sorted(counts),
            "palettes": len(palettes), "contact_offsets": len(offsets),
            "holed_masks": holes, "bbox_gap_steps": bbox_gaps,
            "noncorner_contacts": noncorner, "nonmajority_contacts": nonmajority,
            "long_arm_examples": long_arms, "edge_frame_examples": edge_frames}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
