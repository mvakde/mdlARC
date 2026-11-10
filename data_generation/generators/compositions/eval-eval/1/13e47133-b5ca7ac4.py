"""Magnetic contour welding: stable packing creates a cyclic contour domain.

Only the standard library and NumPy are used. No parent code or saved examples
are loaded. All semantic information needed by solve is visible in its input.
"""
from collections import deque
import hashlib
import json

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CHECK_SEEDS = (0, 791826961, 315389985, 246385855, 595603989, 997663024,
               426629126, 379628829, 939041167, 867059849, 766546575,
               664741943, 203827559, 485755342, 285768689, 654997103,
               266558938, 696189402, 127672241, 556888905, 304585896,
               563805672, 425784095, 483147479, 634490702, 964458135,
               387800611, 704607295, 558008709, 63824684, 48125082,
               73283513, 634712516, 130999032, 387606280, 968323936,
               311102948, 66231156, 702424641, 379398197, 8685801, 286472569)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b):
    return (_band(a, b) and max(a["c"], b["c"])
            < min(a["c"] + a["w"], b["c"] + b["w"]))


def _valid_geometry(objects, shape):
    h, w = shape
    return (all(1 <= o["r"] and o["r"] + o["h"] <= h - 1
                and 0 <= o["c"] and o["c"] + o["w"] <= w for o in objects)
            and not any(_overlap(a, b) for i, a in enumerate(objects)
                        for b in objects[i + 1:]))


def _pack(objects, shape, low, high):
    """Ordered interval constraints, without jumping any predecessor."""
    result = [dict(o) for o in objects]
    for frame in (low, high):
        order = [i for i, o in enumerate(objects) if o["frame"] == frame]
        order.sort(key=lambda i: (objects[i]["c"] if frame == low
                                 else -objects[i]["c"] - objects[i]["w"],
                                 objects[i]["r"]))
        settled = []
        for i in order:
            o = result[i]
            blockers = [p for p in settled if _band(o, p)]
            if frame == low:
                o["c"] = max((p["c"] + p["w"] for p in blockers), default=0)
                assert o["c"] <= objects[i]["c"]
            else:
                o["c"] = min((p["c"] for p in blockers), default=shape[1]) - o["w"]
                assert o["c"] >= objects[i]["c"]
            settled.append(o)
    assert _valid_geometry(result, shape), "packed classes intersect or exceed rails"
    return result


def _simulate(objects, shape, low, high):
    """Separate validation of packing via collision-limited single-cell steps."""
    result = [dict(o) for o in objects]
    for frame, delta in ((low, -1), (high, 1)):
        group = [o for o in result if o["frame"] == frame]
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            occupied[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = True
        for _ in range(shape[1] * max(1, len(group)) + 1):
            moved = False
            for o in group:
                r, c, h, w = (o[k] for k in ("r", "c", "h", "w"))
                occupied[r:r + h, c:c + w] = False
                nc = c + delta
                if 0 <= nc <= shape[1] - w and not np.any(occupied[r:r + h, nc:nc + w]):
                    o["c"] = nc
                    moved = True
                occupied[r:r + h, o["c"]:o["c"] + w] = True
            if not moved:
                break
        else:
            raise AssertionError("physical packing did not converge")
    assert _valid_geometry(result, shape), "packed classes intersect or exceed rails"
    return result


def _depth(mask, diagonal=True):
    """Input-only solver depth: propagate distances from the full boundary."""
    neighbors = DIR8 if diagonal else DIR4
    h, w = mask.shape
    d = np.full(mask.shape, -1, dtype=np.int16)
    q = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in neighbors):
            d[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


def _erode(mask):
    """Generation/check depth: repeatedly strip eight-neighbor boundaries."""
    h, w = mask.shape
    remaining = mask.copy()
    d = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while np.any(remaining):
        padded = np.pad(remaining, 1)
        inside = remaining.copy()
        for dr, dc in DIR8:
            inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        d[remaining & ~inside] = level
        remaining = inside
        level += 1
    return d


def _mask(objects, shape, frame):
    m = np.zeros(shape, dtype=bool)
    for o in objects:
        if o["frame"] == frame:
            m[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = True
    return m


def _paint(objects, shape, bg, method=_erode):
    out = np.full(shape, bg, dtype=np.int8)
    for frame in sorted({o["frame"] for o in objects}):
        period = next(o["period"] for o in objects if o["frame"] == frame)
        m = _mask(objects, shape, frame)
        d = method(m)
        out[m] = np.asarray(period, dtype=np.int8)[d[m] % len(period)]
    return out


def _sparse(objects, shape, bg):
    out = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        out[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = o["patch"]
    return out


def _with_rails(work, low, high):
    out = np.empty((work.shape[0], work.shape[1] + 2), dtype=np.int8)
    out[:, 1:-1] = work
    out[:, 0], out[:, -1] = low, high
    return out


def _decode_period(patch, bg, frame):
    h, w = patch.shape
    cues = [(int(r) + 1, int(c) + 1)
            for r, c in np.argwhere(patch[1:-1, 1:-1] != bg)]
    assert cues, "every piece needs a visible key"
    depths = {p: min(p[0], p[1], h - 1 - p[0], w - 1 - p[1]) for p in cues}
    end = max(depths.values())
    assert 1 <= end <= 3
    periods = set()
    for r, c, dr, dc in ((0, 0, 1, 1), (0, w - 1, 1, -1),
                         (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1)):
        if 2 * end + 1 > min(h, w):
            continue
        if all((rr, cc) == (r + depths[(rr, cc)] * dr,
                            c + depths[(rr, cc)] * dc) for rr, cc in cues):
            period = (frame,) + tuple(int(patch[r + j * dr, c + j * dc])
                                      for j in range(1, end + 1))
            periods.add(period)
    assert len(periods) == 1, "ambiguous or non-diagonal key"
    period = next(iter(periods))
    assert period[-1] != bg
    return period


def _decode(input_grid):
    g = _grid(input_grid)
    axes = []
    if (np.all(g[:, 0] == g[0, 0]) and np.all(g[:, -1] == g[0, -1])
            and g[0, 0] != g[0, -1]):
        axes.append(False)
    if (np.all(g[0] == g[0, 0]) and np.all(g[-1] == g[-1, 0])
            and g[0, 0] != g[-1, 0]):
        axes.append(True)
    assert len(axes) == 1, "input must visibly encode one rail axis"
    transposed = axes[0]
    canonical = g.T if transposed else g
    assert min(canonical.shape) >= 5
    low, high, bg = map(int, (canonical[0, 0], canonical[0, -1], canonical[0, 1]))
    assert len({low, high, bg}) == 3
    work = canonical[:, 1:-1]
    assert np.all(work[0] == bg) and np.all(work[-1] == bg)
    covered = np.zeros(work.shape, dtype=bool)
    objects = []
    for r in range(work.shape[0]):
        for c in range(work.shape[1]):
            if covered[r, c] or work[r, c] == bg:
                continue
            frame = int(work[r, c])
            assert frame in (low, high)
            assert r + 2 < work.shape[0] and c + 2 < work.shape[1]
            bottom, right = r + 2, c + 2
            while bottom < work.shape[0] and work[bottom, c + 1] != frame:
                bottom += 1
            while right < work.shape[1] and work[r + 1, right] != frame:
                right += 1
            assert bottom < work.shape[0] and right < work.shape[1]
            patch = work[r:bottom + 1, c:right + 1].copy()
            assert np.all(patch[0] == frame) and np.all(patch[-1] == frame)
            assert np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
            assert not np.any(np.isin(patch[1:-1, 1:-1], (low, high)))
            assert not np.any(covered[r:bottom + 1, c:right + 1])
            covered[r:bottom + 1, c:right + 1] = True
            objects.append(dict(r=r, c=c, h=bottom - r + 1, w=right - c + 1,
                                frame=frame, period=_decode_period(patch, bg, frame), patch=patch))
    assert objects and _valid_geometry(objects, work.shape)
    for frame in (low, high):
        assert len({o["period"] for o in objects if o["frame"] == frame}) <= 1
    assert np.all(covered | (work == bg))
    return transposed, bg, low, high, work, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode rails, objects and keys entirely from this input, then weld."""
    transposed, bg, low, high, work, objects = _decode(input_grid)
    settled = _pack(objects, work.shape, low, high)
    output = _with_rails(_paint(settled, work.shape, bg, _depth), low, high)
    return output.T.copy() if transposed else output


def _period(rng, frame, bg, rails, length):
    colors = [c for c in range(10) if c not in (bg, *rails)]
    entries = [int(c) for c in rng.choice(colors, length - 1, replace=False)]
    if length >= 3 and rng.random() < 0.45:
        entries[int(rng.integers(length - 2))] = bg
    return (frame, *entries)


def _appearance(rng, objects, periods, bg):
    for o in objects:
        h, w, frame = (o[k] for k in ("h", "w", "frame"))
        period = periods[frame]
        assert min(h, w) >= 2 * len(period) - 1
        patch = np.full((h, w), bg, dtype=np.int8)
        patch[0], patch[-1], patch[:, 0], patch[:, -1] = frame, frame, frame, frame
        r, c, dr, dc = _pick(rng, ((0, 0, 1, 1), (0, w - 1, 1, -1),
                                  (h - 1, 0, -1, 1), (h - 1, w - 1, -1, -1)))
        for j, color in enumerate(period[1:], 1):
            patch[r + j * dr, c + j * dc] = color
        o["period"], o["patch"] = period, patch


def _chain(rng, low, high, n, period_length, variable=False):
    """Compose a chain of pairwise adjacent span overlaps, not a fixed bitmap."""
    k = 2 * period_length - 1
    base_w = int(rng.integers(k, min(7, 26 // n) + 1))
    widths = [base_w] * n
    if variable:
        for i in rng.permutation(n):
            widths[int(i)] += int(rng.integers(min(2, 27 - sum(widths)) + 1))
    h = int(rng.integers(max(k, 5), 8))
    smallest_step = (h + 1) // 2
    steps = [smallest_step] * (n - 1)
    for i in rng.permutation(n - 1):
        cap = h - 1 if variable else h - 2
        steps[int(i)] += int(rng.integers(min(cap - smallest_step, 28 - h - sum(steps)) + 1))
    min_h, min_w = h + sum(steps) + 2, sum(widths)
    height = int(rng.integers(min_h, 31))
    width = int(rng.integers(min_w + 1, 29))
    row = int(rng.integers(1, height - h - sum(steps)))
    gaps = rng.multinomial(width - min_w - 1, np.full(n + 1, 1 / (n + 1)))
    gaps[0] += 1
    col = int(gaps[0])
    frame = _pick(rng, (low, high))
    reverse_rows = bool(rng.integers(2))
    objects = []
    for i, w in enumerate(widths):
        r = height - row - h if reverse_rows else row
        c = col if frame == low else width - col - w
        objects.append(dict(r=r, c=c, h=h, w=w, frame=frame))
        row += steps[i] if i < n - 1 else 0
        col += w + int(gaps[i + 1])
    assert _valid_geometry(objects, (height, width))
    return (height, width), objects, frame


def _add(rng, objects, shape, low, high, periods, target):
    """Bounded independent proposals, retaining only feasible geometry."""
    objects = [dict(o) for o in objects]
    for _ in range(max(0, target - len(objects))):
        for _ in range(65):
            frame = _pick(rng, (low, high))
            minimum = 2 * len(periods[frame]) - 1
            if minimum > min(shape[0] - 2, shape[1]):
                continue
            h = int(rng.integers(minimum, min(12, shape[0] - 2) + 1))
            w = int(rng.integers(minimum, min(9, shape[1]) + 1))
            r = int(rng.integers(1, shape[0] - h))
            c = int(rng.integers(shape[1] - w + 1))
            if objects and rng.random() < 0.4:
                other = _pick(rng, objects)
                r = _pick(rng, (other["r"], other["r"] + other["h"] - 1,
                                other["r"] - h + 1, other["r"] + other["h"]))
            candidate = objects + [dict(r=r, c=c, h=h, w=w, frame=frame)]
            if not _valid_geometry(candidate, shape):
                continue
            try:
                _pack(candidate, shape, low, high)
            except AssertionError:
                continue
            objects = candidate
            break
    return objects


def _make_pair(rng, index, test, transpose, max_length=4):
    bg, low, high = (int(c) for c in rng.choice(10, 3, replace=False))
    periods = {f: _period(rng, f, bg, (low, high), int(rng.integers(2, max_length + 1)))
               for f in (low, high)}
    if index == 0 and not test:
        # A thin aligned pair creates depth 2 only after welding. A third
        # opposite-class piece carries the same first interior color.
        height, width = int(rng.integers(13, 23)), int(rng.integers(11, 16))
        h = int(rng.integers(9, min(12, height - 1)))
        r = int(rng.integers(1, height - h))
        objects = [dict(r=r, c=width - 6, h=h, w=3, frame=low),
                   dict(r=r, c=width - 3, h=h, w=3, frame=low)]
        hh = int(rng.integers(5, min(10, height - 1)))
        objects.append(dict(r=int(rng.integers(r, r + h - hh + 1)), c=0,
                            h=hh, w=width - 6, frame=high))
        colors = [c for c in range(10) if c not in (bg, low, high)]
        a, b = (int(c) for c in rng.choice(colors, 2, replace=False))
        periods = {low: (low, a), high: (high, a, b)}
        shape = (height, width)
    elif index == 1 and not test:
        shape, objects, frame = _chain(rng, low, high, 3, 3)
        cue = int(_pick(rng, [c for c in range(10) if c not in (bg, low, high)]))
        periods[frame] = (frame, bg, cue)
    elif test:
        length = int(rng.integers(2, min(3, max_length) + 1))
        n = int(rng.integers(4, (5 if length == 3 else 6) + 1))
        shape, objects, frame = _chain(rng, low, high, n, length, variable=True)
        periods[frame] = _period(rng, frame, bg, (low, high), length)
        objects = _add(rng, objects, shape, low, high, periods,
                       min(9, n + int(rng.integers(3))))
    else:
        shape = (int(rng.integers(10, 31)), int(rng.integers(10, 29)))
        objects = _add(rng, [], shape, low, high, periods, int(rng.integers(1, 10)))
        assert objects
    assert _valid_geometry(objects, shape)
    _appearance(rng, objects, periods, bg)
    settled = _pack(objects, shape, low, high)
    inp = _with_rails(_sparse(objects, shape, bg), low, high)
    out = _with_rails(_paint(settled, shape, bg), low, high)
    if transpose:
        inp, out = inp.T.copy(), out.T.copy()
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, 2, 10, "num_train")
    ne = _integer(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    first_axis = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    max_length = 3
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            axis = (first_axis if index == 0 else not first_axis) if split == "train" and index < 2 else bool(rng.integers(2))
            for _ in range(40):
                pair = _make_pair(rng, index, split == "test", axis,
                                  max_length if split == "test" else 4)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("bounded uniqueness search exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                max_length = max(max_length, max(len(o["period"]) for o in _decode(pair["input"])[-1]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _individual(objects, shape, bg):
    out = np.full(shape, bg, dtype=np.int8)
    for o in objects:
        rr, cc = np.indices((o["h"], o["w"]))
        d = np.minimum.reduce((rr, cc, o["h"] - 1 - rr, o["w"] - 1 - cc))
        out[o["r"]:o["r"] + o["h"], o["c"]:o["c"] + o["w"]] = np.asarray(o["period"])[d % len(o["period"])]
    return out


def _edge_first(objects, shape, low, high):
    result = [dict(o) for o in objects]
    for frame in (low, high):
        order = sorted((i for i, o in enumerate(objects) if o["frame"] == frame),
                       key=lambda i: objects[i]["c"] if frame == low else -objects[i]["c"] - objects[i]["w"])
        done = []
        for i in order:
            o = result[i]
            positions = range(shape[1] - o["w"] + 1)
            for c in positions if frame == low else reversed(positions):
                trial = dict(o, c=c)
                if not any(_overlap(trial, p) for p in done):
                    o["c"] = c
                    done.append(o)
                    break
    return result


def _measure(pair):
    axis, bg, low, high, work, objects = _decode(pair["input"])
    settled = _simulate(objects, work.shape, low, high)
    canonical_output = np.asarray(pair["output"]).T if axis else np.asarray(pair["output"])
    expected = _paint(settled, work.shape, bg)
    assert np.array_equal(canonical_output, _with_rails(expected, low, high)), "incorrect welded contours"
    assert [o["c"] for o in settled] == [o["c"] for o in _pack(objects, work.shape, low, high)]
    moved = {a["frame"] for a, b in zip(objects, settled) if a["c"] != b["c"]}
    masks = [_mask(settled, work.shape, f) for f in (low, high)]
    depths = [_erode(m) for m in masks]
    for m, d in zip(masks, depths):
        assert np.array_equal(d, _depth(m))
    box = np.full(work.shape, bg, dtype=np.int8)
    emerging = False
    graph_depth = 0
    for frame, m, d in zip((low, high), masks, depths):
        group = [o for o in settled if o["frame"] == frame]
        if not group:
            continue
        period = group[0]["period"]
        rr, cc = np.nonzero(m)
        bd = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        box[m] = np.asarray(period)[bd % len(period)]
        emerging |= int(d.max()) > max((min(o["h"], o["w"]) - 1) // 2 for o in group)
        ordered = sorted((o for o in objects if o["frame"] == frame),
                         key=lambda o: o["c"] if frame == low else -o["c"] - o["w"])
        levels = []
        for i, o in enumerate(ordered):
            levels.append(1 + max((levels[j] for j in range(i) if _band(o, ordered[j])), default=0))
        graph_depth = max(graph_depth, max(levels))
    contacts = int(np.count_nonzero(masks[0][:-1] & masks[1][1:]))
    contacts += int(np.count_nonzero(masks[1][:-1] & masks[0][1:]))
    contacts += int(np.count_nonzero(masks[0][:, :-1] & masks[1][:, 1:]))
    contacts += int(np.count_nonzero(masks[1][:, :-1] & masks[0][:, 1:]))
    all_depth = _erode(masks[0] | masks[1])
    mixed_materials = np.full(work.shape, bg, dtype=np.int8)
    for frame, m in zip((low, high), masks):
        group = [o for o in objects if o["frame"] == frame]
        if group:
            period = group[0]["period"]
            mixed_materials[m] = np.asarray(period)[all_depth[m] % len(period)]
    periods = {o["period"] for o in objects}
    return {"axis": int(axis), "shape": tuple(pair["input"].shape), "bg": bg,
            "rails": (low, high), "count": len(objects),
            "classes": tuple(sum(o["frame"] == f for o in objects) for f in (low, high)),
            "footprints": tuple(sorted({(o["h"], o["w"]) for o in objects})),
            "period_lengths": tuple(sorted({len(p) for p in periods})),
            "neutral_key": any(bg in p[1:] for p in periods),
            "moved": moved, "all_frames": {low, high}, "emerging_depth": emerging,
            "graph_depth": graph_depth, "opposite_contacts": contacts,
            "stationary": sum(a["c"] == b["c"] for a, b in zip(objects, settled)),
            "welding_witness": not np.array_equal(expected, _individual(settled, work.shape, bg)),
            "omit_A_witness": not np.array_equal(expected, _sparse(settled, work.shape, bg)),
            "omit_B_witness": not np.array_equal(expected, _paint(objects, work.shape, bg)),
            "stable_witness": [o["c"] for o in settled] != [o["c"] for o in _edge_first(objects, work.shape, low, high)],
            "material_witness": not np.array_equal(expected, mixed_materials),
            "box_witness": not np.array_equal(expected, box),
            "diagonal_witness": not np.array_equal(expected, _paint(settled, work.shape, bg, lambda m: _depth(m, False))),
            "shared_cue": any(a["frame"] != b["frame"] and a["period"][1] == b["period"][1]
                              for a in objects for b in objects),
            "geometry": hashlib.sha256(bytes(work.shape) + b"".join(m.tobytes() for m in masks)).hexdigest()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            records[split].append(_measure({"input": inp, "output": out}))
    train = records["train"]
    assert {r["axis"] for r in train} == {0, 1}, "demonstrate both visible rail axes"
    for feature in ("welding_witness", "omit_A_witness", "omit_B_witness", "stable_witness",
                    "box_witness", "diagonal_witness", "emerging_depth", "neutral_key", "shared_cue", "material_witness"):
        assert any(r[feature] for r in train), f"training lacks {feature}"
    assert any(r["moved"] == r["all_frames"] for r in train)
    train_max = max(max(r["period_lengths"]) for r in train)
    assert {2, 3} <= {n for r in train for n in r["period_lengths"]}
    for record in records["test"]:
        assert record["graph_depth"] >= 4
        assert max(record["period_lengths"]) <= train_max


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain bounded signatures/measurements, never grids."""
    signatures, shapes, roles, geometry, counts, footprints, lengths = (set() for _ in range(7))
    axes, contacts, stationary, class_counts = set(), set(), set(), set()
    pairs = 0
    for seed in CHECK_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                r = _measure(pair)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(r["shape"])
                roles.add((r["bg"], *r["rails"]))
                geometry.add(r["geometry"])
                counts.add(r["count"])
                footprints.update(r["footprints"])
                lengths.update(r["period_lengths"])
                axes.add(r["axis"])
                contacts.add(bool(r["opposite_contacts"]))
                stationary.add(bool(r["stationary"]))
                class_counts.add(r["classes"])
                pairs += 1
    assert len(signatures) == 42 and len(shapes) >= 50 and len(geometry) >= 120
    assert len(roles) >= 80 and len(footprints) >= 20 and len(counts) >= 6
    assert axes == {0, 1} and lengths == {2, 3, 4}
    assert contacts == {False, True} and stationary == {False, True}
    assert any(0 in c for c in class_counts) and any(min(c) > 0 for c in class_counts)
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        a, b = generate_puzzle(123456, nt, ne), generate_puzzle(123456, nt, ne)
        assert _digest(a) == _digest(b)
        assert len(a["train"]) == nt and len(a["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0, 2, 1)
    for damage in ("contour", "rail", "input_rail", "unwelded", "overtaking"):
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = broken["test"][0]
        if damage == "contour":
            r, c = map(int, np.argwhere(pair["input"] != pair["output"])[0])
            pair["output"][r, c] = pair["input"][r, c]
        elif damage == "rail":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif damage == "input_rail":
            pair["input"][0, 0] = (int(pair["input"][0, 0]) + 1) % 10
        else:
            pair = broken["train"][0 if damage == "unwelded" else 1]
            axis, bg, low, high, work, objects = _decode(pair["input"])
            if damage == "unwelded":
                changed = _individual(_pack(objects, work.shape, low, high), work.shape, bg)
            else:
                changed = _paint(_edge_first(objects, work.shape, low, high), work.shape, bg)
            changed = _with_rails(changed, low, high)
            pair["output"] = changed.T.copy() if axis else changed
            assert not np.array_equal(pair["output"], original["train"][0 if damage == "unwelded" else 1]["output"])
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted damaged {damage}")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "role_assignments": len(roles),
            "packed_geometries": len(geometry), "piece_counts": sorted(counts),
            "footprints": len(footprints), "period_lengths": sorted(lengths)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
