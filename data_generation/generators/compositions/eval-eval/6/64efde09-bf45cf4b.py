"""Coincidence loom: a stamp is also the coordinate key for colored emitters.

Only NumPy and the standard library are used. No files, cached examples, or
generation parameters are consulted by solve. Experiments in self_test stream
one puzzle at a time; generation retains at most 14 pairs and one candidate.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    answer = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        answer.append(cells)
    return answer


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _emitter(template, t, r, c):
    a = _orient(template, t)
    return {"a": a, "labels": _orient(np.arange(template.size).reshape(template.shape), t),
            "r": r, "c": c, "t": t,
            "cells": {(r+i, c+j) for i in range(a.shape[0]) for j in range(a.shape[1])}}


def _ray(emitter, side, slot, shape, canvas):
    th, length = shape
    edge, inner = (0, 1) if side == 0 else (th-1, th-2)
    labels = emitter["labels"]
    ar, ac = map(int, np.argwhere(labels == edge*length+slot)[0])
    ir, ic = map(int, np.argwhere(labels == inner*length+slot)[0])
    dr, dc = ar-ir, ac-ic
    r, c = emitter["r"]+ar+dr, emitter["c"]+ac+dc
    cells = []
    while 0 <= r < canvas[0] and 0 <= c < canvas[1]:
        cells.append((r, c))
        r, c = r+dr, c+dc
    return cells, (dr, dc)


def _weave(emitters, code, side, shape, canvas, allocations=None):
    """Ray intersections, including mismatches, determine the coarse crop."""
    horizontal, vertical = {}, {}
    for i, e in enumerate(emitters):
        for slot, color in code.items():
            if allocations is not None and slot not in allocations[i]:
                continue
            ray, direction = _ray(e, side, slot, shape, canvas)
            target = horizontal if direction[1] else vertical
            for p in ray:
                target.setdefault(p, set()).add(color)
    crossings = horizontal.keys() & vertical.keys()
    assert crossings, "at least one perpendicular crossing is required"
    r0, r1, c0, c1 = _box(crossings)
    mask = np.zeros((r1-r0, c1-c0), dtype=bool)
    for r, c in crossings:
        mask[r-r0, c-c0] = bool(horizontal[r, c] & vertical[r, c])
    return mask, (r0, r1, c0, c1), horizontal, vertical


def _block_write(mask, stamp, bg):
    sh, sw = stamp.shape
    out = np.full((mask.shape[0]*sh, mask.shape[1]*sw), bg, dtype=np.int8)
    for r, c in np.argwhere(mask):
        out[r*sh:(r+1)*sh, c*sw:(c+1)*sw] = stamp
    return out


def _read(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert counts.max() > a.size/2, "background must be a strict majority"
    frames = []
    for color in map(int, colors):
        if color == bg:
            continue
        cells = set(map(tuple, np.argwhere(a == color)))
        r0, r1, c0, c1 = _box(cells)
        if r1-r0 < 6 or c1-c0 < 6:
            continue
        boundary = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                    if r in (r0, r1-1) or c in (c0, c1-1)}
        if cells != boundary:
            continue
        stamp = a[r0+2:r1-2, c0+2:c1-2].copy()
        if min(stamp.shape) < 2 or min(stamp.shape) >= max(stamp.shape):
            continue
        if not np.all(stamp != bg) or color in stamp:
            continue
        interior = a[r0+1:r1-1, c0+1:c1-1].copy()
        interior[1:-1, 1:-1] = bg
        if not np.all(interior == bg):
            continue
        frames.append((color, (r0, r1, c0, c1), stamp))
    assert len(frames) == 1, "one isolated rectangular stamp frame is required"
    frame_color, frame, stamp = frames[0]
    canonical = stamp if stamp.shape[0] < stamp.shape[1] else np.rot90(stamp)
    th, length = canonical.shape
    assert 2 <= th <= 4 and th < length <= 10
    structural = set(map(int, np.unique(stamp)))
    assert 2 <= len(structural) <= 4
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_key(f) for f in forms}) == 8, "template must fix every correspondence"
    r0, r1, c0, c1 = frame
    reserved = {(r, c) for r in range(r0, r1) for c in range(c0, c1)}
    outside = np.isin(a, list(structural))
    outside[r0:r1, c0:c1] = False
    emitters = []
    for component in _components(outside):
        er0, er1, ec0, ec1 = _box(component)
        obj = a[er0:er1, ec0:ec1]
        assert obj.size == len(component)
        ts = [t for t, f in enumerate(forms) if np.array_equal(f, obj)]
        assert len(ts) == 1, "every unframed object must be an exact transformed stamp"
        assert er0 >= 1 and ec0 >= 1 and er1 < a.shape[0] and ec1 < a.shape[1]
        emitters.append(_emitter(canonical, ts[0], er0, ec0))
    assert 2 <= len(emitters) <= 6
    occupied = reserved | set().union(*(e["cells"] for e in emitters))
    for e in emitters:
        assert not any((r+dr, c+dc) in reserved for r, c in e["cells"]
                       for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
    signals = []
    for r, c in np.argwhere((a != bg) & ~np.isin(a, list(structural | {frame_color}))):
        r, c = int(r), int(c)
        assert r in (0, a.shape[0]-1) or c in (0, a.shape[1]-1)
        signals.append((r, c, int(a[r, c])))
    assert signals
    corridors = {}
    for i, e in enumerate(emitters):
        for side in (0, 1):
            for slot in range(length):
                ray, direction = _ray(e, side, slot, canonical.shape, a.shape)
                if len(ray) >= 2 and not occupied.intersection(ray):
                    if all(int(a[p]) == bg for p in ray[:-1]):
                        corridors.setdefault(ray[-1], []).append((i, side, slot))
    code, sides = {}, set()
    allocations = [set() for _ in emitters]
    for r, c, color in signals:
        matches = corridors.get((r, c), [])
        assert len(matches) == 1, "boundary clue has competing interpretations"
        i, side, slot = matches[0]
        assert slot not in code or code[slot] == color, "inconsistent pooled code"
        code[slot] = color
        sides.add(side)
        allocations[i].add(slot)
    assert len(sides) == 1 and 1 <= len(code) <= 5
    side = sides.pop()
    rays, directions, lengths = [], [], []
    for e in emitters:
        for slot, color in code.items():
            ray, direction = _ray(e, side, slot, canonical.shape, a.shape)
            assert len(ray) >= 2 and not occupied.intersection(ray), "ray hits an object"
            assert all(int(a[p]) in (bg, color) for p in ray)
            rays.append((set(ray), direction))
            directions.append(direction)
            lengths.append(len(ray))
    for i, (cells, d) in enumerate(rays):
        for other, od in rays[:i]:
            if bool(d[0]) == bool(od[0]):
                assert not cells.intersection(other), "collinear rays are outside the family"
    assert any(d[0] for d in directions) and any(d[1] for d in directions)
    mask, crop, horizontal, vertical = _weave(emitters, code, side, canonical.shape, a.shape)
    assert mask.shape[0]*stamp.shape[0] <= 30 and mask.shape[1]*stamp.shape[1] <= 30
    assert mask.any(), "there must be a matching crossing"
    return {"grid": a, "bg": bg, "stamp": stamp, "canonical": canonical,
            "emitters": emitters, "code": code, "side": side, "mask": mask,
            "crop": crop, "horizontal": horizontal, "vertical": vertical,
            "allocations": allocations, "frame": frame, "frame_color": frame_color,
            "ray_lengths": lengths, "directions": directions}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    mask, stamp, bg = info["mask"], info["stamp"], info["bg"]
    sh, sw = stamp.shape
    rr, cc = np.indices((mask.shape[0]*sh, mask.shape[1]*sw))
    return np.where(mask[rr//sh, cc//sw], stamp[rr % sh, cc % sw], bg).astype(np.int8)


def _template(rng, thickness, length, colors):
    for _ in range(32):
        style = int(rng.integers(4))
        if style == 0:
            a = rng.choice(colors, (thickness, length)).astype(np.int8)
        else:
            rr, cc = np.indices((thickness, length))
            labels = (rr + (cc // int(rng.integers(1, 4)) if style == 1 else cc)) % len(colors)
            if style == 3:
                labels[:] = 0
                labels[1:, 1:] = 1
            a = np.array(colors, dtype=np.int8)[labels]
            a[0, -1] = colors[-1]
        for pos, color in zip(rng.permutation(a.size), colors):
            a.flat[int(pos)] = color
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((thickness, length), colors[0], dtype=np.int8)
    a[1:, 1:] = colors[1]
    for i, color in enumerate(colors[2:]):
        a[-1, i+1] = color
    assert len({_key(_orient(a, t)) for t in range(8)}) == 8
    return a


def _direction(template, t, side):
    e = _emitter(template, t, 0, 0)
    return _ray(e, side, 0, template.shape, (30, 30))[1]


def _candidate(rng, split, index, regime, palette):
    witness = split == "train" and index < 2
    length = int(rng.integers(6, 9) if split == "test" and regime == "stamp_growth"
                 else rng.integers(3, 6) if regime == "stamp_growth" else rng.integers(3, 9))
    thickness = int(rng.integers(2, min(3, length-1)+1))
    # Short templates permit a genuine increase in interacting emitter count.
    if regime == "branch_growth":
        length, thickness = int(rng.integers(3, 5)), 2
    nstruct = int(rng.integers(2, 5))
    bg, frame_color = palette[:2]
    structural = palette[2:2+nstruct]
    signal_palette = palette[2+nstruct:]
    template = _template(rng, thickness, length, structural)
    stamp = _orient(template, int(rng.integers(8)))
    maxmh, maxmw = 30//stamp.shape[0], 30//stamp.shape[1]
    span_limit = min(length, maxmh, maxmw, 5)
    span = int(rng.integers(3 if witness else 2, span_limit+1))
    start = int(rng.integers(length-span+1))
    slots = {start, start+span-1}
    for slot in range(start+1, start+span-1):
        if not (witness and index == 0) and rng.random() < .7:
            slots.add(slot)
    slots = sorted(slots)
    if witness and index == 1 and len(slots) < 3:
        return None
    # Both witnesses distinguish color equality from mere crossing. The second
    # also deliberately repeats a color when the visible slot set permits it.
    values = [signal_palette[i % min(len(slots), len(signal_palette))] for i in range(len(slots))]
    if len(values) >= 3 and ((witness and index == 1) or rng.random() < .4):
        values[-1] = values[0]
    rng.shuffle(values)
    code = dict(zip(slots, values))
    side = int(rng.integers(2))
    gap_h, gap_w = map(int, rng.integers(1, 4, size=2))
    cap_h = min(4, 1+(maxmh-span)//(length+gap_h))
    cap_v = min(4, 1+(maxmw-span)//(length+gap_w))
    choices = [(nh, nv) for nh in range(1, cap_h+1) for nv in range(1, cap_v+1)
               if nh+nv <= 6]
    if witness and index == 0:
        choices = [(1, 1)]
    elif regime == "branch_growth":
        target = 2 if split == "train" else min(4, max(nh+nv for nh, nv in choices))
        choices = [(nh, nv) for nh, nv in choices if nh+nv == target]
    elif witness and index == 1:
        rich = [(nh, nv) for nh, nv in choices if nh+nv >= 3]
        choices = rich or choices
    nh, nv = choices[int(rng.integers(len(choices)))]
    coreh, corew = (nh-1)*(length+gap_h)+span, (nv-1)*(length+gap_w)+span
    height = int(rng.integers(max(18, coreh+length+5), 31))
    width = int(rng.integers(max(18, corew+length+5), 31))
    if height < coreh+2*length or width < corew+2*length:
        return None
    cr = int(rng.integers(length, height-coreh-length+1))
    cc = int(rng.integers(length, width-corew-length+1))
    emitters = []
    for axis, count in (("h", nh), ("v", nv)):
        for j in range(count):
            desired = ((0, 1), (0, -1))[int(rng.integers(2))] if axis == "h" else (
                (1, 0), (-1, 0))[int(rng.integers(2))]
            ts = [t for t in range(8) if _direction(template, t, side) == desired]
            t = ts[int(rng.integers(len(ts)))]
            e = _emitter(template, t, 0, 0)
            active = []
            edge_row = 0 if side == 0 else thickness-1
            for slot in slots:
                p = np.argwhere(e["labels"] == edge_row*length+slot)[0]
                active.append(int(p[0] if axis == "h" else p[1]))
            # Test distance extrapolation uses exactly the demonstrated geometry.
            low, high = (6, 11) if split == "test" and regime == "distance" else (2, 5)
            distance = int(rng.integers(low, high))
            if axis == "h":
                r = cr+j*(length+gap_h)-min(active)
                c = cc-distance-thickness+1 if desired[1] > 0 else cc+corew-1+distance
            else:
                c = cc+j*(length+gap_w)-min(active)
                r = cr-distance-thickness+1 if desired[0] > 0 else cr+coreh-1+distance
            e = _emitter(template, t, r, c)
            if min(r, c) < 1 or r+e["a"].shape[0] >= height or c+e["a"].shape[1] >= width:
                return None
            if any(max(abs(x-u), abs(y-v)) <= 1 for x, y in e["cells"]
                   for old in emitters for u, v in old["cells"]):
                return None
            emitters.append(e)
    if witness and index == 0:
        # Put the perpendicular emitter BETWEEN the two selected horizontal
        # ray rows. One horizontal ray misses it; the other crosses both of
        # its colors. The unequal outer crossing must remain in the crop.
        # This makes cropping all crossings observably different from cropping
        # only the successful matches, without adding an extra operator.
        h, v = emitters
        edge_row = 0 if side == 0 else thickness-1
        selected_rows = sorted(h["r"]+int(np.argwhere(h["labels"] == edge_row*length+s)[0, 0])
                               for s in slots)
        low, high = selected_rows[0], selected_rows[-1]
        if high-low <= thickness:
            return None
        vr = int(rng.integers(low+1, high-thickness+1))
        emitters[1] = _emitter(template, v["t"], vr, v["c"])
    occupied = set().union(*(e["cells"] for e in emitters))
    rays = []
    for i, e in enumerate(emitters):
        for slot in slots:
            ray, direction = _ray(e, side, slot, template.shape, (height, width))
            if len(ray) < 2 or occupied.intersection(ray):
                return None
            if any(bool(direction[0]) == bool(d[0]) and set(ray).intersection(old)
                   for old, d in rays):
                return None
            rays.append((set(ray), direction))
    ray_cells = set().union(*(r for r, _ in rays))
    fh, fw = stamp.shape[0]+4, stamp.shape[1]+4
    possible = [(r, c) for r in range(1, height-fh) for c in range(1, width-fw)]
    rng.shuffle(possible)
    frame = None
    for r, c in possible:
        area = {(i, j) for i in range(r, r+fh) for j in range(c, c+fw)}
        if area & ray_cells:
            continue
        if any((i+dr, j+dc) in occupied for i, j in area
               for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))):
            continue
        frame = (r, c, area)
        break
    if frame is None:
        return None
    r, c, area = frame
    a = np.full((height, width), bg, dtype=np.int8)
    a[r:r+fh, c:c+fw] = frame_color
    a[r+1:r+fh-1, c+1:c+fw-1] = bg
    a[r+2:r+fh-2, c+2:c+fw-2] = stamp
    for e in emitters:
        er, ec = e["r"], e["c"]
        eh, ew = e["a"].shape
        a[er:er+eh, ec:ec+ew] = e["a"]
    occupied |= area
    # Clue attribution considers every slot and BOTH long sides, not just the
    # hidden sampled code, before any clue is selected.
    alignments = {}
    for i, e in enumerate(emitters):
        for candidate_side in (0, 1):
            for slot in range(length):
                ray, _ = _ray(e, candidate_side, slot, template.shape, a.shape)
                if len(ray) >= 2 and not occupied.intersection(ray):
                    alignments.setdefault(ray[-1], []).append((i, candidate_side, slot))
    choices = {}
    for slot in slots:
        choices[slot] = []
        for i, e in enumerate(emitters):
            ray, _ = _ray(e, side, slot, template.shape, a.shape)
            if alignments.get(ray[-1]) == [(i, side, slot)]:
                choices[slot].append((i, ray[-1]))
        if not choices[slot]:
            return None
    allocation_mode = "split" if witness and index == 0 else str(rng.choice(["split", "full", "redundant"]))
    chosen = []
    if allocation_mode == "full":
        eligible = [i for i in range(len(emitters)) if all(any(k == i for k, _ in choices[s]) for s in slots)]
        if not eligible:
            return None
        which = eligible[int(rng.integers(len(eligible)))]
        chosen = [(s, *next(p for p in choices[s] if p[0] == which)) for s in slots]
    else:
        for j, slot in enumerate(slots):
            options = choices[slot]
            if witness and index == 0:
                # Exactly two perpendicular copies; neither shows a full code.
                options = [p for p in options if p[0] == j % 2]
                if not options:
                    return None
            which, endpoint = options[int(rng.integers(len(options)))]
            chosen.append((slot, which, endpoint))
        if allocation_mode == "redundant":
            for slot in slots:
                for which, endpoint in choices[slot]:
                    if rng.random() < .35 and (slot, which, endpoint) not in chosen:
                        chosen.append((slot, which, endpoint))
    for slot, _, endpoint in chosen:
        if a[endpoint] not in (bg, code[slot]):
            return None
        a[endpoint] = code[slot]
    if np.count_nonzero(a == bg) <= a.size/2:
        return None
    mask, _, _, _ = _weave(emitters, code, side, template.shape, a.shape)
    if not mask.any() or np.all(mask) or mask.shape[0]*stamp.shape[0] > 30 or mask.shape[1]*stamp.shape[1] > 30:
        return None
    return {"input": a, "output": _block_write(mask, stamp, bg)}


def _no_pool_mask(info):
    # Crop remains the full-code crossing crop, so deleting rays cannot hide a
    # difference merely by shrinking the answer rectangle.
    r0, r1, c0, c1 = info["crop"]
    out = np.zeros((r1-r0, c1-c0), dtype=bool)
    h, v = {}, {}
    for i, e in enumerate(info["emitters"]):
        for s in info["allocations"][i]:
            ray, d = _ray(e, info["side"], s, info["canonical"].shape, info["grid"].shape)
            target = h if d[1] else v
            for p in ray:
                target.setdefault(p, set()).add(info["code"][s])
    for r, c in h.keys() & v.keys():
        if r0 <= r < r1 and c0 <= c < c1:
            out[r-r0, c-c0] = bool(h[r, c] & v[r, c])
    return out


def _reversal_changes(info):
    """Ablate the longitudinal correspondence on one copy at a time."""
    length = info["canonical"].shape[1]
    for i, e in enumerate(info["emitters"]):
        altered = list(info["emitters"])
        labels = e["labels"]
        altered[i] = dict(e, labels=(labels//length)*length+length-1-labels % length)
        try:
            mask, crop, _, _ = _weave(altered, info["code"], info["side"],
                                      info["canonical"].shape, info["grid"].shape)
        except AssertionError:
            return True
        if crop != info["crop"] or not np.array_equal(mask, info["mask"]):
            return True
    return False


def _features(info):
    e = info["emitters"]
    crossings = info["horizontal"].keys() & info["vertical"].keys()
    mismatches = sum(not (info["horizontal"][p] & info["vertical"][p]) for p in crossings)
    distances = []
    for emitter in e:
        ray, _ = _ray(emitter, info["side"], next(iter(info["code"])),
                      info["canonical"].shape, info["grid"].shape)
        indices = [i for i, p in enumerate(ray) if p in crossings]
        if indices:
            distances.append(min(indices))
    return {"input_shape": info["grid"].shape, "stamp_shape": info["stamp"].shape,
            "mask_shape": info["mask"].shape, "length": info["canonical"].shape[1],
            "emitters": len(e), "code_slots": len(info["code"]),
            "colors": len(set(info["code"].values())), "background": info["bg"],
            "active": int(info["mask"].sum()), "mismatches": mismatches,
            "blank_outer_line": any(not line.any() for line in
                                   (info["mask"][0], info["mask"][-1],
                                    info["mask"][:, 0], info["mask"][:, -1])),
            "gap_cells": info["mask"].size-len(crossings),
            "pooled": all(len(s) < len(info["code"]) for s in info["allocations"]),
            "full": any(len(s) == len(info["code"]) for s in info["allocations"]),
            "redundant": sum(map(len, info["allocations"])) > len(info["code"]),
            "repeated_color": len(set(info["code"].values())) < len(info["code"]),
            "directions": tuple(sorted(set(info["directions"]))),
            "near_crossing": min(distances), "far_crossing": max(distances),
            "reversal_changes": _reversal_changes(info),
            "no_pool_changes": not np.array_equal(_no_pool_mask(info), info["mask"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            assert np.array_equal(solve(inp), output), "incorrect coincidence substitution"
            sh, sw = info["stamp"].shape
            # Each block independently has exactly the reference motif or is
            # background; scale products and orientation must both be correct.
            assert output.shape == (info["mask"].shape[0]*sh, info["mask"].shape[1]*sw)
            for r, c in np.ndindex(info["mask"].shape):
                block = output[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                if info["mask"][r, c]:
                    assert np.array_equal(block, info["stamp"])
                else:
                    assert np.all(block == info["bg"])
            assert info["frame_color"] not in output
            assert not set(info["code"].values()).intersection(map(int, output.flat))
            records[split].append(_features(info))
    train = records["train"]
    assert any(f["pooled"] and f["no_pool_changes"] for f in train), "pooling needs an observable witness"
    assert any(f["reversal_changes"] for f in train), "longitudinal correspondence must affect the answer"
    assert any(f["mismatches"] > 0 for f in train), "distinguish equality from all crossings"
    assert any(f["blank_outer_line"] for f in train), "demonstrate the all-crossings crop"
    assert len({f["stamp_shape"] for f in train}) >= 2, "expose independent tile dimensions"
    assert any(f["gap_cells"] > 0 for f in train), "show retained coarse spacing"
    return None


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["stamp_growth", "branch_growth", "distance"]))
    stable = bool(rng.random() < .2)
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen, train_infos = {"train": [], "test": []}, set(), []
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            # A maximum of 300 small candidates, no retained retry payloads.
            for attempt in range(300):
                colors = palette if stable else list(map(int, rng.permutation(10)))
                pair = _candidate(rng, split, index, regime, colors)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                try:
                    info = _read(pair["input"])
                except AssertionError:
                    continue
                if not np.array_equal(_block_write(info["mask"], info["stamp"], info["bg"]), pair["output"]):
                    raise AssertionError("construction and pixel interpretation disagree")
                f = _features(info)
                if split == "train" and index == 0 and not (
                        f["pooled"] and f["no_pool_changes"] and f["mismatches"] and f["blank_outer_line"]):
                    continue
                if split == "train" and index == 1:
                    if f["stamp_shape"] == train_infos[0]["stamp_shape"]:
                        continue
                    if not (f["gap_cells"] or train_infos[0]["gap_cells"]):
                        continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    train_infos.append(f)
                break
            else:
                raise RuntimeError(f"bounded construction exhausted for {split} example {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test():
    """42 streamed seeds, count boundaries, equivariances and pixel mutations."""
    features, fingerprints = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read(pair["input"])
                features.append(_features(info))
                digest.update(repr(_key(pair["input"])).encode())
                # Recoloring includes background and frame: neither role has a
                # literal color number. A whole-grid isometry rotates both the
                # coarse mask and the unchanged displayed fine stamp.
                recolor = np.roll(np.arange(10, dtype=np.int8), seed % 9+1)
                assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
                t = seed % 8
                assert np.array_equal(solve(_orient(pair["input"], t)), _orient(pair["output"], t))
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    for name, minimum in (("input_shape", 40), ("stamp_shape", 12), ("mask_shape", 20), ("directions", 5)):
        assert len({f[name] for f in features}) >= minimum, name
    assert {f["length"] for f in features} >= {3, 4, 5, 6, 7, 8}
    assert {f["background"] for f in features} == set(range(10))
    assert max(f["emitters"] for f in features) >= 4
    for name in ("pooled", "full", "redundant", "repeated_color"):
        assert sum(f[name] for f in features) >= 5, name
    assert sum(f["gap_cells"] > 0 for f in features) >= 20
    assert sum(f["blank_outer_line"] for f in features) >= 42
    assert all(f["mismatches"] > 0 for f in features)
    assert sum(f["reversal_changes"] for f in features) >= 100
    p = generate_puzzle(2026, 2, 1)
    assert _same(p, generate_puzzle(2026, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(100+nt*10+ne, nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
