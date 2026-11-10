"""Coordinate stamps carve a noisy field; highlight its largest blank window.

No parent modules or stored examples are imported. All semantics are visible:
background at the blank stage margin, highlight at the four frame corners,
odd mask centers, paired coordinates, and earlier-source overlap priority.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [next(iter(cells))]
    seen = set(todo)
    while todo:
        r, c = todo.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r + dr, c + dc
                if p in cells and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return cells == seen


def _largest(mask):
    """All global-area winners, including thin rectangles; exclusive bounds.

    For every row span, the widest all-True column runs suffice: a narrower
    run cannot be a global-area winner. No minimum-side filter hides rivals.
    """
    h, w = mask.shape
    best, winners = 0, set()
    for top in range(h):
        columns = np.ones(w, dtype=bool)
        for bottom in range(top + 1, h + 1):
            columns &= mask[bottom - 1]
            padded = np.concatenate(([False], columns, [False])).astype(np.int8)
            changes = np.flatnonzero(np.diff(padded))
            for left, right in zip(changes[::2], changes[1::2]):
                area = (bottom - top) * int(right - left)
                rectangle = (top, bottom, int(left), int(right))
                if area > best:
                    best, winners = area, {rectangle}
                elif area == best:
                    winners.add(rectangle)
    return best, sorted(winners)


def _histogram_largest(mask):
    """Different O(HW) search, used only as a bounded self-test cross-check."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best, winners = 0, set()
    for row in range(mask.shape[0]):
        heights = (heights + 1) * mask[row]
        stack = []
        for col in range(mask.shape[1] + 1):
            height = int(heights[col]) if col < mask.shape[1] else 0
            start = col
            while stack and stack[-1][1] > height:
                left, old_height = stack.pop()
                area = old_height * (col - left)
                rectangle = (row + 1 - old_height, row + 1, left, col)
                if area > best:
                    best, winners = area, {rectangle}
                elif area == best and area:
                    winners.add(rectangle)
                start = left
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    return best, sorted(winners)


def _inside(shape, rectangle):
    t, b, l, r = rectangle
    result = np.zeros(shape, dtype=bool)
    result[t + 1:b - 1, l + 1:r - 1] = True
    return result


def _parse(a):
    h, w = a.shape
    if h < 7 or w < 10:
        raise ValueError("scene too small")
    bg = int(a[0, 0])
    dividers = [c for c in range(2, w - 5)
                if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    if len(dividers) != 1:
        raise ValueError("need one constant nonbackground divider")
    d = dividers[0]
    divider = int(a[0, d])
    highlight = int(a[0, d + 1])
    frame = int(a[0, d + 2])  # First field column is reserved, never a label.
    if len({bg, divider, highlight, frame}) != 4:
        raise ValueError("structural and palette-key roles must differ")
    if not np.all(a[0, :d] == bg) or not np.all(a[-1, :d] == bg):
        raise ValueError("stage top and bottom margins must be blank")
    if not all(a[r, c] == highlight for r, c in ((0, d+1), (0, w-1), (h-1, d+1), (h-1, w-1))):
        raise ValueError("four equal corners encode the highlight")
    stage = a[:, :d]
    template_colors = set(map(int, np.unique(stage))) - {bg}
    if template_colors & {divider, frame, highlight} or not template_colors:
        raise ValueError("invalid template colors")
    if not np.array_equal(a[0, d+2:-1], a[-1, d+2:-1]):
        raise ValueError("column labels must be paired")
    if not np.array_equal(a[1:-1, d+1], a[1:-1, -1]):
        raise ValueError("row labels must be paired")
    edge = np.concatenate((a[0, d+2:-1], a[1:-1, d+1]))
    if not set(map(int, edge)) <= template_colors | {frame}:
        raise ValueError("unknown coordinate label")
    field = a[1:-1, d+2:-1]
    if set(map(int, np.unique(field))) & (template_colors | {divider, frame, highlight}):
        raise ValueError("initial texture and structural/template colors must differ")
    templates = []
    for color in template_colors:
        cells = np.argwhere(stage == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        shape = tuple(map(int, hi - lo + 1))
        if not all(n % 2 for n in shape):
            raise ValueError("template boxes need exact odd centers")
        mask = stage[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        if not _connected(mask):
            raise ValueError("one 8-connected template per color is required")
        rows = list(map(int, np.flatnonzero(a[1:-1, d+1] == color)))
        cols = list(map(int, np.flatnonzero(a[0, d+2:-1] == color)))
        offsets = cells - (lo + hi) // 2
        stamps = set()
        boxes = set()
        if rows and cols:
            for r in rows:
                for c in cols:
                    for dr, dc in offsets:
                        y, x = r + int(dr), c + int(dc)
                        if not (0 <= y < field.shape[0] and 0 <= x < field.shape[1]):
                            raise ValueError("stamps cannot clip or cross the frame")
                        stamps.add((y, x))
                    for dr in range(-(shape[0]//2), shape[0]//2+1):
                        for dc in range(-(shape[1]//2), shape[1]//2+1):
                            boxes.add((r + dr, c + dc))
        templates.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                              shape=shape, mask=mask, rows=rows, cols=cols,
                              offsets=offsets, stamps=stamps, boxes=boxes))
    templates.sort(key=lambda t: t["lo"])
    if len({t["lo"][0] for t in templates}) != len(templates):
        raise ValueError("template top rows must distinguish source order")
    return dict(bg=bg, divider=divider, frame=frame, highlight=highlight, d=d,
                field=field.copy(), templates=templates)


def _stamp(a, p, *, order="source", solid=False, erase_boxes=False, single=False, zip_axes=False):
    """Input-only forward stamping; texture is overwritten, holes stay intact."""
    out = a.copy()
    field = out[1:-1, p["d"]+2:-1]
    active = [t for t in p["templates"] if t["rows"] and t["cols"]]
    for t in active:
        if erase_boxes:
            r, c = t["lo"]
            h, w = t["shape"]
            out[r:r+h, c:c+w] = p["bg"]
        else:
            out[t["cells"][:, 0], t["cells"][:, 1]] = p["bg"]
    if order == "reverse":
        active = list(reversed(active))
    elif order == "destination":
        active = sorted(active, key=lambda t: (min(t["rows"]), min(t["cols"])))
    elif order == "size":
        active = sorted(active, key=lambda t: t["shape"][0] * t["shape"][1])
    claimed = set()
    for t in active:
        cells = t["boxes"] if solid else t["stamps"]
        if single or zip_axes:
            axes = ([(t["rows"][0], t["cols"][0])] if single
                    else list(zip(t["rows"], t["cols"])))
            cells = {(r+int(dr), c+int(dc)) for r, c in axes for dr, dc in t["offsets"]}
        for r, c in cells - claimed:
            field[r, c] = t["color"]
        claimed |= cells
    return out


def _finish(stamped, p):
    field = stamped[1:-1, p["d"]+2:-1]
    area, rectangles = _largest(field == p["bg"])
    if len(rectangles) != 1:
        raise ValueError("largest post-stamp background rectangle must be unique")
    t, b, l, r = rectangles[0]
    if min(b-t, r-l) < 3:
        raise ValueError("largest rectangle needs a nonempty strict interior")
    out = stamped.copy()
    out[1+t+1:1+b-1, p["d"]+2+l+1:p["d"]+2+r-1] = p["highlight"]
    return out, rectangles[0], area


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full result only from the current input; never mutate it."""
    a = _grid(input_grid)
    p = _parse(a)
    out, _, _ = _finish(_stamp(a, p), p)
    return out


def _mask(rng, h, w):
    mode = int(rng.integers(7))
    a = np.zeros((h, w), dtype=bool)
    if min(h, w) == 1 or mode == 0:
        a[:] = True
    elif mode == 1:
        a[[0, -1], :] = True
        a[:, [0, -1]] = True
    elif mode == 2:
        for r in range(h):
            a[r, round(r*(w-1)/(h-1))] = True
        for c in range(w):
            a[round(c*(h-1)/(w-1)), c] = True
    elif mode == 3:
        a[h//2, :] = True
        a[:, w//2] = True
    elif mode == 4 and h == w:
        rr, cc = np.indices((h, w))
        a = abs(rr-h//2) + abs(cc-w//2) == h//2
    else:
        # Random 8-neighbor growth from paths joining all four extremes.
        anchors = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                   (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        r, c = anchors[0]
        a[r, c] = True
        for rr, cc in anchors[1:]:
            while (r, c) != (rr, cc):
                r += int(np.sign(rr-r))
                c += int(np.sign(cc-c))
                a[r, c] = True
        for _ in range(int(rng.integers(h*w))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            if a[max(0, r-1):r+2, max(0, c-1):c+2].any():
                a[r, c] = True
    if rng.random() < .5:
        a = a[:, ::-1]
    if rng.random() < .5:
        a = a[::-1]
    if h == w:
        a = np.rot90(a, int(rng.integers(4)))
    return a.copy()


def _place_sources(rng, ts, height, sw, priority):
    occupied = np.zeros((height, sw), dtype=bool)
    tops = set()
    for index, t in enumerate(ts):
        mask = t["mask"]
        h, w = mask.shape
        options = []
        for r in range(1, height-h):
            if r in tops or priority and index == 1 and r <= ts[0]["top"]:
                continue
            if priority and index == 0 and r > (height-h)//2:
                continue
            for c in range(sw-w+1):
                if not np.any(occupied[r:r+h, c:c+w] & mask):
                    options.append((r, c))
        if not options:
            return False
        r, c = options[int(rng.integers(len(options)))]
        t["top"], t["left"] = r, c
        tops.add(r)
        occupied[r:r+h, c:c+w] |= mask
    return True


def _axes(rng, ts, fh, fw, reservoir, shift, priority, idle_axis, kind):
    t, b, l, r = reservoir
    used_rows, used_cols = set(), set()
    if priority:
        # Earlier, larger source lands southeast of the smaller later source.
        rr = int(rng.integers(max(2, t+1), min(fh-3, b-2)))
        cc = int(rng.integers(max(2, l+1), min(fw-3, r-2)))
        ts[0]["rows"], ts[0]["cols"] = [rr+1], [cc+1]
        ts[1]["rows"], ts[1]["cols"] = [rr], [cc]
        used_rows |= {rr, rr+1}
        used_cols |= {cc, cc+1}
    for i, template in enumerate(ts):
        if not template["active"]:
            continue
        if i == 0:
            desired = (3, int(rng.choice([1, 2, 3]))) if shift else (2, 2)
            if shift and rng.random() < .5:
                desired = desired[::-1]
        else:
            desired = tuple(map(int, rng.choice([1, 2], size=2, p=[.8, .2])))
        for axis, (length, name, used, low, high) in enumerate(
                ((fh, "rows", used_rows, t, b), (fw, "cols", used_cols, l, r))):
            rad = template["mask"].shape[axis]//2
            # Full bounding boxes fit; column zero is reserved on the border.
            choices = [v for v in range(max(rad, 1 if axis else 0), length-rad) if v not in used]
            if not choices:
                return False
            if not template.get(name):
                near = [v for v in choices if low <= v < high]
                value = int(rng.choice(near or choices))
                template[name] = [value]
                used.add(value)
            desired_count = desired[axis]
            choices = [v for v in choices if v not in used]
            rng.shuffle(choices)
            # Mixed clustered and dispersed replication.
            if rng.random() < .55:
                choices.sort(key=lambda v: abs(v-template[name][0]))
            for value in choices[:desired_count-len(template[name])]:
                template[name].append(value)
                used.add(value)
            template[name].sort()
    for template in ts:
        if template["active"]:
            continue
        state = ("N" if kind == "cartesian" else idle_axis if kind == "priority"
                 else str(rng.choice(["N", idle_axis])))
        template["state"] = state
        if state == "N":
            continue
        name, length, used = ("rows", fh, used_rows) if state == "R" else ("cols", fw, used_cols)
        choices = [v for v in range(1, length) if v not in used]
        if not choices:
            return False
        value = int(rng.choice(choices))
        template[name] = [value]
        used.add(value)
    return True


def _assemble(rng, ts, field, sw, bg, highlight):
    fh, fw = field.shape
    h, w = fh+2, sw+fw+3
    rest = [int(c) for c in rng.permutation(10) if c not in (bg, highlight)]
    divider, frame = rest[:2]
    colors = rest[2:2+len(ts)]
    noise = rest[2+len(ts):]
    if not noise:
        raise ValueError("palette needs at least one texture color")
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[:, sw] = divider
    inp[[0, -1], sw+1:] = frame
    inp[:, [sw+1, -1]] = frame
    for rr, cc in ((0, sw+1), (0, w-1), (h-1, sw+1), (h-1, w-1)):
        inp[rr, cc] = highlight
    # Field values 0=background, 1/2=texture roles, not literal palette values.
    translated = np.full(field.shape, bg, dtype=np.int8)
    for i in (1, 2):
        translated[field == i] = noise[(i-1) % len(noise)]
    inp[1:-1, sw+2:-1] = translated
    for template, color in zip(ts, colors):
        template["color"] = color
        hh, ww = template["mask"].shape
        view = inp[template["top"]:template["top"]+hh, template["left"]:template["left"]+ww]
        view[template["mask"]] = color
        for row in template.get("rows", []):
            inp[row+1, sw+1] = inp[row+1, -1] = color
        for col in template.get("cols", []):
            inp[0, sw+2+col] = inp[-1, sw+2+col] = color
    # Expected output uses construction records and reverse-order painting,
    # independent of the parser's forward occupancy traversal.
    out = inp.copy()
    for template in ts:
        if template["active"]:
            hh, ww = template["mask"].shape
            view = out[template["top"]:template["top"]+hh, template["left"]:template["left"]+ww]
            view[template["mask"]] = bg
    target = out[1:-1, sw+2:-1]
    ordered = sorted(ts, key=lambda z: (z["top"], z["left"]))
    for template in reversed(ordered):
        if not template["active"]:
            continue
        hh, ww = template["mask"].shape
        for rr in template["rows"]:
            for cc in template["cols"]:
                view = target[rr-hh//2:rr+hh//2+1, cc-ww//2:cc+ww//2+1]
                view[template["mask"]] = template["color"]
    area, rectangles = _largest(target == bg)
    if len(rectangles) != 1:
        return None
    t, b, l, r = rectangles[0]
    if min(b-t, r-l) < 3 or area < 12:
        return None
    target[t+1:b-1, l+1:r-1] = highlight
    return {"input": inp, "output": out}


def _pair_features(pair, detailed=False):
    a = _grid(pair["input"])
    p = _parse(a)
    stamped = _stamp(a, p)
    out, rectangle, area = _finish(stamped, p)
    _require(np.array_equal(out, _grid(pair["output"])), "incorrect composed answer")
    after = stamped[1:-1, p["d"]+2:-1]
    before_area, before_rectangles = _largest(p["field"] == p["bg"])
    # Every generated example has a real causal effect on the selected window.
    _require(len(before_rectangles) == 1 and before_rectangles[0] != rectangle,
             "stamping must change the unique winning rectangle")
    _require(before_area > area, "stamps must reduce the previous winning vacancy")
    _require(area >= 12, "need a legible two-dimensional highlight")
    interior = _inside(after.shape, rectangle)
    result_field = out[1:-1, p["d"]+2:-1]
    _require(np.all(result_field[interior] == p["highlight"]), "highlight missing")
    _require(np.array_equal(result_field[~interior], after[~interior]), "window border or texture altered")
    active = [t for t in p["templates"] if t["rows"] and t["cols"]]
    unions = [t["stamps"] for t in active]
    conflicts = sum(len(x & y) for i, x in enumerate(unions) for y in unions[i+1:])
    holes = sum(sum(after[r, c] != t["color"] for r, c in t["boxes"]-t["stamps"]) for t in active)
    f = dict(shape=a.shape, strip=p["d"], rectangle=rectangle, area=area,
             before_area=before_area, bg=p["bg"], highlight=p["highlight"],
             count=len(p["templates"]), active=len(active), conflicts=conflicts, holes=holes,
             empty_centers=sum(not t["mask"][t["shape"][0]//2, t["shape"][1]//2]
                               for t in active),
             multiplicities=tuple((len(t["rows"]), len(t["cols"])) for t in active),
             states="".join("A" if t["rows"] and t["cols"] else "R" if t["rows"]
                            else "C" if t["cols"] else "N" for t in p["templates"]),
             masks=tuple((t["shape"], t["mask"].tobytes()) for t in active),
             priority_difference=int(np.count_nonzero(stamped != _stamp(a, p, order="reverse"))),
             destination_difference=int(np.count_nonzero(stamped != _stamp(a, p, order="destination"))),
             size_difference=int(np.count_nonzero(stamped != _stamp(a, p, order="size"))),
             erase_difference=int(np.count_nonzero(stamped != _stamp(a, p, erase_boxes=True))),
             solid_difference=int(np.count_nonzero(stamped != _stamp(a, p, solid=True))),
             zip_difference=int(np.count_nonzero(stamped != _stamp(a, p, zip_axes=True))))
    if detailed:
        f.update(p=p, stamped=stamped, after=after, interior=interior)
    return f


def _make_pair(rng, bg, highlight, idle_axis, kind, shift):
    priority = kind == "priority"
    for attempt in range(220):
        fh = int(rng.integers(13 if priority else 11, 27 if shift else 24))
        fw = int(rng.integers(12, 22))
        if priority:
            # Larger earlier mask plus a small later mask. Random mask growth
            # and filtering produce transparent, priority-distinguishing cases.
            large_shape = [(5, 5), (5, 3), (3, 5)][int(rng.integers(3))]
            masks = [_mask(rng, *large_shape), _mask(rng, 3, 3)]
            active_flags = [True, True]
        else:
            nactive = 1 if kind == "cartesian" else int(rng.integers(1, 4))
            sizes = [(3, 3), (3, 5), (5, 3), (5, 5), (1, 3), (3, 1), (1, 5), (5, 1)]
            masks = [_mask(rng, *sizes[int(rng.integers(len(sizes)))]) for _ in range(nactive)]
            active_flags = [True]*nactive
        # Both first training witnesses include an inert source; later scenes
        # may be all-active, so staging position is never an eligibility cue.
        if kind in ("cartesian", "priority") or rng.random() < .5:
            masks.append(_mask(rng, *[(1, 3), (3, 1), (3, 3)][int(rng.integers(3))]))
            active_flags.append(False)
        ts = [dict(mask=m, active=flag, rows=[], cols=[]) for m, flag in zip(masks, active_flags)]
        sw = min(7, max(m.shape[1] for m in masks)+int(rng.integers(0, 3)))
        sw = max(3, sw)
        fw = min(fw, 27-sw)
        if not _place_sources(rng, ts, fh+2, sw, priority):
            continue
        rh = int(rng.integers(6, min(12, fh)+1))
        rw = int(rng.integers(6, min(13, fw)+1))
        top = int(rng.integers(fh-rh+1))
        left = int(rng.integers(fw-rw+1))
        reservoir = top, top+rh, left, left+rw
        if not _axes(rng, ts, fh, fw, reservoir, shift, priority, idle_axis, kind):
            continue
        # Dense irregular texture plus a blank reservoir. No clean external
        # halo is drawn. Side-touching blanks and rival runs are allowed.
        density = float(rng.uniform(.23, .55))
        field = (rng.random((fh, fw)) >= density).astype(np.int8)
        if rng.random() < .75:
            field[(field == 1) & (rng.random(field.shape) < rng.uniform(.08, .4))] = 2
        if rng.random() < .5:
            hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 6))
            rr, cc = int(rng.integers(fh-hh+1)), int(rng.integers(fw-ww+1))
            field[rr:rr+hh, cc:cc+ww] = 0
        field[top:top+rh, left:left+rw] = 0
        pair = _assemble(rng, ts, field, sw, bg, highlight)
        if pair is None:
            continue
        try:
            f = _pair_features(pair)
        except (AssertionError, ValueError):
            continue
        if kind == "priority" and not (f["priority_difference"] and f["destination_difference"]
                                       and f["size_difference"] and f["holes"]):
            continue
        if kind == "cartesian" and not (f["zip_difference"] and f["solid_difference"]):
            continue
        if shift and not any(max(m) == 3 for m in f["multiplicities"]):
            continue
        return pair
    raise RuntimeError(f"bounded construction exhausted for {kind}, shift={shift}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Deterministic family member. All palette conventions are input-visible."""
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bg, highlight = map(int, rng.choice(10, size=2, replace=False))
    idle_axis = str(rng.choice(["R", "C"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            kind = "cartesian" if split == "train" and index == 0 else (
                "priority" if split == "train" and index == 1 else "random")
            for _ in range(20):
                pair = _make_pair(rng, bg, highlight, idle_axis, kind, split == "test")
                key = _key(pair["input"])
                if key in seen:
                    continue
                if split == "test":
                    f = _pair_features(pair)
                    # A sparse source can contain a retained object's cells,
                    # and a template's geometric center can be transparent.
                    # Do not introduce either distinction only at test time.
                    if any(f[name] and not any(t[name] for t in train_features)
                           for name in ("erase_difference", "empty_centers")):
                        continue
                break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
        if split == "train":
            train_features = [_pair_features(pair) for pair in puzzle["train"]]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, visible semantics, causal witnesses and curriculum."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError("illegal counts") from exc
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            f = _pair_features(pair)
            features[split].append(f)
    all_features = features["train"]+features["test"]
    _require(len({(f["bg"], f["highlight"]) for f in all_features}) == 1, "palette roles vary unexpectedly")
    train = features["train"]
    for name in ("zip_difference", "solid_difference", "priority_difference", "destination_difference", "size_difference"):
        _require(any(f[name] for f in train), f"training lacks a {name} witness")
    _require(any("N" in f["states"] or "R" in f["states"] or "C" in f["states"] for f in train), "no eligibility witness")
    for f in features["test"]:
        _require(any(max(m) == 3 for m in f["multiplicities"]), "test needs a novel triple multiplicity")
        _require(set(f["states"]) <= set("".join(t["states"] for t in train)), "undemonstrated eligibility state")
        for name in ("erase_difference", "empty_centers"):
            _require(not f[name] or any(t[name] for t in train),
                     f"test requires an undemonstrated {name} distinction")
    _require(all(max(m) <= 2 for f in train for m in f["multiplicities"]), "training coordinate multiplicities exceed two")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, algorithm cross-checks, mutation rejection; no disk writes."""
    # Explicit search regressions include ties and a thin global-area winner.
    # Thin rivals must not be silently excluded by an interior-size filter.
    tied = np.zeros((5, 11), dtype=bool)
    tied[1:4, 0:4] = tied[1:4, 7:11] = True
    thin = np.zeros((7, 12), dtype=bool)
    thin[:2, :] = True
    thin[4:7, :3] = True
    for mask, expected in ((tied, (12, [(1, 4, 0, 4), (1, 4, 7, 11)])),
                           (thin, (24, [(0, 2, 0, 12)])),
                           (np.zeros((3, 4), dtype=bool), (0, [])),
                           (np.ones((4, 5), dtype=bool), (20, [(0, 4, 0, 5)]))):
        _require(_largest(mask) == expected, "rectangle regression")
        _require(_histogram_largest(mask) == expected, "histogram regression")
    shapes, rectangles, masks, palettes, signatures, products, counts = (set() for _ in range(7))
    erase_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed < 3:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_features(pair, detailed=True)
                a = np.asarray(pair["input"])
                original = a.copy()
                _require(np.array_equal(solve(a), pair["output"]), "input-only solve disagreement")
                _require(np.array_equal(a, original), "solve mutated its input")
                if seed == 0:
                    # No numeric color or integer dtype has a privileged role.
                    recoloring = (np.arange(10, dtype=np.int8) + 3) % 10
                    _require(np.array_equal(solve(recoloring[a].tolist()),
                                            recoloring[np.asarray(pair["output"])]),
                             "palette permutation or nested-list input failed")
                _require(_largest(f["after"] == f["bg"]) == _histogram_largest(f["after"] == f["bg"]), "rectangle algorithms disagree")
                shapes.add(f["shape"])
                t, b, l, r = f["rectangle"]
                rectangles.add((b-t, r-l))
                masks.update(f["masks"])
                palettes.add((f["bg"], f["highlight"]))
                products.update(f["multiplicities"])
                counts.add(f["count"])
                erase_witnesses += bool(f["erase_difference"])
        if seed == 0:
            # A changed highlight cell, a reverse-priority answer, and a
            # pre-stamp-window answer must all be rejected by the validator.
            for mutation in ("highlight", "priority", "pre_stamp"):
                bad = {s: [{k: np.asarray(v).copy() for k, v in x.items()} for x in puzzle[s]] for s in ("train", "test")}
                pair = bad["train"][1]
                f = _pair_features(pair, detailed=True)
                p = f["p"]
                if mutation == "highlight":
                    pair["output"][1, p["d"]] = p["bg"]
                elif mutation == "priority":
                    pair["output"], _, _ = _finish(_stamp(pair["input"], p, order="reverse"), p)
                else:
                    _, rs = _largest(p["field"] == p["bg"])
                    wrong = f["stamped"].copy()
                    view = wrong[1:-1, p["d"]+2:-1]
                    view[_inside(view.shape, rs[0])] = p["highlight"]
                    pair["output"] = wrong
                try:
                    validate_generated_puzzle(bad)
                except (AssertionError, ValueError):
                    pass
                else:
                    raise AssertionError("validator accepted a wrong answer")
    _require(len(signatures) == 42, "duplicate puzzles across seeds")
    _require(len(shapes) >= 45 and len(rectangles) >= 20 and len(masks) >= 40, "insufficient geometric diversity")
    _require(len(palettes) >= 20 and {2, 3, 4} <= counts, "insufficient roles/count variation")
    _require({(2, 2), (3, 1), (1, 3)} <= products, "missing replication variation")
    _require(erase_witnesses > 0, "no source interleaving witness")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(5700+nt*10+ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "wrong explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "rectangle_shapes": len(rectangles), "active_masks": len(masks),
            "background_highlight_roles": len(palettes), "template_counts": sorted(counts),
            "coordinate_products": sorted(products), "source_interleavings": erase_witnesses}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
